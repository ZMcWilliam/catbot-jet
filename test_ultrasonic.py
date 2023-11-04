import asyncio
import time
import gpiozero

# Set pin numbers
trigger_pin = 23
echo_pin = 24

# Set up GPIO pins
trigger = gpiozero.OutputDevice(trigger_pin, active_high=True, initial_value=False)
echo = gpiozero.InputDevice(echo_pin)

async def measure_distance(max_distance: float = 100) -> float:
    """
    Measures the distance using the RCWL-1601 ultrasonic sensor.

    Args:
        max_distance (float, optional): The maximum distance to measure in centimeters. Defaults to 200.

    Returns:
        float: The measured distance in centimeters.
    """
    trigger.on()
    await asyncio.sleep(0.00001)
    trigger.off()

    pulse_start = time.monotonic()
    while not echo.is_active:
        if time.monotonic() - pulse_start > max_distance / 17150:
            return max_distance
        pulse_start = time.monotonic()

    pulse_end = time.monotonic()
    while echo.is_active:
        if time.monotonic() - pulse_end > max_distance / 17150:
            return max_distance
        pulse_end = time.monotonic()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = min(distance, max_distance)
    distance = round(distance, 2)

    return distance

async def main():
    while True:
        dist = await measure_distance()
        print("Distance:", dist, "cm")
        await asyncio.sleep(0.01)

asyncio.run(main())
