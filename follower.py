#  /\_/\  
# ( o.o )
#  > ^ <

import time
import gpiozero
import asyncio
import board
import adafruit_tca9548a
import motorkit_helper as m
import threading
from PiicoDev_VEML6040 import PiicoDev_VEML6040
from ADCPi import ADCPi
from typing import Union, List, Tuple, Dict

# Calibrated values for the reflectivity array
las_min = [68, 65, 68, 66, 66, 64, 67, 66]
las_max = [141, 113, 160, 142, 157, 124, 149, 108]

# Ports for sensors
PORT_USS_TRIG = {
    "front": 23,
    "side": 27,
}
PORT_USS_ECHO = {
    "front": 24,
    "side": 22,
}
PORT_COL_L = 5
PORT_COL_R = 4
PORT_LINE = [8, 7, 6, 5, 4, 3, 2, 1] # ADC Ports, left to right when robot is facing forward

# Constants for PID control
KP = 0.13  # Proportional gain
KI = 0  # Integral gain
KD = 0.10  # Derivative gain
follower_speed = 50

# Variables for PID control
pid_error_sum = 0
pid_last_error = 0

# Constants for obstacle avoidance
obstacle_threshold = 10  # Distance threshold to detect an obstacle
obstacle_distance = 10  # Desired distance to maintain from the obstacle

start_time = time.time()

latest_data = {
    "line": {
        "raw": [0] * 8,
        "scaled": [0] * 8,
    },
    "col_l": {
        "hsv": {
            "hue": 0,
            "sat": 0,
            "val": 0
        },
        "hue": "Unk"
    },
    "col_r": {
        "hsv": {
            "hue": 0,
            "sat": 0,
            "val": 0
        },
        "hue": "Unk"
    },
    "distance_front": 0,
    "distance_side": 0,
}

debug_info = {
    "steering": 0,
    "pos": 0,
    "speeds": [0, 0],
}

itr_stats = {
    name: {"count": 0, "time": 0, "paused": False}
    for name in ["master", "line", "cols", "distance_front", "distance_side"]
}

def update_itr_stat(stat: str, auto_reset: int = False) -> None:
    """
    Updates the specified iteration stat.

    Args:
        stat (str): The stat to update.
        auto_reset (int, optional): The number of iterations before the stat is reset. Defaults to False.
    """
    if itr_stats[stat]["time"] == 0:
        itr_stats[stat]["time"] = time.time()
    
    if auto_reset != False and itr_stats[stat]["count"] % auto_reset == 0:
        itr_stats[stat]["time"] = time.time()
        itr_stats[stat]["count"] = 0
    
    itr_stats[stat]["count"] += 1

def get_itr_stat(stat: str, decimals: int = 0) -> float:
    """
    Gets the current iteration stat, in iterations per second

    Args:
        stat (str): The stat to retieve.
        decimals (int, optional): The number of decimal places to round to. Defaults to 0.

    Returns:
        float: The stat, in iterations per second.
    """
    val = itr_stats[stat]["count"] / (time.time() - itr_stats[stat]["time"])
    return round(val, decimals) if decimals > 0 else int(val)

adc = ADCPi(0x68, 0x69, 12)
i2c = board.I2C()
tca = adafruit_tca9548a.TCA9548A(i2c)

mx_locks = [False] * 8
def mx_remove_locks() -> None:
    """
    Releases the locks on the I2C multiplexer channels.
    """
    for i in range(8):
        if mx_locks[i] == True:
            tca[i].unlock()
            mx_locks[i] = False

def mx_select(port: int) -> None:
    """
    Selects the specified port on the I2C multiplexer and locks it.

    Args:
        port (int): The port number to select. Must be between 0 and 7.
    """
    mx_remove_locks()
    tca[port].try_lock()
    mx_locks[port] = True

mx_select(PORT_COL_L)
col_l = PiicoDev_VEML6040()

mx_select(PORT_COL_R)
col_r = PiicoDev_VEML6040()

def read_col(port: int) -> Dict[str, Union[Tuple[float, float, float], str]]:
    """
    Reads color data from the specified multiplexer port's color sensor.

    Args:
        port (int): The port number of the color sensor.

    Returns:
        dict: A dictionary containing color data with the following keys:
            - "hsv" (tuple): Hue, saturation, and value values of the color.
            - "hue" (str): The classified hue of the color.
    """
    mx_select(port)
    selected_sensor = col_l if port == PORT_COL_L else col_r
    data = {
        "hsv": selected_sensor.readHSV(),
        "hue": selected_sensor.classifyHue({"red":0,"yellow":60,"green":120,"cyan":180,"blue":240,"magenta":300})
    }
    mx_remove_locks()
    latest_data["col_l" if port == PORT_COL_L else "col_r"] = data
    return data

def check_col_green(port: int, threshold: float = 0.5) -> bool:
    """
    Checks if the color sensor on the specified port is detecting green.

    Args:
        port (int): The port number of the color sensor.
        threshold (float, optional): The threshold for the green saturation value. Defaults to 0.5.

    Returns:
        bool: True if the color sensor is detecting green, False otherwise.
    """
    data = latest_data["col_l" if port == PORT_COL_L else "col_r"]
    return data["hue"] == "green" and data["hsv"]["sat"] >= threshold


USS_TRIG = { device: gpiozero.OutputDevice(device_pin, active_high=True, initial_value=False) for device, device_pin in PORT_USS_TRIG.items() }
USS_ECHO = { device: gpiozero.InputDevice(device_pin) for device, device_pin in PORT_USS_ECHO.items() }

async def measure_distance(device: str = "front", max_distance: float = 100) -> float:
    """
    Measures the distance using the RCWL-1601 ultrasonic sensor.

    Args:
        device (str, optional): The device to measure the distance from. Defaults to "front".
        max_distance (float, optional): The maximum distance to measure in centimeters. Defaults to 200.

    Returns:
        float: The measured distance in centimeters.
    """
    USS_TRIG[device].on()
    await asyncio.sleep(0.00001)
    USS_TRIG[device].off()

    pulse_start = time.monotonic()
    while USS_ECHO[device].is_active == False:
        if time.monotonic() - pulse_start > max_distance / 17150:
            return max_distance
        pulse_start = time.monotonic()

    pulse_end = time.monotonic()
    while USS_ECHO[device].is_active == True:
        if time.monotonic() - pulse_end > max_distance / 17150:
            return max_distance
        pulse_end = time.monotonic()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = min(distance, max_distance)
    distance = round(distance, 2)

    latest_data["distance_" + device] = distance
    return distance

def read_line() -> List[List[float]]:
    """
    Reads reflectivity data from the ADCPi channels and scales it.

    Returns:
        List[List[float]]: A list containing two sublists:
            - The raw reflectivity data from the ADCPi channels.
            - The scaled reflectivity data between 0 and 100.
    """
    data = [adc.read_raw(i) for i in PORT_LINE]
    data_scaled = [0] * 8

    for i in range(8):
        data_scaled[i] = (data[i] - las_min[i]) / (las_max[i] - las_min[i])
        data_scaled[i] = round(max(0, min(100, data_scaled[i] * 100)), 1)

    latest_data["line"]["raw"] = data
    latest_data["line"]["scaled"] = data_scaled
    return [data, data_scaled]

last_pos_value = 0
def calculate_position(values: List[float], invert: int = False) -> float:
    """
    Calculates the position on a line based on given reflectivity sensor values.

    Args:
        values (List[float]): List of reflectivity sensor values.
        invert (bool, optional): Flag indicating whether to invert the reflectivity values. Defaults to False.

    Returns:
        float: The calculated position on the line

    Note:
        The position is calculated by taking the weighted average of the reflectivity values.
        
        A global variable, last_pos_value, is used to store the last calculated position in order to provide
        a more accurate position when the robot is not on the line.
    """
    global last_pos_value
    on_line = False
    avg = 0
    sum_values = 0

    for i in range(len(values)):
        value = values[i]
        if invert:
            value = 100 - value
        
        if value > 10:
            on_line = True

        if value > 8:
            avg += value * (i * 1000)
            sum_values += value

    if not on_line:
        if last_pos_value < ((len(values) - 1) * 1000) / 2:
            return 0
        else:
            return (len(values) - 1) * 1000

    last_pos_value = avg / sum_values
    debug_info["pos"] = last_pos_value
    return last_pos_value

def follow_line() -> None:
    """
    Follows a line using PID control.
    """
    global pid_error_sum, pid_last_error

    pos = calculate_position(latest_data["line"]["scaled"])
    error = pos - 3500

    # Update the error sum and limit it within a reasonable range
    pid_error_sum += error
    pid_error_sum = max(-100, min(100, pid_error_sum))

    # Calculate the change in error for derivative control
    error_diff = error - pid_last_error
    pid_last_error = error

    # Calculate the steering value using PID control
    steering = KP * error + KI * pid_error_sum + KD * error_diff

    debug_info["steering"] = steering
    debug_info["speeds"] = m.run_steer(follower_speed, 100, steering)

class Monitor:
    """
    A class that monitors a function in a separate thread.

    Args:
        loop_function (function): The function to monitor.
        itr_stat (str): The name of the iteration stat to update.
        timeout (float): The time to wait between each iteration of the loop.
        is_async (bool): Whether the function is a coroutine and should be awaited.
    """
    def __init__(self, loop_function, itr_stat, timeout=0, is_async=False):
        self.loop_function = loop_function
        self.itr_stat = itr_stat
        self.timeout = timeout
        self.is_async = is_async

        self.paused = False
        self.pause_event = threading.Event()

        self.thread = threading.Thread(target=self.run_loop)
        self.thread.daemon = True
        self.thread.start()

    def run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self._monitor_loop())

    async def _monitor_loop(self):
        while True:
            if not self.paused:
                update_itr_stat(self.itr_stat)

                # Await function if it's async, otherwise just call it
                # Bear in mind, we may be using a lambda function which includes an async function
                if asyncio.iscoroutinefunction(self.loop_function) or self.is_async:
                    await self.loop_function()
                else:
                    self.loop_function()

                if self.timeout > 0:
                    await asyncio.sleep(self.timeout)
            else:
                self.pause_event.wait()

    def pause(self):
        itr_stats[self.itr_stat]["paused"] = True
        self.paused = True

    def resume(self):
        self.paused = False
        self.pause_event.set()
        self.pause_event.clear()
        itr_stats[self.itr_stat]["paused"] = False
        update_itr_stat(self.itr_stat, 1) # Reset the iteration counter as the loop was paused

    def stop(self):
        self.thread.stop()

line_monitor = Monitor(read_line, "line")
cols_monitor = Monitor(lambda: (read_col(PORT_COL_L), read_col(PORT_COL_R)), "cols")
uss_front_monitor = Monitor(lambda: measure_distance("front"), "distance_front", timeout=0.02, is_async=True)
uss_side_monitor = Monitor(lambda: measure_distance("side"), "distance_side", timeout=0.02, is_async=True)

print("Starting Bot")
while True:
    update_itr_stat("master", 1000)

    l_green = check_col_green(PORT_COL_L)
    r_green = check_col_green(PORT_COL_R)

    follow_line()

    print(f"ITR: {get_itr_stat('master')}, {get_itr_stat('line')}, {get_itr_stat('cols')}, {get_itr_stat('distance_front'), get_itr_stat('distance_side')}"
        + f"\t GL: {l_green} ({latest_data['col_l']['hue']}, {round(latest_data['col_l']['hsv']['sat'], 2)})"
        + f"\t GR: {r_green} ({latest_data['col_r']['hue']}, {round(latest_data['col_r']['hsv']['sat'], 2)})"
        + f"\t USS: {latest_data['distance_front']}, {latest_data['distance_side']}"
        + f"\t Pos: {int(debug_info['pos'])},"
        + f"\t Steering: {int(debug_info['steering'])},"
        + f"\t Speeds: {debug_info['speeds']},"
        + f"\t Line: {latest_data['line']['scaled']}")