#  /\_/\  
# ( o.o )
#  > ^ <

import time
import gpiozero
import asyncio
import board
import busio
import adafruit_tca9548a
import adafruit_ads1x15.ads1015 as ADS
import motorkit_helper as m
import threading
from PiicoDev_VEML6040 import PiicoDev_VEML6040
from typing import Union, List, Tuple, Dict
from adafruit_ads1x15.analog_in import AnalogIn as ADSAnalogIn

# Calibrated values for the reflectivity array
las_min = [1312, 1248, 1280, 1280, 1248, 1280, 1264, 1312, 1296, 1248, 1264, 1344] 
las_max = [10352, 9600, 10736, 10944, 8544, 11264, 10480, 11952, 10848, 8080, 9584, 12800]

# Ports for sensors
PORT_USS_TRIG = {
    "front": 23,
    "side": 27,
}
PORT_USS_ECHO = {
    "front": 24,
    "side": 22,
}
PORT_COL_L = 6
PORT_COL_R = 7
# Line array ports, in order from left to right. letter is ADS selection, number is port on ADS
PORT_ADS_LINE = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "C0", "C1", "C2", "C3"]

# Constants for PID control
KP = 0.05  # Proportional gain
KI = 0  # Integral gain
KD = 0.2  # Derivative gain
follower_speed = 40

# Variables for PID control
pid_error_sum = 0
pid_last_error = 0

obstacle_threshold = 4  # Distance threshold to detect an obstacle

col_thresholds = {
    "green": 0.55,
    "red": 0.65,
}

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
        "hue": "Unk",
        "eval": None,
    },
    "col_r": {
        "hsv": {
            "hue": 0,
            "sat": 0,
            "val": 0
        },
        "hue": "Unk",
        "eval": None,
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

i2c = busio.I2C(board.SCL, board.SDA)

ads_addresses = [0x49, 0x48, 0x4B]
ads_gain = 1
ads_mode = ADS.Mode.SINGLE

ads = []
for address in ads_addresses:
    ads_instance = ADS.ADS1015(i2c, address=address)
    ads_instance.gain = ads_gain
    ads_instance.mode = ads_mode
    ads.append(ads_instance)

ads_channels = []
for ads_port in PORT_ADS_LINE:
    ads_instance = ads[ord(ads_port[0]) - ord("A")]
    channel = ADSAnalogIn(ads_instance, getattr(ADS, f"P{ads_port[1]}"))
    ads_channels.append(channel)

tca = adafruit_tca9548a.TCA9548A(i2c)

def mx_select(port: int) -> None:
    """
    Selects the specified port on the I2C multiplexer

    Args:
        port (int): The port number to select. Must be between 0 and 7.
    """
    if port < 0 or port > 7:
        raise ValueError("Port must be between 0 and 7")
    
    i2c.writeto(0x70, bytes([1 << port]))

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
        "hue": selected_sensor.classifyHue({"red":0,"yellow":60,"green":120,"cyan":180,"blue":240,"magenta":300}),
        "eval": None,
    }

    for col_name, col_threshold in col_thresholds.items():
        if data["hue"] == col_name and data["hsv"]["sat"] >= col_threshold:
            data["eval"] = col_name

    latest_data["col_l" if port == PORT_COL_L else "col_r"] = data
    return data

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
    Reads reflectivity data from the line array (ADS1015 channels) and scales it.

    Returns:
        List[List[float]]: A list containing two sublists:
            - The raw reflectivity data from ADS1015 channels.
            - The scaled reflectivity data between 0 and 100.
    """
    data = [asd_channel.value for asd_channel in ads_channels]
    data_scaled = [0] * len(data)

    for i in range(len(data)):
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


def avoid_obstacle() -> None:
    """
    Performs obstacle avoidance when an obstacle is detected.
    """
    obstacle_maintain_distance = 9  # Desired distance to maintain from the obstacle

    # Step 1: Stop the robot
    m.stop_all()
    time.sleep(0.3)
    m.run_tank_for_time(-50, -50, 200)
    time.sleep(0.2)

    # Step 2: Rotate until the side ultrasonic sensor detects the obstacle
    while True:
        distance = latest_data["distance_side"]
        print("Aligning obstacle on side: " + str(distance) + " cm")
        if distance > obstacle_threshold + 8:
            m.run_tank(-50, 50)  # Rotate left
        else:
            m.stop_all()
            time.sleep(0.1)
            m.run_tank_for_time(-50, 50, 300)
            break

    time.sleep(1)

    lost_track_counter = 0

    # Step 3: Move around the obstacle while maintaining a constant distance
    while True:
        # Calculate the difference between the current distance and the desired distance
        distance_diff = round(latest_data["distance_side"] - obstacle_maintain_distance, 2)

        # print("Distance diff: " + str(distance_diff) + " cm")

        # Adjust the robot's position and orientation to maintain the desired distance
        if distance_diff > 35: # Robot has lost track of the obstacle
            print(f"A    {distance_diff}    LOST TRACK")
            lost_track_counter += 1

            if lost_track_counter > 20:
                m.run_tank(40, -40)
            else:
                m.run_tank(-40, 40)
        else:
            if lost_track_counter > 20:
                m.run_tank_for_time(40, -40, 100)
            
            lost_track_counter = 0

            if distance_diff > 5:  # Robot is too far from the obstacle
                print(f"BB   {distance_diff}")
                m.run_tank(100, -30)
            elif distance_diff > 2:
                print(f"EEEEE {distance_diff}")
                m.run_tank(100, 0)
            elif distance_diff > -2:  # Robot is at the desired distance
                print(f"CCC  {distance_diff}")
                m.run_tank(100, 15)
            elif distance_diff <= -2:  # Robot is too close to the obstacle
                print(f"DDDD {distance_diff}")
                m.run_tank(100, 40)
            elif distance_diff <= -4:
                print(f"FFFFFF {distance_diff}")
                m.run_tank(100, 60)

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
    try: 
        update_itr_stat("master", 1000)

        # If both sensors detect red, we have reached the end of the line, stop, and make really sure we have reached the end
        if latest_data["col_l"]["eval"] == "red" and latest_data["col_r"]["eval"] == "red":
            print("End of line reached")
            m.stop_all()
            time.sleep(1)
            if latest_data["col_l"]["eval"] == "red" and latest_data["col_r"]["eval"] == "red":
                print("End of line confirmed")
                break

        if latest_data["distance_front"] < obstacle_threshold and latest_data["distance_front"] > 0:
            print(f"Obstacle detected at {latest_data['distance_front']} cm")
            avoid_obstacle()
        else:
            follow_line()

        print(f"ITR: M{get_itr_stat('master')}, L{get_itr_stat('line')}, C{get_itr_stat('cols')}, U{get_itr_stat('distance_front'), get_itr_stat('distance_side')}"
            + f"\t L: {latest_data['col_l']['eval']} ({latest_data['col_l']['hue']}, {round(latest_data['col_l']['hsv']['sat'], 2)})"
            + f"\t R: {latest_data['col_r']['eval']} ({latest_data['col_r']['hue']}, {round(latest_data['col_r']['hsv']['sat'], 2)})"
            + f"\t USS: {latest_data['distance_front']}, {latest_data['distance_side']}"
            + f"\t Pos: {int(debug_info['pos'])},"
            + f"\t Steering: {int(debug_info['steering'])},"
            + f"\t Speeds: {debug_info['speeds']},"
            + f"\t Line: {latest_data['line']['scaled']}")
    except KeyboardInterrupt:
        print("Stopping Bot")
        break

m.stop_all()
