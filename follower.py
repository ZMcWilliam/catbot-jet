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
PORT_USS_TRIG = 23
PORT_USS_ECHO = 24
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
    "distance": 0,
}

debug_info = {
    "steering": 0,
    "pos": 0,
    "speeds": [0, 0],
}

itr_stats = {
    "master": {
        "count": 0,
        "time": 0,
    },
    "line": {
        "count": 0,
        "time": 0,
    },
    "cols": {
        "count": 0,
        "time": 0,
    },
    "distance": {
        "count": 0,
        "time": 0,
    }
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
    data = {
        "hsv": col_l.readHSV(),
        "hue": col_l.classifyHue({"red":0,"yellow":60,"green":120,"cyan":180,"blue":240,"magenta":300})
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
    return data["hue"] == "green" and data["hsv"]["sat"] > threshold


USS_TRIG = gpiozero.OutputDevice(PORT_USS_TRIG, active_high=True, initial_value=False)
USS_ECHO = gpiozero.InputDevice(PORT_USS_ECHO)

async def measure_distance(max_distance: float = 100) -> float:
    """
    Measures the distance using the RCWL-1601 ultrasonic sensor.

    Args:
        max_distance (float, optional): The maximum distance to measure in centimeters. Defaults to 200.

    Returns:
        float: The measured distance in centimeters.
    """
    USS_TRIG.on()
    await asyncio.sleep(0.00001)
    USS_TRIG.off()

    pulse_start = time.monotonic()
    while USS_ECHO.is_active == False:
        if time.monotonic() - pulse_start > max_distance / 17150:
            return max_distance
        pulse_start = time.monotonic()

    pulse_end = time.monotonic()
    while USS_ECHO.is_active == True:
        if time.monotonic() - pulse_end > max_distance / 17150:
            return max_distance
        pulse_end = time.monotonic()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = min(distance, max_distance)
    distance = round(distance, 2)

    latest_data["distance"] = distance
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
    debug_info["speeds"] = m.run_tank(follower_speed, 100, steering)

def monitor_line_thread() -> None:
    """
    Monitors the line in a separate thread.
    """
    global latest_data
    while True:
        update_itr_stat("line")
        read_line()
    
def monitor_col_thread() -> None:
    """
    Monitors the color sensors in a separate thread.
    """
    global latest_data
    while True:
        update_itr_stat("cols")
        read_col(PORT_COL_L)
        read_col(PORT_COL_R)

async def monitor_uss_async() -> None:
    """
    Monitors the ultrasonic sensor asynchronously.
    """
    global latest_data
    while True:
        update_itr_stat("distance")
        await measure_distance()
        await asyncio.sleep(0.02)

def monitor_uss_thread() -> None:
    """
    Monitors the ultrasonic sensor in a separate thread.
    """
    asyncio.run(monitor_uss_async())
    

line_thread = threading.Thread(target=monitor_line_thread)
line_thread.daemon = True
line_thread.start()

col_thread = threading.Thread(target=monitor_col_thread)
col_thread.daemon = True
col_thread.start()

uss_thread = threading.Thread(target=monitor_uss_thread)
uss_thread.daemon = True
uss_thread.start()

while True:
    update_itr_stat("master", 1000)

    l_green = check_col_green(PORT_COL_L)
    r_green = check_col_green(PORT_COL_R)

    follow_line()
    print(f"ITR: {get_itr_stat('master')}, {get_itr_stat('line')}, {get_itr_stat('cols')}, {get_itr_stat('distance')}\t"
        + f"GL: {l_green} ({latest_data['col_l']['hue']}, {int(latest_data['col_l']['hsv']['sat'])})\t"
        + f"GR: {r_green} ({latest_data['col_r']['hue']}, {int(latest_data['col_r']['hsv']['sat'])})\t"
        + f"USS: {latest_data['distance']}\t"
        + f"\t Pos: {int(debug_info['pos'])},"
        + f"\t Steering: {int(debug_info['steering'])},"
        + f"\t Speeds: {debug_info['speeds']},"
        + f"\t Line: {latest_data['line']['scaled']}")