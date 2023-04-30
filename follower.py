#  /\_/\  
# ( o.o )
#  > ^ <

import time
import board
import adafruit_tca9548a
import motorkit_helper as m
from PiicoDev_VEML6040 import PiicoDev_VEML6040
from ADCPi import ADCPi
from typing import Union, List, Tuple, Dict

# Calibrated values for the reflectivity array
las_min = [68, 65, 68, 66, 66, 64, 67, 66]
las_max = [141, 113, 160, 142, 157, 124, 149, 108]

# Ports for sensors
PORT_COL_L = 5
PORT_COL_R = 4
PORT_LINE = [8, 7, 6, 5, 4, 3, 2, 1] # ADC Ports, left to right when robot is facing forward

# Constants for PID control
KP = 0.05  # Proportional gain
KI = 0  # Integral gain
KD = 0.01  # Derivative gain
follower_speed = 40

# Variables for PID control
pid_error_sum = 0
pid_last_error = 0

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
    return data

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
    return last_pos_value

while True:
    l = read_col(PORT_COL_L)
    r = read_col(PORT_COL_R)
    line = read_line()

    pos = calculate_position(line[1])

    # Calculate the error as the difference between the desired position (3.5) and the weighted average
    # error = sum([(i - 3.5) * line[1][i] for i in range(8)])
    error = pos - 3500

    # Update the error sum and limit it within a reasonable range
    pid_error_sum += error
    pid_error_sum = max(-100, min(100, pid_error_sum))

    # Calculate the change in error for derivative control
    error_diff = error - pid_last_error
    pid_last_error = error

    # Calculate the steering value using PID control
    steering = KP * error + KI * pid_error_sum + KD * error_diff

    # # Limit the steering value within the valid range
    # steering = max(-100, min(100, steering))

    # Provide the calculated steering value to the run_tank function
    speeds = m.run_tank(follower_speed, 100, steering)

    # Print the line color and steering value for debugging
    print(f"CL: {l['hue']} ({int(l['hsv']['hue'])})\tCR: {r['hue']} ({int(r['hsv']['hue'])})\tERR: {int(error)}\tSTEER: {round(steering, 1)}\tSPEEDS: {speeds}\tLINE: {line[1]}")