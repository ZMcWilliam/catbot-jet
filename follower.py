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
import multiprocessing
from PiicoDev_VEML6040 import PiicoDev_VEML6040
from typing import Union, List, Tuple, Dict
from adafruit_ads1x15.analog_in import AnalogIn as ADSAnalogIn

# Calibrated values for the reflectivity array
las_min = [1312, 1248, 1280, 1280, 1248, 1280, 1264, 1312, 1296, 1248, 1264, 1344] 
las_max = [10352, 9600, 10736, 10944, 8544, 11264, 10480, 11952, 10848, 8080, 9584, 12800]

# Ports for IO
PORT_DEBUG_SWITCH = 19
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
KP = 0.06 # Proportional gain
KI = 0  # Integral gain
KD = 0.0  # Derivative gain
follower_speed = 30

# Variables for PID control
pid_error_sum = 0
pid_last_error = 0

obstacle_threshold = 5  # Distance threshold to detect an obstacle

col_thresholds = {
    "green": 0.55,
    "red": 0.65,
}

start_time = time.monotonic()

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
    "line_state": 0, # 0 = Lost line, 1 = Found line, 2 = Full line
    "is_inverted": False,
    "line_contour_full": "",
    "line_contour_switches": "",
    "pos": ((len(PORT_ADS_LINE) - 1) * 1000) / 2,
    "speeds": [0, 0],
}

debug_switch = gpiozero.DigitalInputDevice(PORT_DEBUG_SWITCH, pull_up=True)

USS = {
    key: gpiozero.DistanceSensor(echo=USS_ECHO, trigger=USS_TRIG)
    for key, USS_ECHO, USS_TRIG in zip(PORT_USS_ECHO.keys(), PORT_USS_ECHO.values(), PORT_USS_TRIG.values())
}

itr_stats = {
    name: {"count": 0, "time": 0, "paused": False}
    for name in ["master", "line", "cols"]
}

def update_itr_stat(stat: str, auto_reset: int = False) -> None:
    """
    Updates the specified iteration stat.

    Args:
        stat (str): The stat to update.
        auto_reset (int, optional): The number of iterations before the stat is reset. Defaults to False.
    """
    if itr_stats[stat]["time"] == 0:
        itr_stats[stat]["time"] = time.monotonic()
    
    if auto_reset != False and itr_stats[stat]["count"] % auto_reset == 0:
        itr_stats[stat]["time"] = time.monotonic()
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
    val = itr_stats[stat]["count"] / (time.monotonic() - itr_stats[stat]["time"])
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
        data_scaled[i] = int(max(0, min(100, data_scaled[i] * 100)) * 10)

    return [data, data_scaled]

def calculate_position(values: List[float], last_pos: int, invert: bool = False, prevent_invert: bool = False) -> float:
    """
    Calculates the position on a line based on given reflectivity sensor values.

    Args:
        values (List[float]): List of reflectivity sensor values.
        last_pos (int): The last calculated position on the line.
        invert (bool, optional): Flag indicating whether to invert the reflectivity values. Defaults to False.
        prevent_invert (bool, optional): Flag indicating whether to prevent triggering an invert. Defaults to False.

    Returns:
        float: The calculated position on the line

    Note:
        The position is calculated by taking the weighted average of the reflectivity values.
        
        last_pos is used to check the last calculated position in order to provide
        a more accurate position when the robot is not on the line.

        A global variable debug_info["line_state"] is set based on the state of the line.
    """
    conf_off_line_gap = 0.4 # Percent
    conf_on_line_threshold = 100 # Val - A value above this indicates that the robot is on the line
    conf_threshold = 20 # Val - Values above this will be included in the weighted average
    conf_black_threshold = 300 # Val - Values above this will be considered "black"

    max_val = (len(values) - 1) * 1000
    on_line = False
    avg = 0
    sum_values = 0

    if invert:
        values = [1000 - value for value in values]

    contour_full = ""
    contour_switches = ""
    for i in range(len(values)):
        contour_full += "1" if values[i] > conf_black_threshold else "0"

        if contour_switches == "":
            contour_switches += contour_full[i]
        elif contour_full[i] != contour_full[i - 1]:
            contour_switches += contour_full[i]
    
    debug_info["line_contour_full"] = contour_full
    debug_info["line_contour_switches"] = contour_switches

    if not debug_switch.value:
        print(f"{itr_stats['master']['count']:5d}", contour_full, invert, "  " + contour_switches + " "*(10 - len(contour_switches)), "".join([f"{x:5d}" for x in values]), "----", "".join([f"{x:5d}" for x in latest_data['line']['scaled']]))

    if contour_switches == "1": # Line is fully black
        debug_info["line_state"] = 2
        return max_val / 2

    if (not prevent_invert
            and contour_switches == "101" 
            and contour_full.startswith("111") 
            and contour_full.endswith("111")
            and values[0] > 800
            and values[-1] > 800
        ): # Line is black-white-black, we need to invert the values
        print("DETECTED INVERTED LINE, NOW: " + str(not invert))
        debug_info["is_inverted"] = not invert

        return calculate_position(values, last_pos, invert = True)
    
    for i, value in enumerate(values):
        if value > conf_on_line_threshold:
            on_line = True

        if value > conf_threshold:
            avg += value * (i * 1000)
            sum_values += value

    if not on_line or sum_values == 0:
        debug_info["line_state"] = 0

        if last_pos <= conf_off_line_gap * (len(values) - 1) * 1000:
            return 0
        elif last_pos >= max_val - conf_off_line_gap * (len(values) - 1) * 1000:
            return max_val
        else:
            return max_val / 2

    debug_info["line_state"] = 1
    return avg / sum_values

def follow_line() -> None:
    """
    Follows a line using PID control.
    """
    global pid_error_sum, pid_last_error

    pos = calculate_position(latest_data["line"]["scaled"], debug_info["pos"], invert=debug_info["is_inverted"])
    debug_info["pos"] = pos
    error = pos - (len(latest_data["line"]["scaled"]) - 1) * 1000 / 2

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
    side_distance_threshold = 30 # Distance threshold for the side ultrasonic sensor to detect an obstacle

    m.run_tank_for_time(-40, -40, 500)
    m.run_tank_for_time(-100, 100, 1100) # 90 degree left turn
    
    def forward_until(distance: int, less_than: bool, check_for_line: bool = False, timeout: int = 10, debug_prefix: str = "") -> bool:
        """
        Obstacle avoidance helper:
        Goes forward until the side ultrasonic sensor sees something less than the given distance.
        Allows for checking for the existence of a line while going forward.

        Args:
            distance (int): The distance threshold for the side ultrasonic sensor to detect an obstacle.
            less_than (bool): Flag indicating whether to check for a distance less than or greater than the given distance.
            check_for_line (bool, optional): Flag indicating whether to check for a line while going forward. Defaults to False.
            timeout (int, optional): The timeout in seconds. Defaults to 10.
            debug_prefix (str, optional): The debug prefix to use. Defaults to "".
        """
        m.run_tank(30, 30)
        start_time = time.time()
        while time.time() - start_time < timeout:
            print(debug_prefix + "Side Distance: " + str(latest_data["distance_side"]))
            if check_for_line:
                line = latest_data["line"]["scaled"]
                debug_info["pos"] = calculate_position(line, debug_info["pos"], invert=False, prevent_invert=True)

                if debug_info["line_contour_switches"] in ["010"]:
                    print(debug_prefix + "Found line")
                    return True
                
                return True
            
            if latest_data["distance_side"] < distance and less_than:
                return False
            elif latest_data["distance_side"] >= distance and not less_than:
                return False
        
        return False

    turn_count = 0
    while turn_count < 4:
        m.run_tank(30, 30)
        # Go forward until side ultrasonic sees something less than 30cm 
        # After the second turn, check for a line while going forward
        if forward_until(side_distance_threshold, less_than=True, check_for_line=turn_count >= 2, debug_prefix="STEP A, TURN " + str(turn_count) + " - "):
            break # Found a line
        print("Found obstacle, side distance: " + str(latest_data["distance_side"]))
        
        # Keep going forward until side ultrasonic no longer sees object
        if forward_until(side_distance_threshold, less_than=False, check_for_line=turn_count >= 2, debug_prefix="STEP B, TURN " + str(turn_count) + " - "):
            break # Found a line
        print("Lost obstacle, side distance: " + str(latest_data["distance_side"]))

        # Did not find a line, so keep going forward a bit, turn right, and try again
        m.run_tank_for_time(25, 25, 900)
        m.run_tank_for_time(100, -100, 950) # 90 degree right turn
        turn_count += 1
        
        time.sleep(0.5)

    # We found the line, now reconnect and follow it
    m.stop_all()
    time.sleep(3)

class Monitor:
    """
    A class that monitors a function in a separate process.

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
        self.pause_event = multiprocessing.Event()

        self.queue = multiprocessing.Queue()

        self.process = multiprocessing.Process(target=self.run_loop)
        self.process.daemon = True
        self.process.start()

        self.latest_result = None

    def run_loop(self):
        tick = 0
        start = time.monotonic()
        while True:
            tick += 1
            if not self.paused:
                # update_itr_stat(self.itr_stat)

                # Call the function and put the result in the queue
                result = self.loop_function()
                self.queue.put([result, [start, tick]])

                if self.timeout > 0:
                    time.sleep(self.timeout)
            else:
                self.pause_event.wait()

    def get_data(self):
        if not self.queue.empty():
            self.latest_result = self.queue.get()
        if self.latest_result is None:
            return None
        
        itr_stats[self.itr_stat]["time"] = self.latest_result[1][0]
        itr_stats[self.itr_stat]["count"] = self.latest_result[1][1]
        return self.latest_result[0]

    def wait_for_data(self):
        while self.latest_result is None:
            if self.get_data() is not None:
                break
            time.sleep(0.05)
        return self.latest_result
    
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
        self.process.terminate()

monitor_ready = False
async def run_monitors():
    """
    Runs several sensor monitors, and updates the latest data.
    """
    global monitor_ready
    line_monitor = Monitor(read_line, "line")
    line_monitor.wait_for_data()
    print("LINE_MONITOR: First data point received")

    cols_monitor = Monitor(lambda: (read_col(PORT_COL_L), read_col(PORT_COL_R)), "cols", timeout=0.02)
    cols_monitor.wait_for_data()
    print("COLS_MONITOR: First data point received")

    monitor_ready = True
    while True:
        latest_data["line"]["raw"], latest_data["line"]["scaled"] = line_monitor.get_data()
        latest_data["col_l"], latest_data["col_r"] = cols_monitor.get_data()

        latest_data["distance_front"] = round(USS["front"].distance * 100, 2)
        latest_data["distance_side"] = round(USS["side"].distance * 100, 2)
        await asyncio.sleep(0.01)

monitor_thread = threading.Thread(target=lambda: asyncio.run(run_monitors()))
monitor_thread.daemon = True
monitor_thread.start()

print("Waiting for monitors to start")
while not monitor_ready:
    time.sleep(0.05)
print("Monitors started, starting main loop")

while True:
    try: 
        update_itr_stat("master", 10000)

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

        if itr_stats["master"]["count"] % 1 == 0 and debug_switch.value:
            print(f"ITR: {itr_stats['master']['count']:4d} M{get_itr_stat('master')}, L{get_itr_stat('line')}, C{get_itr_stat('cols')}"
                + f"\t Line: {['LOST', ' ON ', 'ALL '][debug_info['line_state']]} "
                    + (" INVERT " if debug_info['is_inverted'] else "        ")
                    + "".join([f"{x:5d}" for x in latest_data['line']['scaled']])
                + f"\t L: {latest_data['col_l']['eval']} ({latest_data['col_l']['hue']}, {round(latest_data['col_l']['hsv']['sat'], 2)})"
                + f"\t R: {latest_data['col_r']['eval']} ({latest_data['col_r']['hue']}, {round(latest_data['col_r']['hsv']['sat'], 2)})"
                + f"\t USS: {latest_data['distance_front'], latest_data['distance_side']}"
                + f"\t Pos: {int(debug_info['pos']):6d},"
                + f"\t Steering: {int(debug_info['steering']):4d},"
                + f"\t Speeds: {debug_info['speeds']},"
            )

        time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping Bot")
        break

m.stop_all()
