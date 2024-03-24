#                                ██████╗ █████╗ ████████╗██████╗  ██████╗ ████████╗         ██╗███████╗████████╗
#   _._     _,-'""`-._          ██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝         ██║██╔════╝╚══██╔══╝
#   (,-.`._,'(       |\`-/|     ██║     ███████║   ██║   ██████╔╝██║   ██║   ██║            ██║█████╗     ██║
#       `-.-' \ )-`( , o o)     ██║     ██╔══██║   ██║   ██╔══██╗██║   ██║   ██║       ██   ██║██╔══╝     ██║
#           `-    \`_`"'-       ╚██████╗██║  ██║   ██║   ██████╔╝╚██████╔╝   ██║       ╚█████╔╝███████╗   ██║
#                                ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝        ╚════╝ ╚══════╝   ╚═╝

# RoboCup Junior Rescue Line 2023 - Asia Pacific (South Korea)
# https://github.com/zmcwilliam/catbot-rcjap

print("Starting CatBot")
import time
import os
import sys
import traceback
import math
import signal
import board
import cv2
import busio
import numpy as np
import tensorflow as tf
from colorama import init, Fore, Style

from helpers import camera as c
from helpers import camerakit as ck
from helpers import motorkit as m
from helpers import intersections
from helpers import config
from helpers.servokit import ServoManager
from helpers.cmps14 import CMPS14
from helpers.tof import RangeSensorMonitor

init()
m.stop_all()
os.system("cat motd.txt")
print(Fore.CYAN + "Loaded Modules" + Style.RESET_ALL)

DEBUGGER = True # Should the debug switch actually work? This should be set to false if using the runner

# -------------
# CONFIGURATION
# -------------
max_error = 145                     # Maximum error value when calculating a percentage
max_angle = 90                      # Maximum angle value when calculating a percentage
error_weight = 0.5                  # Weight of the error value when calculating the PID input
black_contour_threshold = 4000      # Minimum area of a contour to be considered valid
turning_line_iterations = 7         # While doing a green turn, dilate/erode by this much to fill in gaps

KP = 1.2                            # Proportional gain
KI = 0                              # Integral gain
KD = 0.1                            # Derivative gain

follower_speed = 37                 # Base speed of the line follower
obstacle_threshold = 60             # Minimum distance threshold for obstacles (mm)

pitch_flat = 8
pitch_flat_error = 7 # Allow +- this error for flat
min_pitch_ramp_up_slight = 20
min_pitch_ramp_up_full = 30

# ----------------
# SYSTEM VARIABLES
# ----------------
program_active = True
has_attempted_exit = False
has_moved_windows = False
program_sleep_time = 0.001

current_steering = 0
current_time = time.time()
current_follower_bearing = 0
last_line_pos = np.array([100, 100])
last_break_in_line = time.time() - 60

turning = None
last_green_time = 0
initial_green_time = 0
current_linefollowing_state = None
intersection_state_debug = ["", time.time()]
red_stop_check = 0
evac_detect_check = 0

silver_prediction = False
silver_first_detect_time = time.time() - 60

pid_last_error = 0
pid_integral = 0

frames = 0
fpsTime = time.time()
fpsLoop = 0
fpsCamera = 0

time_since_ramp_start = 0
time_ramp_end = 0

no_black_contours_mode = "straight"
no_black_contours = False

# --------------
# LOAD ML MODELS
# --------------
model_silver = tf.saved_model.load("ml/model/silver-trt")
infer_func_silver = model_silver.signatures["serving_default"]

def infer_silver(frame):
    """
    Runs inference using the silver model on a given frame
    Expects a grayscale input image

    Args:
        frame (np.array): The frame to the inference on

    Returns:
        int: 0 for "without", 1 for "with"
    """
    resized_frame = cv2.resize(frame, (72, 66))
    input_data = np.expand_dims(resized_frame, axis=-1) # Add a channel dimension

    # Convert input_data to a TensorFlow Tensor with a batch dimension
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    input_data = tf.expand_dims(input_data, axis=0)

    # Run inference
    labelling = infer_func_silver(tf.constant(input_data, dtype=float))
    label_key = list(labelling.keys())[0] # Currently dense_2
    labels = labelling[label_key]

    # Get the predicted class (0 for "without" and 1 for "with")
    predicted_class = int(np.argmax(labels, axis=1))
    return predicted_class == 1

# ------------------
# INITIALISE DEVICES
# ------------------
i2c = busio.I2C(board.SCL, board.SDA)

cam = c.CameraStream(
    camera_num=0,
    processing_conf=config.processing_conf
)

servo = ServoManager()

# debug_switch = gpiozero.DigitalInputDevice(PORT_DEBUG_SWITCH, pull_up=True) if DEBUGGER else None

cmps = CMPS14(7, 0x61)

tof = RangeSensorMonitor()
tof.start()

# -----------------
# VARIOUS FUNCTIONS
# -----------------
def exit_gracefully(signum=None, frame=None) -> None:
    """
    Handles program exit gracefully. Called by SIGINT signal.

    Args:
        signum (int, optional): Signal number. Defaults to None.
        frame (frame, optional): Current stack frame. Defaults to None.
    """
    global program_active
    global has_attempted_exit
    if has_attempted_exit:
        # We already tried to exit, but may have gotten stuck. Force the exit.
        print("\nForcefully Exiting")
        sys.exit()

    print("\n\nExiting Gracefully...\n")
    has_attempted_exit = True
    program_active = False
    tof.stop()
    cv2.destroyAllWindows()
    cam.stop()
    tof.join()
    m.stop_all()
    print(Fore.GREEN + "SAFE TO EXIT" + Style.RESET_ALL)
    sys.exit()

signal.signal(signal.SIGINT, exit_gracefully)

def debug_state(mode=None) -> bool:
    """
    Returns the current debugger state

    Args:
        mode (str, optional): The type of debugger. Defaults to None.

    Returns:
        bool: False for OFF, True for ON
    """
    if mode == "rescue":
        return True

    return DEBUGGER # and debug_switch.value

def align_to_bearing(target_bearing: int, cutoff_error: int, timeout: int = 10, debug_prefix: str = "") -> bool:
    """
    Aligns to the given bearing.

    Args:
        target_bearing (int): The bearing to align to.
        cutoff_error (int): The error threshold to stop aligning.
        timeout (int, optional): The timeout in seconds. Defaults to 10.
        debug_prefix (str, optional): The debug prefix to use. Defaults to "".
    """
    # Restrict target_bearing to 0-359
    target_bearing = target_bearing % 360

    last_bearings = []
    speed_adj = 0
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_bearing = cmps.read_bearing_16bit()
        error = min(abs(current_bearing - target_bearing), abs(target_bearing - current_bearing + 360))

        if len(last_bearings) >= 20:
            last_bearings.pop(0)
        last_bearings.append(int(current_bearing/2))

        if len(last_bearings) >= 20 and sum(last_bearings) / len(last_bearings) == last_bearings[0]:
            speed_adj += 3
            print(f"{debug_prefix} Bearing not changing, increasing speed by {speed_adj}")
        else:
            speed_adj = 0

        if speed_adj > 60:
            print("Jumping forward")
            m.run_tank_for_time(40, 40, 100)

        if error < cutoff_error:
            print(f"{debug_prefix} FOUND Bearing: {current_bearing}\tTarget: {target_bearing}\tError: {error}")
            m.stop_all()
            return True

        max_speed = 50 # Speed to rotate when error is 180
        min_speed = 25 # Speed to rotate when error is 0

        rotate_speed = ((max_speed - min_speed)/180) + min_speed
        rotate_speed += speed_adj

        # Rotate in the direction closest to the bearing
        if (current_bearing - target_bearing) % 360 < 180:
            m.run_tank(-rotate_speed, rotate_speed)
        else:
            m.run_tank(rotate_speed, -rotate_speed)

        print(f"{debug_prefix}Bearing: {current_bearing}\tTarget: {target_bearing}\tError: {error}\tSpeed: {rotate_speed}")

def run_to_dist(target_dist: int, cutoff_error: int = 5, speed: int = 40, speed_slow: int = 25, timeout: int = 5000, use_min: bool = False) -> bool:
    """
    Drives motors until the given distance is reached.

    Args:
        target_dist (int): The distance to drive to, in mm
        cutoff_error (int, optional): The error threshold to stop driving. Defaults to 10.
        speed (int, optional): The speed to drive at. Defaults to 40.
        speed_slow (int, optional): The speed to drive at when the error is near the cutoff. Defaults to 30.
        timeout (int, optional): The timeout in milliseconds. Defaults to 5000.
        use_min (bool, optional): Whether to accept any distance less than the target. Defaults to False.

    Returns:
        bool: True if the distance was reached, False if the timeout was reached.
    """
    start_time = time.time()
    while time.time() - start_time < (timeout / 1000):
        front_dist = tof.range_mm

        if use_min and 0 < front_dist <= target_dist:
            m.stop_all()
            return True
        
        error = front_dist - target_dist

        if abs(error) < cutoff_error:
            print(f"FOUND Dist: {front_dist}\tTarget: {target_dist}\tError: {error}")
            m.stop_all()
            return True
        
        selected_speed = speed if abs(error) > 100 else speed_slow
        if error < 0:
            selected_speed = -selected_speed
        
        m.run_tank(selected_speed, selected_speed)
        print(f"Dist: {front_dist}\tTarget: {target_dist}\tError: {error}\tSpeed: {selected_speed}")
    
    print("Timeout")
    m.stop_all()
    return False

# ------------------
# OBSTACLE AVOIDANCE
# ------------------
def avoid_obstacle() -> None:
    """
    Performs obstacle avoidance when an obstacle is detected.
    """
    global program_active

    a_dist_back = 500
    a_ang_turn = 85
    a_second_steer = 28
    a_timeout = 2.4

    b_dist_back = 600
    b_ang_turn = 50
    b_timeout = 1

    print("START OF OBSTACLE")
    # run_to_dist(30, 2, 30, 23, 1200, False)
    servo.cam.to(45)
    m.run_tank_for_time(-40, -40, 200)
    time.sleep(0.8)

    initial_obs_bearing = cmps.read_bearing_16bit()

    # Figure out which side the line is on
    frame_processed = cam.read_stream_processed()
    img0_raw = frame_processed["raw"].copy()
    
    img0_resized_obst = cam.resize_image_obstacle(img0_raw)
    img0_gray_obst = cv2.cvtColor(img0_resized_obst, cv2.COLOR_BGR2GRAY)
    img0_gray_obst = cv2.GaussianBlur(img0_gray_obst, (5, 5), 0)
    img0_gray_obst_scaled = img0_gray_obst * config.get("calibration_map_obst")

    img0_binary_obstacle = ((img0_gray_obst_scaled > config.get("obstacle_line_threshold")) * 255).astype(np.uint8)
    img0_binary_obstacle = cv2.morphologyEx(img0_binary_obstacle, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    img0_gray_obst_not = cv2.bitwise_not(img0_binary_obstacle)
    black_contours, _ = cv2.findContours(img0_gray_obst_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_contours_filtered = [[c, cv2.contourArea(c)] for c in black_contours]
    black_contours_filtered = sorted([[c, a] for c, a in black_contours_filtered if a > 1500], key=lambda x: x[1], reverse=True)

    line_side = None
    for c, a in black_contours_filtered:
        print(a)
        x, y, w, h = cv2.boundingRect(c)

        cond_r = x + w > img0_resized_obst.shape[1] - 15
        cond_l = x < 15

        if cond_r and cond_l:
            cv2.drawContours(img0_resized_obst, [c], -1, (0, 255, 255), 3)
        elif cond_r:
            line_side = "R"
            cv2.drawContours(img0_resized_obst, [c], -1, (0, 255, 0), 3)
            break
        elif cond_l:
            line_side = "L"
            cv2.drawContours(img0_resized_obst, [c], -1, (255, 0, 0), 3)
            break
        else:
            line_side = "S"
            cv2.drawContours(img0_resized_obst, [c], -1, (0, 0, 255), 3)

    print(f"Line Side: {line_side}")

    time.sleep(0.3)

    servo.cam.toMin()

    img0 = frame_processed["resized"]
    cv2.imshow("img0", img0)
    cv2.imshow("img0_resized_obst", img0_resized_obst)
    cv2.imshow("img0_binary_obstacle", img0_binary_obstacle)

    cv2.moveWindow("img0_resized_obst", 950, 400)
    cv2.moveWindow("img0_binary_obstacle", 950, 750)

    k = cv2.waitKey(5)
    if k & 0xFF == ord('q'):
        program_active = False
        return
    
    if line_side is None:
        print("WARNING: Failed to set line side, assuming straight")
        line_side = "S"
        time.sleep(1)

    if line_side == "S":
        m.stop_all()
        m.run_tank_for_time(-40, -40, a_dist_back)
        align_to_bearing(initial_obs_bearing - a_ang_turn, 1, debug_prefix="OBSTACLE ALIGN: ")
        time.sleep(0.2)
        
        obstacle_dir = -1
        if tof.range_mm < 220:
            # Likely a wall, let's go the other way
            obstacle_dir = 1
            align_to_bearing(initial_obs_bearing + a_ang_turn, 1, debug_prefix="OBSTACLE AVOID WALL: ")

        if obstacle_dir == -1: m.run_tank(100, a_second_steer)
        else: m.run_tank(a_second_steer, 100)
    else:
        m.stop_all()
        obstacle_dir = -1 if line_side == "L" else 1
        m.run_tank_for_time(-40, -40, b_dist_back)
        align_to_bearing(initial_obs_bearing + (b_ang_turn * obstacle_dir), 1, debug_prefix=f"OBSTACLE {line_side} ALIGN: ")

        m.run_tank(40, 40)

    time_avoid_start = time.time()

    # Start checking for a line while continuing to rotate around the obstacle
    while True:
        frame_processed = cam.read_stream_processed()
        
        if debug_state():
            img0 = frame_processed["resized"]

            if time.time() - time_avoid_start <= (a_timeout if line_side == "S" else b_timeout):
                cv2.putText(img0, "TIMEOUT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("img0", img0)

            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                program_active = False
                break

        img0_line_not = cv2.bitwise_not(frame_processed["line"])
        black_contours, _ = cv2.findContours(img0_line_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_contours_filtered = [c for c in black_contours if cv2.contourArea(c) > 5000]

        if len(black_contours_filtered) >= 1 and time.time() - time_avoid_start > (a_timeout if line_side == "S" else b_timeout):
            print("[OBSTACLE] Found Line")
            print(f"Init: {initial_obs_bearing}\tCurrent: {cmps.read_bearing_16bit()}")
            break

    m.stop_all()
    time.sleep(0.6)

    if line_side == "S":
        m.run_tank_for_time(40, 40, 700)
        align_to_bearing(initial_obs_bearing, 1, debug_prefix="OBSTACLE FINAL: ")
    else:
        m.run_tank_for_time(40, 40, 700)
        align_to_bearing(initial_obs_bearing + (90 * obstacle_dir), 1, debug_prefix=f"OBSTACLE {line_side} FINAL: ")
        m.run_tank_for_time(-30, -30, 400)

    # Back up while image is all white
    while True:
        frame_processed = cam.read_stream_processed()
        img0_line_not = cv2.bitwise_not(frame_processed["line"])
        black_contours, _ = cv2.findContours(img0_line_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_contours_filtered = [c for c in black_contours if cv2.contourArea(c) > 4000]

        if len(black_contours_filtered) == 0:
            m.run_tank(-30, -30)
        else:
            break

    m.stop_all()

    cv2.destroyWindow("img0_resized_obst")
    cv2.destroyWindow("img0_binary_obstacle")
    print("END OF OBSTACLE")

# ---------------
# EVACUATION ZONE
# ---------------
def run_evac():
    global fpsCamera
    global program_active
    
    framesEvac = 0
    fpsTimeEvac = time.time()

    rescue_mode = "init"
    victim_capture_qty = 0
    victim_check_counter = 0
    corner_check_counter = 0
    current_victim_start = time.time()
    last_victim_time = time.time()

    m.stop_all()
    
    print("EVAC: Loading Models...", end="")
    from ultralytics import YOLO # Yolo takes a few seconds to load, so do it here to avoid every program load being slow
    model_victims = YOLO("ml/model/victims.pt")
    model_corners = YOLO("ml/model/corners.pt")
    print(f"Done. Took {int((time.time() - fpsTimeEvac) * 1000)}ms")

    print("EVAC: Waiting for first inference of models...")
    frame_processed = cam.read_stream_processed()
    img0_raw = frame_processed["raw"].copy()
    img0_resized_evac = cv2.resize(img0_raw, (160, 120))

    evac_start = time.time()
    model_victims(img0_resized_evac, task="detect")
    print("Loaded Victims Model. Took " + str(int((time.time() - evac_start) * 1000)) + "ms")

    evac_start = time.time()
    model_corners(img0_resized_evac, task="detect")
    print("Loaded Corners Model. Took " + str(int((time.time() - evac_start) * 1000)) + "ms")

    evac_start = time.time()
    start_of_evac_bearing = cmps.read_bearing_16bit()

    while True:
        if int(time.time() - evac_start) % 10 == 0:
            print(f"EVAC MODE: {rescue_mode} EVAC TIME: {int(time.time() - evac_start)}")

        framesEvac += 1

        fpsEvac = int(framesEvac/(time.time()-fpsTimeEvac))
        fpsCamera = cam.get_fps()
        if framesEvac % 20 == 0 and framesEvac != 0:
            print(f"Evac Mode: {rescue_mode} | Evac FPS: {fpsEvac} | Camera FPS: {cam.get_fps()} | Sleep time: {int(program_sleep_time*1000)}")

        frame_processed = cam.read_stream_processed()
        img0 = frame_processed["resized"]
        img0_raw = frame_processed["raw"]
        img0_resized_evac = cv2.resize(img0_raw, (160, 120))

        # -------------
        # INITIAL ENTER
        # -------------
        if rescue_mode == "init":
            print("Entering Evacuation Zone")
            # Ensure robot won't drive out of the arena if exit is opposite to entry
            if tof.range_mm > 1300:
                m.run_tank_for_time(40, 40, 2000)
                align_to_bearing(cmps.read_bearing_16bit() - 90, 10, debug_prefix="EVAC ENTRY: ")
                run_to_dist(130, 5, 40, 25, 8000, False)
                m.run_tank_for_time(-40, -40, 2000)
            else:
                initial_time = time.time()
                run_to_dist(130, 5, 40, 25, 8000, False)
                if time.time() - initial_time > 4:
                    m.run_tank_for_time(-40, -40, 2000)
                else: # Probably ran into a wall, let's try force align to it
                    for i in range(6): # avoid climbing a wall lol
                        m.run_tank_for_time(30, 30, 500)
                        time.sleep(0.1)
                    m.run_tank_for_time(-40, -40, 1000)

            # Rotate and find the bearing matching the smallest distance
            target_bearing = cmps.read_bearing_16bit()

            lowest_dist = [tof.range_mm, target_bearing]

            initial_time = time.time()
            last_pass_time = time.time()
            passes = 0

            m.run_tank(30, -30)
            while time.time() - initial_time < 10:
                current_dist = tof.range_mm
                if current_dist < lowest_dist[0]:
                    lowest_dist = [current_dist, cmps.read_bearing_16bit()]

                current_bearing = cmps.read_bearing_16bit()
                error = min(abs(current_bearing - target_bearing), abs(target_bearing - current_bearing + 360))

                if (time.time() - last_pass_time > 3 and error < 5) or time.time() - last_pass_time > 8:
                    if passes < 1:
                        last_pass_time = time.time()
                        passes += 1
                    else:
                        m.stop_all()
                        break
                print(f"Dist: {current_dist:.2f}\tBearing: {current_bearing:.2f}\tError: {error:.2f}")

            print(f"Lowest: {lowest_dist[0]} at {lowest_dist[1]}")

            time.sleep(1)

            # Rotate to the bearing with the lowest distance
            target_bearing = lowest_dist[1]
            align_to_bearing(target_bearing, 2, 10, "ROTATE: ")

            if tof.range_mm > 1000:
                align_to_bearing(target_bearing + 90, 2, 10, "ROTATE: ")

            if 0 < tof.range_mm < 1000:
                run_to_dist(400, 5, 40, 25, 5000, False)

            print("Done. Finding Victims...")

            servo.cam.toMin()
            servo.lift.toMax()
            servo.claw.toMax()
            servo.cam.to(80)
            time.sleep(0.5)

            rescue_mode = "victim"
            fpsTimeEvac = time.time()
            current_victim_start = time.time()
            last_victim_time = time.time()
            framesEvac = 0
            continue

        # --------------
        # LOCATE VICTIMS
        # --------------
        elif rescue_mode == "victim":
            if victim_capture_qty >= 3:
                print("Found all victims")
                rescue_mode = "corner_green"

            if time.time() - fpsTimeEvac > 30 + (victim_capture_qty * 40):
                print("VICTIMS TOOK TOO LONG - SKIPPING TO GREEN CORNER")
                rescue_mode = "corner_green"

            if time.time() - last_victim_time > 5:
                print("Could not find a victim, moving forward")
                if tof.range_mm > 1200:
                    m.run_tank_for_time(40, -40, 900)

                # TODO: Instead of this, find the largest (non-exit) distance, and drive towards that
                m.run_tank_for_time(40, 40, 1000)
                last_victim_time = time.time()

            servo.gate.toMin()
            servo.lift.toMax()
            servo.claw.toMax()
            servo.cam.to(80)

            start_inf = time.time()
            victims_results = model_victims(img0_resized_evac)
            inference_time_ms = (time.time() - start_inf) * 1000

            found_victims = []
            for r in victims_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1 * img0_raw.shape[1] / img0_resized_evac.shape[1]),
                        int(y1 * img0_raw.shape[0] / img0_resized_evac.shape[0]),
                        int(x2 * img0_raw.shape[1] / img0_resized_evac.shape[1]),
                        int(y2 * img0_raw.shape[0] / img0_resized_evac.shape[0]),
                    )

                    conf = int(box.conf * 100)

                    obj_type = int(box.cls[0])

                    vert_weight = 2
                    midpoint_x = (x1 + x2) / 2
                    midpoint_y = (y1 + y2) / 2
                    horizontal_distance = midpoint_x - (img0_raw.shape[1] / 2)
                    vertical_distance = midpoint_y - img0_raw.shape[0]
                    dist_amt = int(math.sqrt(horizontal_distance ** 2 + (vertical_distance * vert_weight) ** 2))
                        
                    found_victims.append([obj_type, conf, dist_amt, [midpoint_x, midpoint_y], [horizontal_distance, vertical_distance], [x1, y1, x2, y2]])

                    cv2.rectangle(img0_raw, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img0_raw, f"{obj_type} ({conf}) {dist_amt}", [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            v_can_approach = False
            v_target = None

            # Only approach silver victims until both are found or 80 seconds have passed
            if victim_capture_qty < 2 and time.time() - fpsTimeEvac < (60 if victim_capture_qty > 0 else 20):
                found_victims = [v for v in found_victims if v[0] == 1]

            # Regardless of count (perhaps we incorrectly counted...), always ignore a black victim if a silver exists too
            if len(found_victims) >= 2:
                filtered_victims = [v for v in found_victims if v[0] == 1]
                if len(filtered_victims) > 0: # This would only be False if we had 2 black victims... somehow
                    found_victims = filtered_victims

            if len(found_victims) >= 1:
                last_victim_time = time.time()
                found_victims = sorted(found_victims, key=lambda x: x[2])
                v_target = found_victims[0]
                
                # If a victim is within these limits (horz/vert distance offsets), we can approach it
                approach_range = [[-85, 85], [-100, 0]]
                if time.time() - current_victim_start > 8:
                    print("EXPANDING APPROACH RANGE")
                    approach_range = [[-85, 85], [-200, 0]]

                v_can_approach = approach_range[0][0] < v_target[4][0] < approach_range[0][1] and approach_range[1][0] < v_target[4][1] < approach_range[1][1]
                    
                if v_can_approach:
                    victim_check_counter += 1
                    cv2.putText(img0_raw, f"APPROACH {victim_check_counter}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    victim_check_counter = 0

                # Highlight the target
                cv2.rectangle(img0_raw, (v_target[5][0], v_target[5][1]), (v_target[5][2], v_target[5][3]), (0, 255, 0), 3)

                # Draw a box around the appraoch range
                cv2.rectangle(
                    img0_raw, 
                    (int(img0_raw.shape[1] / 2) + approach_range[0][0], int(img0_raw.shape[0]) + approach_range[1][0]), 
                    (int(img0_raw.shape[1] / 2) + approach_range[0][1], int(img0_raw.shape[0]) + approach_range[1][1]), 
                    (0, 255, 255), 
                    2
                )

                # Draw midpoint with coord
                cv2.circle(img0_raw, (int(v_target[3][0]), int(v_target[3][1])), 5, (0, 0, 255), -1)
                cv2.putText(img0_raw, f"{v_target[3][0]}, {v_target[3][1]}", (int(v_target[3][0]) + 10, int(v_target[3][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw the horizontal and vertical distance just below
                cv2.putText(img0_raw, f"X: {v_target[4][0]}", (int(v_target[3][0]) + 10, int(v_target[3][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 165, 255), 2)
                cv2.putText(img0_raw, f"Y: {v_target[4][1]}", (int(v_target[3][0]) + 10, int(v_target[3][1]) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 165, 255), 2)

                cv2.putText(img0_raw, f"{len(found_victims)} FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)
            else:
                current_victim_start = time.time()
                victim_check_counter = 0
                cv2.putText(img0_raw, "NONE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

            cv2.putText(img0_raw, f"{inference_time_ms:.2f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0_raw, f"{fpsEvac:.1f} fps", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)

            # Define the motor speeds
            left_speed = None
            right_speed = None

            if v_target is not None:
                if victim_check_counter >= 3:
                    cv2.putText(img0_raw, "COLLECTING", (10, img0_raw.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    img0_raw_scaled = cv2.resize(img0_raw, (img0.shape[1], int(img0.shape[1] * (img0_raw.shape[0] / img0_raw.shape[1]))))
                    cv2.imshow("img0_raw", img0_raw_scaled)

                    k = cv2.waitKey(1)
                    if k & 0xFF == ord('q'):
                        program_active = False
                        break
                    
                    # 3 valid approach signals in a row, stop and grab
                    victim_capture_qty += 1

                    if time.time() - current_victim_start > 8:
                        m.run_tank_for_time(-30, 30, 200)
                        servo.claw.to(50)
                        m.run_tank(40, 40)
                        for i in range(30):
                            servo.claw.to(50 - i)
                            time.sleep(0.05)
                    else:
                        m.run_tank(30, 30)
                    time.sleep(0.2)
                    servo.claw.toMin()
                    time.sleep(0.6)
                    m.run_tank_for_time(-30, -30, 900)
                    servo.cam.toMin() # Get the camera out of the way
                    servo.lift.toMin()
                    time.sleep(0.8)

                    if v_target[0] == 1: # Let go of silver balls immediately
                        servo.claw.toMax()
                        time.sleep(1)

                        # Return to search
                        servo.lift.toMax()
                        servo.cam.to(80)
                        time.sleep(1)
                        current_victim_start = time.time()
                        last_victim_time = time.time()
                    
                    if v_target[0] == 1 and victim_capture_qty > 2:
                        # There are only 2 silver balls, so we must have missed one earlier if we now have >2
                        victim_capture_qty = 2

                    if v_target[0] == 0 or victim_capture_qty == 3:
                        # If we've grabbed a black ball, hold onto it and end victim search
                        victim_check_counter = 0
                        victim_capture_qty = max(3, victim_capture_qty)
                        rescue_mode = "corner_green"
                    continue
                elif victim_check_counter > 0:
                    # Stop, and make sure we see the circle a bit more
                    left_speed = 0
                    right_speed = 0

                # Calculate the horizontal and vertical distances
                horizontal_distance = v_target[4][0]
                vertical_distance = v_target[4][1]

                if vertical_distance >= -200:
                    # Target is closer in height, steer on the spot
                    if -60 <= horizontal_distance <= 60:
                        # Within acceptable horizontal range, slowly go forward
                        left_speed = 30
                        right_speed = 30
                    elif horizontal_distance < -60:
                        # Target is to the left, turn left
                        left_speed = -30
                        right_speed = 30
                    elif horizontal_distance > 60:
                        # Target is to the right, turn right
                        left_speed = 30
                        right_speed = -30
                else:
                    # Target is further away
                    if -80 <= horizontal_distance <= 80:
                        # Within acceptable horizontal range, move forward
                        left_speed = 40
                        right_speed = 40
                    else:
                        steering_factor = horizontal_distance / (img0_raw.shape[1] / 2)
                        left_speed = 20 + (40 * steering_factor)
                        right_speed = 20 - (40 * steering_factor)

            if left_speed is None or right_speed is None:
                left_speed = -30
                right_speed = 30

            m.run_tank(left_speed, right_speed)

            # Draw speeds
            cv2.putText(img0_raw, f"L: {left_speed}", (img0_raw.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)
            cv2.putText(img0_raw, f"R: {right_speed}", (img0_raw.shape[1] - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)

            cv2.putText(img0_raw, f"{victim_capture_qty}/3", (img0_raw.shape[1] - 200, img0_raw.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if debug_state("rescue"):
                img0_raw_scaled = cv2.resize(img0_raw, (img0.shape[1], int(img0.shape[1] * (img0_raw.shape[0] / img0_raw.shape[1]))))
                cv2.imshow("img0_raw", img0_raw_scaled)

                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break

        # -------------------------
        # LOCATE CORNERS AND RESCUE
        # -------------------------
        elif rescue_mode == "corner_green" or rescue_mode == "corner_red":
            servo.gate.toMin()
            servo.lift.toMin()
            servo.claw.toMin()
            servo.cam.toMax()

            start_inf = time.time()
            corners_results = model_corners(img0_resized_evac)
            inference_time_ms = (time.time() - start_inf) * 1000

            found_corners = []
            for r in corners_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1 * img0_raw.shape[1] / img0_resized_evac.shape[1]),
                        int(y1 * img0_raw.shape[0] / img0_resized_evac.shape[0]),
                        int(x2 * img0_raw.shape[1] / img0_resized_evac.shape[1]),
                        int(y2 * img0_raw.shape[0] / img0_resized_evac.shape[0]),
                    )

                    conf = int(box.conf * 100)

                    if conf < 70: continue

                    obj_type = "red" if int(box.cls[0]) else "green"
                    obj_col = (0, 0, 255) if int(box.cls[0]) else (0, 255, 0)

                    midpoint_x = (x1 + x2) / 2
                    midpoint_y = (y1 + y2) / 2
                    horizontal_distance = midpoint_x - (img0_raw.shape[1] / 2)
                    vertical_distance = midpoint_y - img0_raw.shape[0]
                    box_area = (x2 - x1) * (y2 - y1)

                    found_corners.append([obj_type, conf, box_area, [midpoint_x, midpoint_y], [horizontal_distance, vertical_distance], [x1, y1, x2, y2]])

                    cv2.rectangle(img0_raw, (x1, y1), (x2, y2), obj_col, 3)
                    cv2.putText(img0_raw, f"{obj_type} ({conf})", [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, obj_col, 2)

            c_can_approach = False
            c_target = None

            # Filter out the corner type we're not looking for
            found_corners = [c for c in found_corners if c[0] == ("green" if rescue_mode == "corner_green" else "red")]

            if len(found_corners) >= 1:
                # Sort by area, and get the largest
                found_corners = sorted(found_corners, key=lambda x: x[2])
                c_target = found_corners[0]

                # If a victim is within these limits (horz/vert distance offsets), we can approach it
                approach_range = [[-85, 85], [-100, 0]]

                xL, xR = sorted([c_target[5][0], c_target[5][2]])
                yT, yB = sorted([c_target[5][1], c_target[5][3]])
                iW = img0_raw.shape[1]
                iH = img0_raw.shape[0]
                
                # Two conditions for approach:
                # - If both x coords are within side_thresh_a of each edge of the image 
                # - If the box touches the bottom of the image 
                #   whilst one side is within side_thresh_a of the edge and the other is within side_thresh_b# of the other edge
                side_thresh_a = 20
                side_thresh_b = 25 

                c_can_approach = (
                    (xL < side_thresh_a and xR > iW - side_thresh_a)
                    or 
                    (
                        yB > iH - side_thresh_a and
                        (
                            (xL < iW * (side_thresh_b / 100) and xR > iW - side_thresh_a)
                            or
                            (xL < side_thresh_a and xR > iW * ((100 - side_thresh_b) / 100))
                        )
                    )
                )

                if c_can_approach:
                    corner_check_counter += 1
                    cv2.putText(img0_raw, f"APPROACH {corner_check_counter}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    corner_check_counter = 0

                # Draw a box around the appraoch range
                cv2.rectangle(
                    img0_raw, 
                    (int(img0_raw.shape[1] / 2) + approach_range[0][0], int(img0_raw.shape[0]) + approach_range[1][0]), 
                    (int(img0_raw.shape[1] / 2) + approach_range[0][1], int(img0_raw.shape[0]) + approach_range[1][1]), 
                    (0, 255, 255), 
                    2
                )

                # Draw midpoint with coord
                cv2.circle(img0_raw, (int(c_target[3][0]), int(c_target[3][1])), 5, (0, 0, 255), -1)
                cv2.putText(img0_raw, f"{c_target[3][0]}, {c_target[3][1]}", (int(c_target[3][0]) + 10, int(c_target[3][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw the horizontal and vertical distance just below
                cv2.putText(img0_raw, f"X: {c_target[4][0]}", (int(c_target[3][0]) + 10, int(c_target[3][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 165, 255), 2)
                cv2.putText(img0_raw, f"Y: {c_target[4][1]}", (int(c_target[3][0]) + 10, int(c_target[3][1]) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 165, 255), 2)
            else:
                corner_check_counter = 0
                cv2.putText(img0_raw, "NONE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

            cv2.putText(img0_raw, f"{inference_time_ms:.2f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0_raw, f"{fpsEvac:.1f} fps", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)

            # Define the motor speeds
            left_speed = None
            right_speed = None

            if c_target is not None:
                if corner_check_counter >= 3:
                    cv2.putText(img0_raw, "RESCUING", (10, img0_raw.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    img0_raw_scaled = cv2.resize(img0_raw, (img0.shape[1], int(img0.shape[1] * (img0_raw.shape[0] / img0_raw.shape[1]))))
                    cv2.imshow("img0_raw", img0_raw_scaled)
    
                    k = cv2.waitKey(1)
                    if k & 0xFF == ord('q'):
                        program_active = False
                        break

                    m.run_tank_front(15, 15)
                    m.run_tank_back(35, 35)
                    time.sleep(2)

                    m.run_tank_for_time(-40, -40, 1400)
                    time.sleep(0.1)
                    start_bearing = cmps.read_bearing_16bit()
                    align_to_bearing(start_bearing - 180, 10, debug_prefix="EVAC Align - ")
                    time.sleep(0.1)
                    m.run_tank_for_time(-35, -35, 1000)
                    if rescue_mode == "corner_red":
                        servo.claw.toMax() # Release the held black ball
                    servo.gate.toMax()
                    time.sleep(0.5)
                    for i in range(6):
                        m.run_tank_for_time(100, 100, 150)
                        m.run_tank_for_time(-100, -100, 250)
                    servo.gate.toMin()
                    servo.claw.toMin()
                    m.run_tank_for_time(35, 35, 1000)
                    
                    corner_check_counter = 0
                    if rescue_mode == "corner_green" and victim_capture_qty >= 3:
                        rescue_mode = "corner_red"
                    else:
                        rescue_mode = "exit"
                    continue
                elif corner_check_counter > 0:
                    # Stop, and make sure we see the circle a bit more
                    left_speed = 0
                    right_speed = 0

                # Calculate the horizontal and vertical distances
                horizontal_distance = c_target[4][0]
                vertical_distance = c_target[4][1]

                if vertical_distance >= -200:
                    # Target is closer in height, steer on the spot
                    if -60 <= horizontal_distance <= 60:
                        # Within acceptable horizontal range, slowly go forward
                        left_speed = 30
                        right_speed = 30
                    elif horizontal_distance < -60:
                        # Target is to the left, turn left
                        left_speed = -30
                        right_speed = 30
                    elif horizontal_distance > 60:
                        # Target is to the right, turn right
                        left_speed = 30
                        right_speed = -30
                else:
                    # Target is further away
                    if -80 <= horizontal_distance <= 80:
                        # Within acceptable horizontal range, move forward
                        left_speed = 40
                        right_speed = 40
                    else:
                        steering_factor = horizontal_distance / (img0_raw.shape[1] / 2)
                        left_speed = 20 + (40 * steering_factor)
                        right_speed = 20 - (40 * steering_factor)

            if left_speed is None or right_speed is None:
                left_speed = -30
                right_speed = 30

            m.run_tank(left_speed, right_speed)

            # Draw speeds
            cv2.putText(img0_raw, f"L: {left_speed}", (img0_raw.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)
            cv2.putText(img0_raw, f"R: {right_speed}", (img0_raw.shape[1] - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)

            cv2.putText(img0_raw, f"{victim_capture_qty}/3", (img0_raw.shape[1] - 200, img0_raw.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if debug_state("rescue"):
                img0_raw_scaled = cv2.resize(img0_raw, (img0.shape[1], int(img0.shape[1] * (img0_raw.shape[0] / img0_raw.shape[1]))))
                cv2.imshow("img0_raw", img0_raw_scaled)

                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break

        # ----
        # EXIT
        # ----
        elif rescue_mode == "exit":
            print("Exiting Evacuation Zone")      
            servo.gate.toMin()
            servo.lift.toMin()
            servo.claw.toMin()
            servo.cam.toMin()
      
            m.run_tank_for_time(40, 40, 1200, True)
            align_to_bearing(cmps.read_bearing_16bit() - 180, 10, debug_prefix="ROTATE: ")

            dists = []
            dists_sec_deriv = []

            target_bearing = cmps.read_bearing_16bit()
            initial_time = time.time()

            m.run_tank(25, -25)
            while time.time() - initial_time < 10:
                current_dist = tof.range_mm
                current_bearing = cmps.read_bearing_16bit()

                dists.append([current_dist, current_bearing])

                i = len(dists) - 1
                if i > 2:
                    dists_sec_deriv.append([dists[i - 2][0] + dists[i - 1][0] - 2 * dists[i][0], current_bearing])

                error = min(abs(current_bearing - target_bearing), abs(target_bearing - current_bearing + 360))

                if time.time() - initial_time > 3 and error < 5:
                    m.stop_all()
                    break
                print(f"Dist: {current_dist:.2f}\tBearing: {current_bearing:.2f}\tError: {error:.2f}")

            largest_i = [0, 0]
            for i in range(len(dists_sec_deriv)):
                if dists_sec_deriv[i][0] > 120:
                    if dists_sec_deriv[i - 1][0] < dists_sec_deriv[i][0] and dists_sec_deriv[i + 1][0] < dists_sec_deriv[i][0]:
                        if dists_sec_deriv[largest_i[0]][0] < dists_sec_deriv[i][0]:
                            if dists_sec_deriv[largest_i[1]][0] < dists_sec_deriv[largest_i[0]][0]:
                                largest_i[1] = largest_i[0]
                            largest_i[0] = i
                        elif dists_sec_deriv[largest_i[1]][0] < dists_sec_deriv[i][0]:
                            largest_i[1] = i

            potential_exits = [
                dists_sec_deriv[largest_i[0]][1],
                dists_sec_deriv[largest_i[1]][1]
            ]

            # Pick which exit based on entry evac bearing. Potential exits - 180 degrees should be the entry bearing, so pick the one furthest from that
            potential_exits_offsets = [(x - 180) % 360 for x in potential_exits]
            bearing_diffs = [abs(exit_bearing - start_of_evac_bearing) for exit_bearing in potential_exits_offsets]
            sorted_exits = [x for _, x in sorted(zip(bearing_diffs, potential_exits), key=lambda pair: pair[0], reverse=False)]

            align_to_bearing(sorted_exits[0], 2, 10, "EXIT TARGET: ")

            m.stop_all()
            time.sleep(0.6)
            m.run_tank_for_time(-35, 35, 100)
            time.sleep(0.3)
            if tof.range_mm < 500:
                m.run_tank_for_time(35, -35, 130)

            print("Potential Exits: ", potential_exits)
            print("Potential Exits Offset:", potential_exits_offsets)
            print("Bearing Diffs:", bearing_diffs)
            print("Sorted Exits:", sorted_exits)
            print("Bearing Dif UNPREF: ", (sorted_exits[1] - start_of_evac_bearing) % 360)
            print("Bearing Dif TARGET: ", (sorted_exits[0] - start_of_evac_bearing) % 360)

            m.run_tank_for_time(35, 35, 1000, True)
            m.run_tank(35, 35)
            found_line = False
            start_time_exit = time.time()
            while True:
                if time.time() - start_time_exit > 1.8:
                    m.stop_all()
                    time.sleep(2)
                    break

                if tof.range_mm < 60:
                    # We hit a wall... back up and find the exit again
                    m.run_tank_for_time(-35, -35, 800)
                    
                    # If the bearing is greater than the target, go left until see the exit
                    if (sorted_exits[0] - cmps.read_bearing_16bit()) % 360 > 180:
                        m.run_tank(-35, 35)
                        while tof.range_mm < 500:
                            pass
                        m.stop_all()
                        m.run_tank_for_time(-35, 35, 200)
                    else:
                        m.run_tank(35, -35)
                        while tof.range_mm < 500:
                            pass
                        m.stop_all()
                        m.run_tank_for_time(35, -35, 200)
                    
                    m.run_tank_for_time(35, 35, 400)
                    time.sleep(1)
                    break

                frame_processed = cam.read_stream_processed()
                img0 = frame_processed["resized"]
                img0_line = frame_processed["line"]
                black_contours, _ = cv2.findContours(cv2.bitwise_not(img0_line), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                black_contours_sorted = sorted([[c, cv2.contourArea(c)] for c in black_contours], key=lambda x: x[1], reverse=True)

                meets_target = False
                if len(black_contours_sorted) > 0:
                    largest_contour = black_contours_sorted[0]
                    meets_target = largest_contour[1] > 18000 if not found_line else 7000 < largest_contour[1] < 12000

                # For each black contour, draw a border and the area
                for i, (c, a) in enumerate(black_contours_sorted):
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.drawContours(img0, [c], -1, (0, 255, 0) if i == 0 else (0, 0, 255), 2)
                    cv2.putText(img0, str(a), (x + int(w / 2), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if meets_target else (0, 0, 255), 2)

                if found_line and meets_target:
                    m.run_tank_for_time(40, 40, 300, True)
                    break
                
                if meets_target:
                    print("FOUND LINE")
                    found_line = True

                cv2.imshow("img0", img0)

                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break
            break

# ------------------------
# WAIT FOR VISION TO START
# ------------------------
m.stop_all()
os.system("cat motd.txt")

# Ensure frames have been processed at least once
if cam.read_stream_processed()["raw"] is None:
    print("Waiting for first frame...", end="")
    while cam.read_stream_processed()["raw"] is None:
        time.sleep(0.1)
    print(" Done")

# ------------------------
# WAIT FOR FIRST INFERENCE
# ------------------------
print("Waiting for first silver inference...")
start_inf = time.time()
infer_silver(cam.read_stream_processed()["gray"])
inference_time_ms = (time.time() - start_inf) * 1000
print(f"Initial Silver Inference: {inference_time_ms:.2f} ms")

# ---------
# MAIN LOOP
# ---------
fpsTime = time.time()
while program_active:
    try:
        # ---------------
        # FRAME BALANCING
        # ---------------
        frames += 1
        fpsLoop = int(frames / (time.time() - fpsTime))
        fpsCamera = cam.get_fps()

        # if frames > 400:
        #     sleep_adjustment_amt = 0.0001
        #     sleep_time_max = 0.02
        #     sleep_time_min = 0.0001
        #     target_fps = 70
        #     if fpsLoop > target_fps + 5 and program_sleep_time < sleep_time_max:
        #         program_sleep_time += sleep_adjustment_amt
        #     elif fpsLoop < target_fps and program_sleep_time > sleep_time_min:
        #         program_sleep_time -= sleep_adjustment_amt

        # if fpsLoop > 65:
        #     time.sleep(program_sleep_time)

        if frames > 5000:
            fpsTime = time.time()
            frames = 0

        if frames % 30 == 0 and frames != 0:
            print(f"{frames:4d} {fpsLoop:3.0f} {fpsCamera:2.0f}")

        servo.cam.toMin()

        # ------------------
        # OBSTACLE AVOIDANCE
        # ------------------
        front_dist = tof.range_mm
        if 0 < front_dist < obstacle_threshold:
            print(f"Obstacle detected at {front_dist}cm... ", end="")
            m.stop_all()
            time.sleep(0.7)
            front_dist = tof.range_mm
            if 0 < front_dist < obstacle_threshold + 5:
                print("Confirmed.")
                avoid_obstacle()
            else:
                print("False positive, continuing.")
            continue

        # -------------
        # VISION HANDLING
        # -------------
        changed_black_contour = False
        frame_processed = cam.read_stream_processed()

        # We need to .copy() the images because we are going to be modifying them
        # This prevents us from reading a modified image on the next loop, and things breaking

        img0 = frame_processed["resized"].copy()
        img0_raw = frame_processed["raw"].copy()
        img0_clean = img0.copy() # Used for displaying the image without any overlays

        img0_gray = frame_processed["gray"].copy()
        img0_silver = frame_processed["silver"].copy()
        img0_silver_binary = frame_processed["silver_binary"].copy()
        # img0_gray_scaled = frame_processed["gray_scaled"].copy()
        img0_binary = frame_processed["binary"].copy()
        img0_hsv = frame_processed["hsv"].copy()
        img0_green = frame_processed["green"].copy()
        img0_line = frame_processed["line"].copy()

        img0_line_not = cv2.bitwise_not(img0_line)

        # The tiles at the competition have white gaps in between, which caused issues in the first round
        # This ensures that while responding to a green turn, all lines will be connected
        if turning:
            img0_line = cv2.erode(img0_line, np.ones((5, 5), np.uint8), iterations=turning_line_iterations)
            img0_line = cv2.dilate(img0_line, np.ones((5, 5), np.uint8), iterations=turning_line_iterations)

        current_pitch = cmps.read_pitch()

        if min_pitch_ramp_up_slight < current_pitch < min_pitch_ramp_up_full + 20:
            img0_binary[0:int(img0_binary.shape[0] * 0.4), :] = 255
            img0_line[0:int(img0_line.shape[0] * 0.4), :] = 255

        is_ramp_up = min_pitch_ramp_up_slight < current_pitch < min_pitch_ramp_up_full + 20

        # If we've been ramping up for more than 1 second, start to slowly extend the lift for weight distribution
        ramp_lift_thresh = 1
        if is_ramp_up and time_since_ramp_start > 0 and time.time() - time_since_ramp_start > ramp_lift_thresh:
            ramp_lift_factor = 4 # seconds to go from min to max
            r_min = servo.lift.r_min
            r_max = (servo.lift.r_max - servo.lift.r_min) / 2 # only go half-way
            
            lift_target = r_min - ((r_min - r_max) * ((time.time() - time_since_ramp_start - ramp_lift_thresh) / ramp_lift_factor))
            if lift_target < r_min:
                lift_target = r_min
            if lift_target > r_max:
                lift_target = r_max
            
            servo.lift.to(lift_target)
        else:
            servo.lift.toMin() # Otherwise, keep the lift up            

        raw_white_contours, _ = cv2.findContours(img0_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_contours, _ = cv2.findContours(img0_line_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if debug_state():
            cv2.imshow("img0", img0)

        # ----------------
        # CHECK FOR SILVER
        # ----------------
        silver_inference_start = time.time()
        if silver_prediction > 0 or frames % 2 == 0:
            if infer_silver(img0_silver):
                # We need to run some extra checks, just in case the model returned a false positive
                # As silver makes the image output inconsistent, we can check that:
                # - The top 20px of the image must all be white
                # - A black comtour that touches the bottom of the image must exist
                # - Very little to no green is in the image
                # - The TOF sensor is not detecting anything within 10cm
                # - There are more than 3 black contours above 500px in area
                # - The current pitch of the robot is within pitch_flat +- pitch_flat_error

                black_contours_silver, _ = cv2.findContours(cv2.bitwise_not(img0_silver_binary), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                top_check_threshold = 20
                sides_touching = []
                black_filtered = [c for c in black_contours_silver if cv2.contourArea(c) > 500]
                for black_rect in [cv2.boundingRect(c) for c in black_filtered]:
                    if "bottom" not in sides_touching and black_rect[1] + black_rect[3] > img0_silver_binary.shape[0] - 3:
                        sides_touching.append("bottom")
                    if "left" not in sides_touching and black_rect[0] < 20:
                        sides_touching.append("left")
                    if "right" not in sides_touching and black_rect[0] + black_rect[2] > img0_silver_binary.shape[1] - 20:
                        sides_touching.append("right")
                    if "top" not in sides_touching and black_rect[1] < 20:
                        sides_touching.append("top")
                total_green_area = np.count_nonzero(img0_green == 0)

                checks = [
                    sum([sum(a) for a in img0_silver_binary[0:top_check_threshold]]) == img0_silver_binary.shape[1] * 255 * top_check_threshold,
                    "bottom" in sides_touching,
                    total_green_area < 500,
                    tof.range_mm > 100,
                    len(black_filtered) >= 2,
                    pitch_flat - pitch_flat_error < current_pitch < pitch_flat + pitch_flat_error
                ]

                if sum(checks) == len(checks):
                    silver_prediction += 1
                    cv2.putText(img0_silver, "CONFIRM", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)
                else:
                    silver_prediction = 0

                cv2.putText(img0_silver, "SILVER", (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

                cv2.putText(
                    img0_silver, 
                    ' '.join([chr(65 + i) if check else '-' for i, check in enumerate(checks)]),
                    (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, 
                    (66, 32, 11), 
                    2
                )
                cv2.putText(img0_silver, "Green Area: " + str(total_green_area), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 32, 11), 2)
                cv2.putText(img0_silver, "Sides: " + str(sides_touching), (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 32, 11), 2)
                cv2.putText(img0_silver, f"Blk Num: {len(black_contours)}/{len(black_filtered)}", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 32, 11), 2)

                cv2.putText(img0_silver, f"{silver_prediction}/5", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                silver_prediction = 0
            silver_inference_time_ms = (time.time() - silver_inference_start) * 1000
        else:
            silver_prediction = 0
            silver_inference_time_ms = 0

        if silver_prediction >= 5:
            print("Silver Confirmed")
            run_evac()
            silver_prediction = 0
            continue
        elif silver_prediction > 1:
            print(f"Silver Detected - {silver_prediction}/5") 
            m.run_tank(15, 15)

            cv2.imshow("img0_silver / img0_red", img0_silver)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                program_active = False
                break
            continue

        # -----------

        # Filter white contours based on area
        white_contours = []
        for contour in raw_white_contours:
            if cv2.contourArea(contour) > 1000:
                white_contours.append(contour)

        if len(white_contours) == 0:
            print("No white contours found")
            continue

        if len(black_contours) == 0:
            is_broken = False
            new_steer = current_steering
            if time.time() - last_break_in_line < 3:
                new_steer = 0

                is_broken = True
                cv2.putText(img0, "BROKEN LINE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif new_steer < -50: new_steer = -100
            elif new_steer > 50: new_steer = 100

            m.run_steer(follower_speed, 100, new_steer)

            cv2.putText(img0, f"Steer: {int(new_steer)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if debug_state():
                cv2.imshow("img0_nbc", img0)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break
                
            print(f"No black contours found | {int(time.time() - last_break_in_line)} | Steer: {new_steer:.2f}" + (" | BROKEN" if is_broken else ""))
            continue

        # -----------
        # GREEN TURNS
        # -----------
        img0_line_new = img0_line.copy()
        total_green_area = np.count_nonzero(img0_green == 0)

        if total_green_area > (4000 if not turning else 1000):
            changed_img0_line = None

            unfiltered_green_contours, _ = cv2.findContours(cv2.bitwise_not(img0_green), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            green_contours_filtered = [contour for contour in unfiltered_green_contours if cv2.contourArea(contour) > 1000]
            white_contours_filtered = [contour for contour in white_contours if cv2.contourArea(contour) > 500]

            followable_green = []
            for g_contour in green_contours_filtered:
                # cv2.drawContours(img0, [g_contour], -1, (0, 255, 0), 2)

                # Find which white contour contains the green contour
                containing_white_contour = None
                for w_contour in white_contours_filtered:
                    if cv2.pointPolygonTest(w_contour, ck.centerOfContour(g_contour), False) > 0:
                        containing_white_contour = w_contour
                        break

                if containing_white_contour is not None:
                    w_bounding_rect = cv2.boundingRect(containing_white_contour)
                    # Check that the white contour touches the bottom of the screen, if not, we can ignore this green contour
                    if w_bounding_rect[1] + w_bounding_rect[3] >= img0.shape[0] - 3:
                        # Let's follow this green turn. Mark it for processing
                        followable_green.append({
                            "g": g_contour,
                            "w": containing_white_contour,
                            "w_bounds": w_bounding_rect,
                        })
                        if len(followable_green) >= 2:
                            break # There should never be more than 2 followable green contours, so we can stop looking for more

            if len(followable_green) == 2 and not turning:
                # We have found 2 followable green contours, this means we need turn around 180 degrees
                print("DOUBLE GREEN")
                m.run_tank_for_time(40, 40, 400)
                start_bearing = cmps.read_bearing_16bit()
                align_to_bearing(start_bearing - 180, 10, debug_prefix="Double Green Rotate - ")
                m.run_tank_for_time(40, 40, 200)
                turning = "RIGHT"
                continue

            if len(followable_green) >= 2 and turning:
                followable_green.sort(key=lambda x: x["w_bounds"][0])
                if turning == "LEFT":
                    # If we are turning left, we want the leftmost green contour
                    followable_green = followable_green[:1]
                elif turning == "RIGHT":
                    # If we are turning right, we want the rightmost green contour
                    followable_green = followable_green[-1:]

            if len(followable_green) == 1:
                can_follow_green = True
                if not turning:
                    # With double green, we may briefly see only 1 green contour while entering.
                    # Hence, add some delay to when we start turning to prevent this and ensure we can see all green contours
                    if time.time() - initial_green_time < 1: initial_green_time = time.time() # Reset the initial green time
                    if time.time() - initial_green_time < 0.3: can_follow_green = False

                    # As an additional check, make sure that a black contour touches the side of the screen before starting a turn
                    # This helps the above condition to make sure we don't miss double green
                    black_touches_side = False
                    for black_rect in [cv2.boundingRect(c) for c in black_contours if cv2.contourArea(c) > 3000]:
                        if black_rect[0] < 3 or black_rect[0] + black_rect[2] > img0_binary.shape[1] - 3:
                            black_touches_side = True
                            break

                    if not black_touches_side:
                        can_follow_green = False 

                if can_follow_green:
                    selected = followable_green[0]
                    # Dilate selected["w"] to make it larger, and then use it as a mask
                    img_black = np.zeros((img0.shape[0], img0.shape[1]), np.uint8)
                    cv2.drawContours(img_black, [selected["w"]], -1, 255, 100)

                    # Mask the line image with the dilated white contour
                    img0_line_new = cv2.bitwise_and(cv2.bitwise_not(img0_line), img_black)
                    # Erode the line image to remove slight inconsistencies we don't want
                    img0_line_new = cv2.erode(img0_line_new, np.ones((3, 3), np.uint8), iterations=2)

                    changed_img0_line = img0_line_new

                    last_green_time = time.time()
                    if not turning:
                        # Based on the centre location of the white contour, we are either turning left or right
                        if selected["w_bounds"][0] + selected["w_bounds"][2] / 2 < img0.shape[1] / 2:
                            print("Start Turn: left")
                            turning = "LEFT"
                        else:
                            print("Start Turn: right")
                            turning = "RIGHT"

            if changed_img0_line is not None:
                print("Green caused a change in the line")
                img0_line_new = changed_img0_line
                new_black_contours, _ = cv2.findContours(img0_line_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img0, new_black_contours, -1, (0, 0, 255), 2)
                if len(new_black_contours) > 0:
                    black_contours = new_black_contours
                else:
                    print("No black contours found after changing contour")

            print("GREEN TURN STUFF")
        elif turning is not None and last_green_time + 0.4 < time.time():
            turning = None
            print("No longer turning")

        # -----------------
        # STOP ON RED CHECK
        # -----------------
        img0_red = cv2.inRange(img0_hsv, config.get("red_hsv_threshold")[0], config.get("red_hsv_threshold")[1])
        img0_red = cv2.dilate(img0_red, np.ones((5, 5), np.uint8), iterations=2)

        red_contours = [[contour, cv2.contourArea(contour)] for contour in cv2.findContours(img0_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]]
        red_contours = sorted(red_contours, key=lambda contour: contour[1], reverse=True)
        red_contours_filtered = [contour[0] for contour in red_contours if contour[1] > 20000]

        if len(red_contours_filtered) > 0:
            edges = sorted(ck.getTouchingEdges(ck.simplifiedContourPoints(red_contours_filtered[0]), img0_binary.shape))
            if edges == ["left", "right"]:
                m.stop_all()
                red_stop_check += 1
                print(f"RED IDENTIFIED - {red_stop_check}/3 tries")

                if red_stop_check == 1:
                    time.sleep(0.1)
                    m.run_tank_for_time(-40, -40, 100)
                    time.sleep(0.1)

                time.sleep(7)

                if debug_state():
                    cv2.imshow("img0_silver / img0_red", img0_red)
                    cv2.waitKey(1)

                if red_stop_check > 3:
                    print("DETECTED RED STOP 3 TIMES, STOPPING")
                    break

                continue # Don't run the rest of the follower, we don't really want to move forward in case we accidentally loose the red...
            red_stop_check = 0

        # -------------
        # INTERSECTIONS
        # -------------
        if not turning:
            # Filter white contours to have a minimum area before we accept them
            white_contours_filtered = [contour for contour in white_contours if cv2.contourArea(contour) > 500]
            
            # Check for a break in the line
            # This only occurs when:
            # - There is exactly 1 black and 1 white contour
            # - The previous steering is within [-20, 20]
            # - The black contour touches the bottom of the image
            # - The black contour does not touch the top of the image by at least 30%
            # - The black contour is at least 20% away from the sides of the image
            if len(white_contours_filtered) == 1:
                black_rect = cv2.boundingRect(black_contours[0])

                # Draw rect
                cv2.rectangle(img0, (black_rect[0], black_rect[1]), (black_rect[0] + black_rect[2], black_rect[1] + black_rect[3]), (0, 0, 255), 2)
                A = "A" if -30 < current_steering < 30 else "-"
                B = "B" if black_rect[1] + black_rect[3] > img0_binary.shape[0] - 5 else "-"
                C = "C" if black_rect[1] > img0_binary.shape[0] * 0.3 else "-"
                D = "D" if black_rect[0] > img0_binary.shape[1] * 0.2 else "-"
                E = "E" if black_rect[0] + black_rect[2] < img0_binary.shape[1] * (1 - 0.2) else "-"

                cv2.putText(img0, f"{A} {B} {C} {D} {E}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if (
                    len(black_contours) == 1
                    and -30 < current_steering < 30 
                    and black_rect[1] + black_rect[3] > img0_binary.shape[0] - 5
                    and black_rect[1] > img0_binary.shape[0] * 0.3
                    and black_rect[0] > img0_binary.shape[1] * 0.2
                    and black_rect[0] + black_rect[2] < img0_binary.shape[1] * (1 - 0.2)
                ):
                    last_break_in_line = time.time()

            elif len(white_contours_filtered) == 2:
                contour_L = white_contours_filtered[0]
                contour_R = white_contours_filtered[1]

                if ck.centerOfContour(contour_L) > ck.centerOfContour(contour_R):
                    contour_L = white_contours_filtered[1]
                    contour_R = white_contours_filtered[0]

                # Simplify contours to get key points
                contour_L_simple = ck.simplifiedContourPoints(contour_L, 0.03)
                contour_R_simple = ck.simplifiedContourPoints(contour_R, 0.03)

                if len(contour_L_simple) < 2 or len(contour_R_simple) < 2:
                    print("2WC NOT ENOUGH POINTS?")
                    continue

                contour_L_vert_sort = sorted(contour_L_simple, key=lambda point: point[1])
                contour_R_vert_sort = sorted(contour_R_simple, key=lambda point: point[1])

                if current_linefollowing_state is None or "2-top-ng" in current_linefollowing_state:
                    contour_L_2_top = contour_L_vert_sort[:2]
                    contour_R_2_top = contour_R_vert_sort[:2]

                    combined_top_points = contour_L_2_top + contour_R_2_top
                    top_dists = [p[1] for p in combined_top_points]

                    # If all points are near the top, then we should check if we are at an intersection
                    if sum(d < 60 for d in top_dists) == 4:
                        # If none of the following are true, then make the top of the image white, anywhere above the lowest point
                        #   - All points at the top
                        #   - Left top points are at the top, right top points are close to the top (disabled for now, as it breaks entry to 3WC)
                        #   - Right top points are at the top, left top points are close to the top (disabled for now, as it breaks entry to 3WC)
                        if (
                            not sum(d < 3 for d in top_dists) == 4
                            # and not (sum(d < 3 for d in top_dists[:2]) == 2 and sum(12 < d and d < 80 for d in top_dists[2:]) == 2)
                            # and not (sum(d < 3 for d in top_dists[2:]) == 2 and sum(12 < d and d < 80 for d in top_dists[:2]) == 2)
                        ):
                            current_linefollowing_state = "2-top-ng"
                            lowest_point = sorted(combined_top_points, key=lambda point: point[1])[-1]

                            img0_line_new[:lowest_point[1] + 3, :] = 255
                            changed_black_contour = cv2.bitwise_not(img0_line_new)
                        else:
                            current_linefollowing_state = None
                    else:
                        current_linefollowing_state = None
                else: # We are exiting an intersection
                    contour_L_2_bottom = contour_L_vert_sort[2:]
                    contour_R_2_bottom = contour_R_vert_sort[2:]

                    combined_bottom_points = contour_L_2_bottom + contour_R_2_bottom
                    bottom_dists = [img0.shape[0] - p[1] for p in combined_bottom_points]

                    # If all points are at the bottom, we probably exited the intersection
                    if sum(d < 3 for d in bottom_dists) == 4:
                        current_linefollowing_state = None
                        print("Exited intersection")
                    # If all points are still near the bottom, since we already know we are existing an intersection, remove the bottom to prevent the robot from turning
                    elif sum(d < 120 for d in bottom_dists) == 4:
                        current_linefollowing_state = "2-btm-ng"
                        highest_point = sorted(combined_bottom_points, key=lambda point: point[1])[0]

                        img0_line_new[highest_point[1] - 3:, :] = 255
                        changed_black_contour = cv2.bitwise_not(img0_line_new)
                    # If we are not at the bottom, then we are probably still in the intersection... we shouldn't really end up here, so just reset the state
                    else:
                        current_linefollowing_state = None
                        print("Exited intersection - but not really?")

            elif len(white_contours_filtered) == 3:
                # We are entering a 3-way intersection
                if not current_linefollowing_state or "2-ng" in current_linefollowing_state:
                    current_linefollowing_state = "3-ng-en"
                # We are exiting a 4-way intersection
                if "4-ng" in current_linefollowing_state:
                    current_linefollowing_state = "3-ng-4-ex"

                # Make sure no white contour touches all sides of the screen
                # If one does, this is not a 3-way intersection
                valid_3wc = True
                for contour in white_contours_filtered:
                    edges = ck.getTouchingEdges(ck.simplifiedContourPoints(contour), img0_binary.shape)
                    if len(edges) == 4:
                        valid_3wc = False
                        print("Invalid 3WC - Contour touches all sides")
                        break
                
                if not valid_3wc:
                    # Find all the black contours that still touch the bottom of the screen
                    black_bottom_touching = []
                    for black_contour in black_contours:
                        black_rect = cv2.boundingRect(black_contour)
                        if black_rect[1] + black_rect[3] >= img0_binary.shape[0] - 3:
                            black_bottom_touching.append(black_contour)
                    
                    if len(black_bottom_touching) > 0:
                        black_contours = black_bottom_touching
                        print("Restricted black contours to only those touching the bottom of the screen")
                else:
                    intersection_state_debug = ["3-ng", time.time()]
                    # Get the center of each contour
                    white_contours_filtered_with_center = [(contour, ck.centerOfContour(contour)) for contour in white_contours_filtered]

                    # Sort the contours from left to right - Based on the centre of the contour's horz val
                    sorted_contours_horz = sorted(white_contours_filtered_with_center, key=lambda contour: contour[1][0])

                    # Simplify the contours to get the corner points
                    approx_contours = [ck.simplifiedContourPoints(contour[0], 0.03) for contour in sorted_contours_horz]

                    # Middle of contour centres
                    mid_point = (
                        int(sum(contour[1][0] for contour in sorted_contours_horz) / len(sorted_contours_horz)),
                        int(sum(contour[1][1] for contour in sorted_contours_horz) / len(sorted_contours_horz))
                    )

                    # Get the closest point of the approx contours to the mid point
                    def closestPointToMidPoint(approx_contour, mid_point):
                        return sorted(approx_contour, key=lambda point: ck.pointDistance(point, mid_point))[0]

                    # Get the closest points of each approx contour to the mid point, and store the index of the contour to back reference later
                    closest_points = [
                        [closestPointToMidPoint(approx_contour, mid_point), i]
                        for i, approx_contour in enumerate(approx_contours)
                    ]

                    # Get the closest points, sorted by distance to mid point
                    sorted_closest_points = sorted(closest_points, key=lambda point: ck.pointDistance(point[0], mid_point))
                    closest_2_points_vert_sort = sorted(sorted_closest_points[:2], key=lambda point: point[0][1])

                    # If a point is touching the top/bottom of the screen, it is quite possibly invalid and will cause some issues with cutting
                    # So, we will find the next best point, the point inside the other contour that is at the top of the screen, and is closest to the X value of the other point
                    for i, point in enumerate(closest_2_points_vert_sort):
                        if point[0][1] > img0_line_new.shape[0] - 10 or point[0][1] < 10:
                            # Find the closest point to the x value of the other point
                            other_point_x = closest_2_points_vert_sort[1 - i][0][0]
                            other_point_approx_contour_i = closest_2_points_vert_sort[1 - i][1]

                            closest_points_to_other_x = sorted(
                                approx_contours[other_point_approx_contour_i],
                                key=lambda point, x=other_point_x: abs(point[0] - x)
                            )
                            new_valid_points = [
                                point for point in closest_points_to_other_x
                                if not np.isin(point, [
                                    closest_2_points_vert_sort[0][0],
                                    closest_2_points_vert_sort[1][0]
                                ]).any()
                            ]
                            if len(new_valid_points) == 0:
                                # print(f"Point {i} is at an edge, but no new valid points were found")
                                continue

                            closest_2_points_vert_sort = sorted([[new_valid_points[0], other_point_approx_contour_i], closest_2_points_vert_sort[1 - i]], key=lambda point: point[0][1])
                            # print(f"Point {i} is at an edge, replacing with {new_valid_points[0]}")

                    split_line = [point[0] for point in closest_2_points_vert_sort]

                    contour_center_point_sides = [[], []] # Left, Right
                    for i, contour in enumerate(sorted_contours_horz):
                        if split_line[1][0] == split_line[0][0]:  # Line is vertical, so x is constant
                            side = "right" if contour[1][0] < split_line[0][0] else "left"
                        else:
                            slope = (split_line[1][1] - split_line[0][1]) / (split_line[1][0] - split_line[0][0])
                            y_intercept = split_line[0][1] - slope * split_line[0][0]

                            if contour[1][1] < slope * contour[1][0] + y_intercept:
                                side = "left" if slope > 0 else "right"
                            else:
                                side = "right" if slope > 0 else "left"

                        contour_center_point_sides[side == "left"].append(contour[1])

                    # Get the edges that the contour not relevant to the closest points touches
                    edges_big = sorted(ck.getTouchingEdges(approx_contours[sorted_closest_points[2][1]], img0_binary.shape))

                    # Cut direction is based on the side of the line with the most contour center points (contour_center_point_sides)
                    cut_direction = len(contour_center_point_sides[0]) > len(contour_center_point_sides[1])

                    # If we are just entering a 3-way intersection, and the 'big contour' does not connect to the bottom,
                    # we may be entering a 4-way intersection... so follow the vertical line
                    if len(edges_big) >= 2 and "bottom" not in edges_big and "-en" in current_linefollowing_state:
                        edges_a = sorted(ck.getTouchingEdges(approx_contours[sorted_closest_points[0][1]], img0_binary.shape))
                        edges_b = sorted(ck.getTouchingEdges(approx_contours[sorted_closest_points[1][1]], img0_binary.shape))

                        tight_bend = False
                        if edges_a == ["bottom", "left", "top"] or edges_b == ["bottom", "left", "top"]:
                            tight_bend = True
                            cut_direction = 0
                        if edges_a == ["bottom", "right", "top"] or edges_b == ["bottom", "right", "top"]:
                            tight_bend = True
                            cut_direction = 1

                        if tight_bend:
                            # Change the cut line to be a vertical line from the mid point
                            # Avoids false positive issues a tight U bend leading to incorrect cutting
                            split_line = [(mid_point[0], 0), (mid_point[0], img0.shape[0])]
                        else:
                            cut_direction = not cut_direction

                    # We are exiting a 4-way intersection, so follow the vertical line
                    elif current_linefollowing_state == "3-ng-4-ex":
                        cut_direction = not cut_direction
                    else:
                        # We have probably actually entered now, lets stop following the vert line and do the normal thing.
                        current_linefollowing_state = "3-ng"

                        # If this is true, the line we want to follow is the smaller, perpendicular line to the large line.
                        # This case should realistically never happen, but it's here just in case.
                        if edges_big == ["bottom", "left", "right"] or edges_big == ["left", "right", "top"]:
                            cut_direction = not cut_direction
                        # If the contour not relevant to the closest points is really small (area), we are probably just entering the intersection,
                        # So we need to follow the line that is perpendicular to the large line
                        # We ignore this if edges_big does not include the bottom, because we could accidently have the wrong contour in some weird angle
                        elif cv2.contourArea(sorted_contours_horz[sorted_closest_points[2][1]][0]) < 7000 and "bottom" in edges_big:
                            cut_direction = not cut_direction

                    # CutMaskWithLine will fail if the line is flat, so we need to make sure that the line is not flat
                    if split_line[0][1] == split_line[1][1]:
                        split_line[0][1] += 1 # Move the first point up by 1 pixel

                    img0_line_new = intersections.CutMaskWithLine(closest_2_points_vert_sort[0][0], closest_2_points_vert_sort[1][0], img0_line_new, "left" if cut_direction else "right")
                    img0_line_new = intersections.CutMaskWithLine(split_line[0], split_line[1], img0_line_new, "left" if cut_direction else "right")
                    changed_black_contour = cv2.bitwise_not(img0_line_new)

            elif len(white_contours_filtered) == 4:
                intersection_state_debug = ["4-ng", time.time()]
                # Get the center of each contour
                white_contours_filtered_with_center = [(contour, ck.centerOfContour(contour)) for contour in white_contours_filtered]

                # Sort the contours from left to right - Based on the centre of the contour's horz val
                sorted_contours_horz = sorted(white_contours_filtered_with_center, key=lambda contour: contour[1][0])

                # Sort the contours from top to bottom, for each side of the image - Based on the centre of the contour's vert val
                contour_BL, contour_TL = tuple(sorted(sorted_contours_horz[:2], reverse=True, key=lambda contour: contour[1][1]))
                contour_BR, contour_TR = tuple(sorted(sorted_contours_horz[2:], reverse=True, key=lambda contour: contour[1][1]))

                # Simplify the contours to get the corner points
                approx_BL = ck.simplifiedContourPoints(contour_BL[0], 0.03)
                approx_TL = ck.simplifiedContourPoints(contour_TL[0], 0.03)
                approx_BR = ck.simplifiedContourPoints(contour_BR[0], 0.03)
                approx_TR = ck.simplifiedContourPoints(contour_TR[0], 0.03)

                # Middle of contour centres
                mid_point = (
                    int((contour_BL[1][0] + contour_TL[1][0] + contour_BR[1][0] + contour_TR[1][0]) / 4),
                    int((contour_BL[1][1] + contour_TL[1][1] + contour_BR[1][1] + contour_TR[1][1]) / 4)
                )

                # Get the closest point of the approx contours to the mid point
                def closestPointToMidPoint(approx_contour, mid_point):
                    return sorted(approx_contour, key=lambda point: ck.pointDistance(point, mid_point))[0]

                closest_BL = closestPointToMidPoint(approx_BL, mid_point)
                closest_TL = closestPointToMidPoint(approx_TL, mid_point)
                closest_BR = closestPointToMidPoint(approx_BR, mid_point)
                closest_TR = closestPointToMidPoint(approx_TR, mid_point)

                # If closest_TL or closest_TR is touching the top of the screen, it is quite possibly invalid and will cause some issues with cutting
                # So, we will find the next best point, the point inside the relevant contour, and is closest to the X value of the other point
                if closest_TL[1] < 10:
                    closest_TL = closest_BL
                    closest_BL = sorted(approx_BL, key=lambda point: abs(point[0] - closest_BL[0]))[1]
                elif closest_BL[1] > img0_binary.shape[0] - 10:
                    closest_BL = closest_TL
                    closest_TL = sorted(approx_TL, key=lambda point: abs(point[0] - closest_TL[0]))[1]
                # # We will do the same with the right-side contours
                if closest_TR[1] < 10:
                    closest_TR = closest_BR
                    closest_BR = sorted(approx_BR, key=lambda point: abs(point[0] - closest_BR[0]))[1]
                elif closest_BR[1] > img0_binary.shape[0] - 10:
                    closest_BR = closest_TR
                    closest_TR = sorted(approx_TR, key=lambda point: abs(point[0] - closest_TR[0]))[1]

                img0_line_new = intersections.CutMaskWithLine(closest_BL, closest_TL, img0_line_new, "left")
                img0_line_new = intersections.CutMaskWithLine(closest_BR, closest_TR, img0_line_new, "right")

                current_linefollowing_state = "4-ng"
                changed_black_contour = cv2.bitwise_not(img0_line_new)

            if changed_black_contour is not False:
                print("Changed black contour, LF State: ", current_linefollowing_state)
                cv2.drawContours(img0, black_contours, -1, (0, 0, 255), 2)
                new_black_contours, _ = cv2.findContours(changed_black_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(new_black_contours) > 0:
                    black_contours = new_black_contours
                else:
                    print("No black contours found after changing contour")

        # --------------------------
        # REST OF LINE LINE FOLLOWER
        # --------------------------

        #Find the black contours
        sorted_black_contours = ck.findBestContours(black_contours, black_contour_threshold, last_line_pos)
        if len(sorted_black_contours) == 0 and silver_prediction > 0:
            print(f"No black contours, but silver was found ({int(silver_prediction)})")
            m.stop_all()
            time.sleep(0.2)
            continue
        elif len(sorted_black_contours) == 0:
            is_broken = False
            new_steer = current_steering
            if time.time() - last_break_in_line < 2:
                if is_ramp_up or time.time() < time_ramp_end: # On a ramp? Account for slip so allow slight steering instead of 0
                    if current_steering > 20: new_steer = 15
                    elif current_steering < -20: new_steer = -15
                else:
                    new_steer = 0

                is_broken = True
                cv2.putText(img0, f"BROKEN LINE {round(time.time() - last_break_in_line, 1)}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            elif new_steer < -50: new_steer = -100
            elif new_steer > 50: new_steer = 100

            # If we lost the line on a ramp, it's likely we are at the end of the ramp
            if is_ramp_up or time.time() < time_ramp_end: 
                if new_steer < -30: new_steer = -100
                elif new_steer > 30: new_steer = 100
                print("No black contours found (SORT/RAMP)")

            m.run_steer(follower_speed, 100, new_steer)

            cv2.putText(img0, f"Steer: {int(new_steer)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if debug_state():
                cv2.imshow("img0_nbc", img0)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break
                
            print(f"No black contours found (SORT) | {int(time.time() - last_break_in_line)} | Steer: {new_steer:.2f}" + (" | BROKEN" if is_broken else ""))
            continue

        no_black_contours = False

        chosen_black_contour = sorted_black_contours[0]

        # Update the reference position for subsequent calculations
        last_line_pos = np.array([chosen_black_contour[1][0][0], chosen_black_contour[1][0][1]])

        # Retrieve the four courner points of the chosen contour
        black_bounding_box = np.intp(cv2.boxPoints(chosen_black_contour[1]))

        # Error (distance from the center of the image) and angle (of the line) of the chosen contour
        black_contour_error = int(last_line_pos[0] - (img0.shape[1] / 2))

        # Sort the black bounding box points based on their y-coordinate (bottom to top)
        vert_sorted_black_bounding_points = sorted(black_bounding_box, key=lambda point: -point[1])

        # Find the bottom left, and top right points
        black_bounding_box_BL = sorted(vert_sorted_black_bounding_points[:2], key=lambda point: point[0])[0]
        black_bounding_box_TR = sorted(vert_sorted_black_bounding_points[-2:], key=lambda point: point[0])[1]

        # Get the angle of the line between the bottom left and top right points
        black_contour_angle = int(math.degrees(math.atan2(black_bounding_box_TR[1] - black_bounding_box_BL[1], black_bounding_box_TR[0] - black_bounding_box_BL[0])))
        black_contour_angle_new = black_contour_angle + 80

        # The two top-most points, sorted from left to right
        horz_sorted_black_bounding_points_top_2 = sorted(vert_sorted_black_bounding_points[2:], key=lambda point: point[0])

        # If the angle of the contour is big enough and the contour is close to the edge of the image (within bigTurnSideMargin pixels)
        # Then, the line likely is a big turn and we will need to turn more
        # 0 if None, 1 if left, 2 if right
        bigTurnSideMargin = 30
        bigTurnAngleMargin = 30

        lineHitsEdge = 0
        if horz_sorted_black_bounding_points_top_2[0][0] < bigTurnSideMargin:
            lineHitsEdge = 1
        elif horz_sorted_black_bounding_points_top_2[1][0] > img0.shape[1] - bigTurnSideMargin:
            lineHitsEdge = 2

        isBigTurn = lineHitsEdge and abs(black_contour_angle_new) > bigTurnAngleMargin

        if isBigTurn and ((lineHitsEdge == 1 and black_contour_angle_new > 0) or
                          (lineHitsEdge == 2 and black_contour_angle_new < 0)):
            black_contour_angle_new = black_contour_angle_new * -1

        # The closer the topmost point is to the bottom of the screen, the more we want to turn
        topmost_point = sorted(black_bounding_box, key=lambda point: point[1])[0]
        extra_pos = (topmost_point[1] / img0.shape[0]) * 100

        extra_mult = 0
        if isBigTurn and extra_pos > 25:
            extra_mult = 0.1 * extra_pos
        elif lineHitsEdge and extra_pos > 60:
            extra_mult = 0.07 * extra_pos
        elif lineHitsEdge and extra_pos > 35:
            extra_mult = 0.03 * extra_pos

        chosen_error_weight = 0.5 if not is_ramp_up else 0.7
        current_position = (black_contour_angle_new / max_angle) * (1 - chosen_error_weight) + (black_contour_error / max_error) * chosen_error_weight
        current_position *= 100 * max(extra_mult, 1)

        # PID stuff
        error = -current_position

        timeDiff = time.time() - current_time
        if timeDiff == 0:
            timeDiff = 1 / 10
        proportional = KP * error
        pid_integral += KI * error * timeDiff
        derivative = KD * (error - pid_last_error) / timeDiff
        current_steering = -(proportional + pid_integral + derivative)

        pid_last_error = error
        current_time = time.time()

        motor_vals = None

        if is_ramp_up:
            if time_since_ramp_start == 0:
                time_since_ramp_start = time.time()
            print(f"RAMP ({int(time.time() - time_since_ramp_start)})")
            if time.time() - time_since_ramp_start > 15:
                motor_vals = m.run_tank(100, 100)
            elif time.time() - time_since_ramp_start > 8:
                motor_vals = m.run_steer(80, 100, current_steering, ramp="active")
            else:
                motor_vals = m.run_steer(follower_speed, 100, current_steering, ramp="active")
        else:
            if time_since_ramp_start > 3:
                time_ramp_end = time.time() + 1

            time_since_ramp_start = 0
            if time.time() < time_ramp_end:
                print("END RAMP")
                motor_vals = m.run_steer(follower_speed, 100, current_steering, ramp="ending")

        if motor_vals is None:
            motor_vals = m.run_steer(follower_speed, 100, current_steering)

        # ----------
        # DEBUG INFO
        # ----------
        print(f"{frames:4d} {fpsLoop:3.0f} {fpsCamera:2.0f}"
            # + f" @{(program_sleep_time*1000):3.1f}"
            + (f"  S: {int(silver_prediction) or ' '} {silver_inference_time_ms:5.2f}ms" if silver_inference_time_ms > 0 else " "*14)
            + f" |"
            + f"  An:{black_contour_angle:4d}/{black_contour_angle_new:4d}"
            + f"  Er:{black_contour_error:4d}"
            + f"  Po:{int(current_position):4d}"
            + f"  Ex:{int(extra_pos):4d}"
            + f"  BT: {isBigTurn:1d}-{extra_mult:4.2f}"
            + f"  ST:{int(current_steering):4d}"
            + f" = [{motor_vals[0]:4.0f}, {motor_vals[1]:4.0f}]"
            + f"  LF: {'B' if changed_black_contour is not False else '-'} {(current_linefollowing_state or '-'):8}"
            + f"  TP:{int(topmost_point[1]):3d}"
            + f"  OB:{int(front_dist):3d}"
            + f"  PI:{int(current_pitch):3d}"
            + f"  GR:{total_green_area:5d}")

        if debug_state():
            preview_image_img0_contours = img0_clean.copy()
            cv2.drawContours(preview_image_img0_contours, white_contours, -1, (255, 0, 0), 3)
            cv2.drawContours(preview_image_img0_contours, black_contours, -1, (0, 255, 0), 3)
            cv2.drawContours(preview_image_img0_contours, [chosen_black_contour[2]], -1, (0, 0, 255), 3)

            # Draw black_bounding_box
            cv2.line(preview_image_img0_contours, black_bounding_box[0], black_bounding_box[1], (125, 200, 0), 2)
            cv2.line(preview_image_img0_contours, black_bounding_box[1], black_bounding_box[2], (125, 200, 0), 2)
            cv2.line(preview_image_img0_contours, black_bounding_box[2], black_bounding_box[3], (125, 200, 0), 2)
            cv2.line(preview_image_img0_contours, black_bounding_box[3], black_bounding_box[0], (125, 200, 0), 2)

            cv2.circle(preview_image_img0_contours, black_bounding_box_BL, 5, (125, 200, 0), -1)
            cv2.circle(preview_image_img0_contours, black_bounding_box_TR, 5, (125, 200, 0), -1)
            cv2.putText(preview_image_img0_contours, "BL", (black_bounding_box_BL[0] - 10, black_bounding_box_BL[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 200, 0), 2)
            cv2.putText(preview_image_img0_contours, "TR", (black_bounding_box_TR[0] - 10, black_bounding_box_TR[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 200, 0), 2)

            cv2.circle(preview_image_img0_contours, horz_sorted_black_bounding_points_top_2[0], 5, (125, 125, 0), -1)
            cv2.circle(preview_image_img0_contours, horz_sorted_black_bounding_points_top_2[1], 5, (125, 125, 0), -1)

            cv2.putText(preview_image_img0_contours, f"{black_contour_angle:4d} Angle Raw", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{black_contour_angle_new:4d} Angle", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{black_contour_error:4d} Error", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{int(current_position):4d} Position", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{int(current_steering):4d} Steering", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{int(extra_pos):4d} Extra", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 125, 255), 2) # DEBUG

            if turning is not None:
                cv2.putText(preview_image_img0_contours, f"{turning} Turning", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 0, 255), 2) # DEBUG

            if isBigTurn:
                cv2.putText(preview_image_img0_contours, "Big Turn", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if silver_prediction:
                cv2.putText(img0, "SILVER", (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

            cv2.putText(preview_image_img0_contours, f"LF State: {current_linefollowing_state}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(preview_image_img0_contours, f"INT Debug: {intersection_state_debug[0]} - {int(time.time() - intersection_state_debug[1])}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(preview_image_img0_contours, f"FPS: {fpsLoop} | {fpsCamera}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            # preview_image_img0_contours = cv2.resize(preview_image_img0_contours, (0, 0), fx=0.8, fy=0.7)
            cv2.imshow("img0_contours", preview_image_img0_contours)

            cv2.imshow("img0_silver / img0_red", img0_silver)

            if not has_moved_windows:
                # Placeholder for images that are yet to show
                img0_empty = np.zeros(img0.shape, np.uint8)
                cv2.imshow("img0_nbc", img0_empty)
                cv2.imshow("img0_silver / img0_red", img0_empty)
                cv2.imshow("img0_raw", img0_empty)

            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                program_active = False
                break

            if k & 0xFF == ord('p'):
                m.stop_all()
                pids = input(f"Enter Speed, KP, KI, KD, split by spaces ({follower_speed} {KP} {KI} {KD}): ").split(" ")
                follower_speed = int(pids[0])
                KP = float(pids[1])
                KI = float(pids[2])
                KD = float(pids[3])

            if not has_moved_windows:
                cv2.moveWindow("img0", 75, 50)
                cv2.moveWindow("img0_contours", 375, 50)
                cv2.moveWindow("img0_silver / img0_red", 675, 50)
                cv2.moveWindow("img0_nbc", 975, 50)
                cv2.moveWindow("img0_raw", 1275, 50)
                has_moved_windows = True
    except Exception:
        print("UNHANDLED EXCEPTION: ")
        traceback.print_exc()
        print("Returning to start of loop in 5 seconds...")
        m.stop_all()
        time.sleep(5)

exit_gracefully()
