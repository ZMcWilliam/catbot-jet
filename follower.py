#                                ██████╗ █████╗ ████████╗██████╗  ██████╗ ████████╗    ███╗   ██╗███████╗ ██████╗
#   _._     _,-'""`-._          ██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝    ████╗  ██║██╔════╝██╔═══██╗
#   (,-.`._,'(       |\`-/|     ██║     ███████║   ██║   ██████╔╝██║   ██║   ██║       ██╔██╗ ██║█████╗  ██║   ██║
#       `-.-' \ )-`( , o o)     ██║     ██╔══██║   ██║   ██╔══██╗██║   ██║   ██║       ██║╚██╗██║██╔══╝  ██║   ██║
#           `-    \`_`"'-       ╚██████╗██║  ██║   ██║   ██████╔╝╚██████╔╝   ██║       ██║ ╚████║███████╗╚██████╔╝
#                                ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝       ╚═╝  ╚═══╝╚══════╝ ╚═════╝

# RoboCup Junior Rescue Line 2023 - Asia Pacific (South Korea)
# https://github.com/zmcwilliam/catbot-rcjap

import time
import os
import sys
import json
import traceback
import math
import signal
import board
import cv2
import busio
import gpiozero
import numpy as np
import adafruit_vl6180x
import helper_camera
import helper_camerakit as ck
import helper_motorkit as m
import helper_intersections
from helper_cmps14 import CMPS14

DEBUGGER = False # Should the debug switch actually work? This should be set to false if using the runner

# ------------
# DEVICE PORTS
# ------------
PORT_DEBUG_SWITCH = 21
PORT_SERVO_GATE = 12
PORT_SERVO_CLAW = 13
PORT_SERVO_LIFT = 18
PORT_SERVO_CAM = 19
PORT_USS_TRIG = {
    "front": 23,
    "side": 27,
}
PORT_USS_ECHO = {
    "front": 24,
    "side": 22,
}

# -------------
# CONFIGURATION
# -------------
max_error = 285                     # Maximum error value when calculating a percentage
max_angle = 90                      # Maximum angle value when calculating a percentage
error_weight = 0.5                  # Weight of the error value when calculating the PID input
angle_weight = 1 - error_weight       # Weight of the angle value when calculating the PID input
black_contour_threshold = 5000      # Minimum area of a contour to be considered valid

KP = 1.3                            # Proportional gain
KI = 0                              # Integral gain
KD = 0.08                           # Derivative gain

follower_speed = 45                 # Base speed of the line follower
obstacle_treshold = 9               # Minimum distance treshold for obstacles (cm)

evac_cam_angle = 7                  # Angle of the camera when evacuating

# ---------------------
# LOAD STORED JSON DATA
# ---------------------
with open("calibration.json", "r", encoding="utf-8") as json_file:
    calibration_data = json.load(json_file)
calibration_map = 255 / np.array(calibration_data["calibration_map_w"])
calibration_map_rescue = 255 / np.array(calibration_data["calibration_map_rescue_w"])

with open("config.json", "r", encoding="utf-8") as json_file:
    config_data = json.load(json_file)

config_values = {
    "black_line_threshold": config_data["black_line_threshold"],
    "black_rescue_threshold": config_data["black_rescue_threshold"],
    "rescue_circle_conf": config_data["rescue_circle_conf"],
    "green_turn_hsv_threshold": [np.array(bound) for bound in config_data["green_turn_hsv_threshold"]],
    "red_hsv_threshold": [np.array(bound) for bound in config_data["red_hsv_threshold"]],
    "obstacle_hsv_threshold": [np.array(bound) for bound in config_data["obstacle_hsv_threshold"]],
    "rescue_block_hsv_threshold": [np.array(bound) for bound in config_data["rescue_block_hsv_threshold"]],
    "rescue_circle_minradius_offset": config_data["rescue_circle_minradius_offset"],
    "rescue_binary_gray_scale_multiplier": config_data["rescue_binary_gray_scale_multiplier"]
}

# ----------------
# SYSTEM VARIABLES
# ----------------
program_active = True
has_moved_windows = False
program_sleep_time = 0.01

current_steering = 0
last_line_pos = np.array([100, 100])

turning = None
last_green_time = 0
initial_green_time = 0
changed_black_contour = False
current_linefollowing_state = None
intersection_state_debug = ["", time.time()]
red_stop_check = 0
evac_detect_check = 0

pid_last_error = 0
pid_integral = 0

frames = 0
current_time = time.time()
fpsTime = time.time()
fpsLoop = 0
fpsCamera = 0

rescue_mode = "init"
ball_found_qty = [0, 0] # Silver, Black

last_circle_pos = None
circle_check_counter = 0
bottom_block_approach_counter = 0

time_since_ramp_start = 0
time_ramp_end = 0

no_black_contours_mode = "straight"
no_black_contours = False

current_bearing = None
last_significant_bearing_change = 0

# Choose a random side for the obstacle in case the first direction is not possible
# A proper check should be added, but this is a quick fix for now
obstacle_dir = np.random.choice([-1, 1])

# ------------------
# INITIALISE DEVICES
# ------------------
i2c = busio.I2C(board.SCL, board.SDA)

cam = helper_camera.CameraStream(
    camera_num=0,
    processing_conf={
        "calibration_map": calibration_map,
        "black_line_threshold": config_values["black_line_threshold"],
        "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
        "red_hsv_threshold": config_values["red_hsv_threshold"],
    }
)

servo = {
    "gate": gpiozero.AngularServo(PORT_SERVO_GATE, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-90),    # -90=Close, 90=Open
    "claw": gpiozero.AngularServo(PORT_SERVO_CLAW, min_pulse_width=0.0005, max_pulse_width=0.002, initial_angle=-80),    # 0=Open, -90=Close
    "lift": gpiozero.AngularServo(PORT_SERVO_LIFT, min_pulse_width=0.0005, max_pulse_width=0.0025, initial_angle=-88),   # -90=Up, 40=Down
    "cam": gpiozero.AngularServo(PORT_SERVO_CAM, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-64)       # -90=Down, 90=Up
}

debug_switch = gpiozero.DigitalInputDevice(PORT_DEBUG_SWITCH, pull_up=True) if DEBUGGER else None

USS = {
    key: gpiozero.DistanceSensor(echo=USS_ECHO, trigger=USS_TRIG)
    for key, USS_ECHO, USS_TRIG in zip(PORT_USS_ECHO.keys(), PORT_USS_ECHO.values(), PORT_USS_TRIG.values())
}

cmps = CMPS14(1, 0x61)

vl6180x = adafruit_vl6180x.VL6180X(i2c)
vl6180x_gain = adafruit_vl6180x.ALS_GAIN_1 # See test_tof.py for more values

def exit_gracefully(signum=None, frame=None) -> None:
    """
    Handles program exit gracefully. Called by SIGINT signal.

    Args:
        signum (int, optional): Signal number. Defaults to None.
        frame (frame, optional): Current stack frame. Defaults to None.
    """
    global program_active
    if not program_active:
        # We already tried to exit, but may have gotten stuck. Force the exit.
        print("\nForcefully Exiting")
        sys.exit()

    print("\n\nExiting Gracefully\n")
    program_active = False
    m.stop_all()
    cam.stop()
    cv2.destroyAllWindows()

    for u in USS.values():
        u.close()

    for s in servo.values():
        s.detach()
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

    return DEBUGGER and debug_switch.value

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

    start_time = time.time()
    while time.time() - start_time < timeout:
        current_bearing = cmps.read_bearing_16bit()
        error = min(abs(current_bearing - target_bearing), abs(target_bearing - current_bearing + 360))

        if error < cutoff_error:
            print(f"{debug_prefix} FOUND Bearing: {current_bearing}\tTarget: {target_bearing}\tError: {error}")
            m.stop_all()
            return True

        max_speed = 50 # Speed to rotate when error is 180
        min_speed = 25 # Speed to rotate when error is 0

        rotate_speed = min_speed + ((max_speed - min_speed) / math.sqrt(180)) * math.sqrt(error)

        # Rotate in the direction closest to the bearing
        if (current_bearing - target_bearing) % 360 < 180:
            m.run_tank(-rotate_speed, rotate_speed)
        else:
            m.run_tank(rotate_speed, -rotate_speed)

        print(f"{debug_prefix}Bearing: {current_bearing}\tTarget: {target_bearing}\tError: {error}\tSpeed: {rotate_speed}")

# ------------------
# OBSTACLE AVOIDANCE
# ------------------
def avoid_obstacle() -> None:
    """
    Performs obstacle avoidance when an obstacle is detected.

    This uses a very basic metho of just rotating a fixed speed around the obstacle until a line is detected.
    It will not work for varying sizes of obstacles, and hence should be worked on if time permits, but it's fine for now.
    """
    global obstacle_dir

    # Check if it's actually an obstacle by using the camera
    frame_processed = cam.read_stream_processed()
    while (frame_processed is None or frame_processed["resized"] is None):
        print("Waiting for image...")
        time.sleep(0.1)

    img0_hsv = frame_processed["hsv"]

    img0_obstacle = cv2.inRange(img0_hsv, config_values["obstacle_hsv_threshold"][0], config_values["obstacle_hsv_threshold"][1])
    img0_obstacle = cv2.dilate(img0_obstacle, np.ones((5, 5), np.uint8), iterations=2)

    obstacle_contours = [[contour, cv2.contourArea(contour)] for contour in cv2.findContours(img0_obstacle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]]
    obstacle_contours = sorted(obstacle_contours, key=lambda contour: contour[1], reverse=True)
    obstacle_contours_filtered = [contour[0] for contour in obstacle_contours if contour[1] > 10000]

    if len(obstacle_contours_filtered) == 0:
        # EVACUATION ZONE PROBABLY
        print("DETECTED EVAC")
        m.run_tank_for_time(-40, -40, 500)

        time.sleep(2)
        # Random rotate dir
        evac_rotate_dir = np.random.choice([-1, 1])
        align_to_bearing(cmps.read_bearing_16bit() - (80 * evac_rotate_dir), 1, debug_prefix="EVAC ROTATE: ")
        time.sleep(1)
        print("STARTING EVAC")
        run_evac()

    # Avoid the obstacle

    obstacle_dir = -1 if obstacle_dir == 1 else 1
    m.run_tank_for_time(-40, -40, 900)
    align_to_bearing(cmps.read_bearing_16bit() - (70 * obstacle_dir), 1, debug_prefix="OBSTACLE ALIGN: ")
    time.sleep(0.2)
    if obstacle_dir > 0: m.run_tank(100, 30)
    else: m.run_tank(30, 100)
    time.sleep(1) # Threshold before accepting any possibility of a line

    # Start checking for a line while continuing to rotate around the obstacle
    while True:
        frame_processed = cam.read_stream_processed()

        img0_line_not = cv2.bitwise_not(frame_processed["line"])
        black_contours, _ = cv2.findContours(img0_line_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_contours_filtered = [c for c in black_contours if cv2.contourArea(c) > 5000]

        if len(black_contours_filtered) >= 1:
            print("[OBSTACLE] Found Line")
            break

    m.stop_all()
    time.sleep(0.2)

# ----------
# EVACUATION
# ----------
def approach_victim(time_to_approach):
    global last_circle_pos
    global ball_found_qty
    time.sleep(0.2)
    m.run_tank(40, 40)
    time.sleep(time_to_approach - 0.1)
    servo["claw"].angle = -90
    time.sleep(0.1)
    m.stop_all()
    time.sleep(0.2)
    servo["cam"].angle = -80 # Ensure cam is out of the way before we do lifting actions
    m.run_tank_for_time(-40, -40, 800)

    if check_found_ball():
        print("Successful capture, lifting")
        servo["lift"].angle = -80
        time.sleep(0.8)
        servo["claw"].angle = -45
    else:
        print("Did not capture a ball.")

    print("Total balls found: " + str(sum(ball_found_qty)))
    if sum(ball_found_qty) < 3:
        time.sleep(0.5)
        servo["lift"].angle = 40
        servo["claw"].angle = 0

    time.sleep(0.2)
    servo["cam"].angle = evac_cam_angle
    time.sleep(0.7)

    last_circle_pos = None

def check_found_ball():
    global ball_found_qty

    try:
        range_mm = vl6180x.range
        light_lux = vl6180x.read_lux(adafruit_vl6180x.ALS_GAIN_1)

        # TODO: Check ball type (However as we don't handle this differently, there's currently no point)
        print(f"Ball: {range_mm}mm, {light_lux}lux")
        if range_mm < 5:
            ball_found_qty[0] += 1
            return True
        return False
    except Exception:
        print("ERR BALL")
        ball_found_qty[0] += 1
        return True

def run_evac():
    global frames
    global fpsLoop
    global fpsTime
    global fpsCamera
    global rescue_mode
    global ball_found_qty
    global program_active
    global last_circle_pos
    global circle_check_counter
    global bottom_block_approach_counter

    evac_start = time.time()

    while True:
        if int(time.time() - evac_start) % 10 == 0:
            print("EVAC TIME: " + str(int(time.time() - evac_start)))

        if time.time() - evac_start > 120 and rescue_mode == "victim":
            print("VICTIMS TOOK TOO LONG - SKIPPING TO BLOCK")
            rescue_mode = "block"
        # time.sleep(program_sleep_time)
        frames += 1

        if frames % 20 == 0 and frames != 0:
            fpsLoop = int(frames / (time.time() - fpsTime))
            fpsCamera = cam.get_fps()

            if frames > 500:
                fpsTime = time.time()
                frames = 0
            print(f"Processing FPS: {fpsLoop} | Camera FPS: {cam.get_fps()} | Sleep time: {int(program_sleep_time*1000)}")

        frame_processed = cam.read_stream_processed()
        if (frame_processed is None or frame_processed["resized"] is None):
            print("Waiting for image...")
            time.sleep(0.1)
            fpsTime = time.time()
            frames = 0
            continue

        img0 = frame_processed["resized"]

        img0_gray = frame_processed["gray"]
        img0_hsv = frame_processed["hsv"]

        img0_gray_rescue_calibrated = calibration_map_rescue * img0_gray
        img0_binary_rescue = ((img0_gray_rescue_calibrated > config_values["black_rescue_threshold"]) * 255).astype(np.uint8)
        img0_binary_rescue = cv2.morphologyEx(img0_binary_rescue, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        img0_gray_rescue_scaled = img0_gray_rescue_calibrated * (config_values["rescue_binary_gray_scale_multiplier"] - 2.5)
        img0_gray_rescue_scaled = np.clip(img0_gray_rescue_calibrated, 0, 255).astype(np.uint8)

        img0_binary_rescue_clean = img0_binary_rescue.copy()

        img0_blurred = cv2.medianBlur(img0_gray_rescue_scaled, 9)

        segment_width = 40

        for x in range(0, img0_binary_rescue.shape[1], segment_width):
            segment = img0_binary_rescue[:, x:x + segment_width]

            # Find the lowest point in the segment that has a full column of white pixels at least 10 pixels high
            bottom_white_column = -1
            for y in range(0, img0_binary_rescue.shape[0] - 10):
                if np.all(segment[y:y + 10, :] == 255):
                    bottom_white_column = y
                    break

            # Set all pixels below this column to black
            if bottom_white_column != -1:
                img0_binary_rescue[:bottom_white_column:, x:x + segment_width] = 255
                img0_blurred[:bottom_white_column:, x:x + segment_width] = 255

        # -------------
        # INITIAL ENTER
        # -------------
        if rescue_mode == "init":
            # TODO: Optimally, this should try and use the ultrasonic sensor
            #       to get to a decent spot in the middle of the evac zone
            # For now, just drive forward for a bit
            m.run_tank_for_time(60, 60, 1500)

            # Spin around to try and get us off any wall
            align_to_bearing(cmps.read_bearing_16bit() - 180, 7, debug_prefix="EVAC Align - ")
            align_to_bearing(cmps.read_bearing_16bit() - 90, 7, debug_prefix="EVAC Align - ")

            front_dist = USS["front"].distance * 100

            if front_dist < 10:
                m.run_tank_for_time(-60, -60, 2500)
                align_to_bearing(cmps.read_bearing_16bit() - 180, 7, debug_prefix="EVAC Align - ")

            servo["cam"].angle = -80
            servo["lift"].angle = 40
            servo["claw"].angle = 0
            time.sleep(0.5)
            servo["cam"].angle = evac_cam_angle
            time.sleep(0.5)
            m.run_tank_for_time(60, 60, 1000)

            ball_found_qty = [0, 0]
            rescue_mode = "victim"

        # ------------
        # VICTIM STUFF
        # ------------
        elif rescue_mode == "victim":
            if sum(ball_found_qty) >= 3:
                print("Found all victims")
                rescue_mode = "block"

            servo["cam"].angle = evac_cam_angle
            servo["lift"].angle = 40
            servo["claw"].angle = 0

            # minDist = min distance between circles
            # param1 = high threshold for canny edge detection (sensitivity) - lower = more circles
            # param2 = accumulator threshold for circle detection - Higher = more reliable circles, but may miss some
            # minRadius = minimum radius of circle
            # maxRadius = maximum radius of circle
            circles = cv2.HoughCircles(img0_blurred, cv2.HOUGH_GRADIENT, **{
                "dp": config_values["rescue_circle_conf"]["dp"],
                "minDist": config_values["rescue_circle_conf"]["minDist"],
                "param1": config_values["rescue_circle_conf"]["param1"],
                "param2": config_values["rescue_circle_conf"]["param2"],
                "minRadius": config_values["rescue_circle_conf"]["minRadius"],
                "maxRadius": config_values["rescue_circle_conf"]["maxRadius"],
            })

            sorted_circles = []
            if circles is not None:
                # Round the circle parameters to integers
                detected_circles = np.round(circles[0, :]).astype(int)
                detected_circles = sorted(detected_circles, key = lambda v: [v[1], v[1]], reverse=True)

                # If the circle is in the bottom half of the image, check that the radius is greater than 40
                # valid_circles = []
                # for (x, y, r) in detected_circles:
                #     if y > config_values["rescue_circle_conf"]["heightBuffer"] and r > config_values["rescue_circle_conf"]["lowHeightMinRadius"]:
                #         valid_circles.append([x, y, r])
                #     elif y <= config_values["rescue_circle_conf"]["heightBuffer"]:
                #         valid_circles.append([x, y, r])

                # Draw the detected circles on the original image
                for (x, y, r) in detected_circles:
                    cv2.circle(img0, (x, y), r, (0, 0, 255), 2)
                    cv2.circle(img0, (x, y), 2, (0, 0, 255), 3)

                height_bar_qty = 13
                height_bar_minRadius = [a - 7 for a in [90, 85, 80, 72, 66, 58, 52, 44, 38, 31, 23, 16, 15]] # This could become a linear function... but it works for now
                height_bar_maxRadius = [b + config_values["rescue_circle_minradius_offset"] for b in height_bar_minRadius]

                height_bars = [(img0.shape[0] / height_bar_qty) * i for i in range(height_bar_qty - 1, -1, -1)]
                height_bar_circles = [[] for i in range(height_bar_qty)]
                for (x, y, r) in detected_circles:
                    for i in range(height_bar_qty):
                        if y >= height_bars[i]:
                            if height_bar_minRadius[i] <= r <= height_bar_maxRadius[i]:
                                height_bar_circles[i].append([x, y, r, i])
                            break

                    # Sort the circles in each bar by x position
                    height_bar_circles[i] = sorted(height_bar_circles[i], key = lambda v: [v[0], v[0]])

                # Compile circles into a single list, horizontally in height bar (closest to furthest)
                for i in range(height_bar_qty):
                    sorted_circles += height_bar_circles[i]

                # Draw horizontal lines for height bars
                for i in range(height_bar_qty):
                    cv2.line(img0, (0, int(height_bars[i])), (img0.shape[1], int(height_bars[i])), (255, 255, 255), 1)

                # Draw the sorted circles on the original image
                for i, (x, y, r, height_bar) in enumerate(sorted_circles):
                    cv2.circle(img0, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(img0, (x, y), 2, (0, 255, 0), 3)
                    cv2.putText(img0, f"{height_bar}-{i}-{r}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 125, 255), 2)

            if debug_state("rescue"):
                # cv2.imshow("img0_blurred", img0_blurred)
                # cv2.imshow("img0_gray_rescue_scaled", img0_gray_rescue_scaled)
                cv2.imshow("img0_binary_rescue", cv2.resize(img0_binary_rescue, (0, 0), fx=0.8, fy=0.7))
                cv2.imshow("img0_binary_rescue_clean", cv2.resize(img0_binary_rescue_clean, (0, 0), fx=0.8, fy=0.7))
                cv2.imshow("img0", cv2.resize(img0, (0, 0), fx=0.8, fy=0.7))

                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break

            if len(sorted_circles) > 0:
                lowest_circle = sorted_circles[0]
                circle_check_counter = 0

                # If the circle is +-100 pixels horizontally away from the centre, steer the robot towards it
                if lowest_circle[0] < (img0.shape[1] / 2) - 100:
                    print("Circle is to the left")
                    m.run_tank_for_time(-40, 40, 50, False)
                elif lowest_circle[0] > (img0.shape[1] / 2) + 100:
                    print("Circle is to the right")
                    m.run_tank_for_time(40, -40, 50, False)
                else:
                    last_circle_pos = lowest_circle
                    print("Circle is in the centre-ish")

                    THRESH_FINAL_APPROACH = 360
                    if lowest_circle[1] > THRESH_FINAL_APPROACH:
                        print("Approaching")
                        approach_victim(1)
                    m.run_tank_for_time(40, 40, 300)
            else:
                if last_circle_pos is not None and last_circle_pos[1] > 340:
                    print("Approaching with extra distance")
                    approach_victim(1.5)
                    continue

                if last_circle_pos is not None and last_circle_pos[1] <= 340 and abs(last_circle_pos[0] - (img0.shape[1] / 2)) < 250:
                    # Did the circle vanish? Lets double check...
                    if circle_check_counter < 3:
                        print("Circle may have vanished, double checking")
                        circle_check_counter += 1
                        time.sleep(0.3)
                        continue
                    print("Circle vanished")
                last_circle_pos = None
                print("Rotating")
                m.run_tank_for_time(60, -60, 200)
                time.sleep(0.2)

        # ------------------------
        # BLOCK FINDING AND RESCUE
        # ------------------------
        elif rescue_mode == "block":
            # Find the contours of the rescue blocks
            img0_block_mask = cv2.inRange(img0_hsv, config_values["rescue_block_hsv_threshold"][0], config_values["rescue_block_hsv_threshold"][1])
            img0_binary_rescue_block = cv2.bitwise_and(cv2.bitwise_not(img0_binary_rescue), img0_block_mask)
            img0_binary_rescue_block = cv2.morphologyEx(img0_binary_rescue_block, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))

            contours_block = cv2.findContours(img0_binary_rescue_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            contours_block = [{
                "contour": c,
                "contourArea": cv2.contourArea(c),
                "boundingRect": cv2.boundingRect(c),
                "touching_sides": [],
                "near_sides": []
            } for c in contours_block]

            # Figure out which sides of the image the contours are touching
            side_threshold = [10, 50]
            for i, contour in enumerate(contours_block):
                x, y, w, h = contour["boundingRect"]

                if x < side_threshold[0]: contours_block[i]["touching_sides"].append("left")
                if y < side_threshold[0]: contours_block[i]["touching_sides"].append("top")
                if x + w > img0.shape[1] - side_threshold[0]: contours_block[i]["touching_sides"].append("right")
                if y + h > img0.shape[0] - side_threshold[0]: contours_block[i]["touching_sides"].append("bottom")

                if x < side_threshold[1]: contours_block[i]["near_sides"].append("left")
                if y < side_threshold[1]: contours_block[i]["near_sides"].append("top")
                if x + w > img0.shape[1] - side_threshold[1]: contours_block[i]["near_sides"].append("right")
                if y + h > img0.shape[0] - side_threshold[1]: contours_block[i]["near_sides"].append("bottom")

            # Filter by area and ensure it doesn't touch the top
            contours_block = [c for c in contours_block if c["contourArea"] > 10000 and "top" not in c["touching_sides"]]

            # Sort by area
            contours_block = sorted(contours_block, key=lambda c: c["contourArea"], reverse=True)

            cv2.drawContours(img0, [c["contour"] for c in contours_block], -1, (0, 0, 255), 2)
            if len(contours_block) > 0:
                contour_block = contours_block[0]

                cv2.rectangle(img0, contour_block["boundingRect"], (0, 255, 0), 2)

                cx = contour_block["boundingRect"][0] + contour_block["boundingRect"][2] / 2
                cy = contour_block["boundingRect"][1] + contour_block["boundingRect"][3] / 2

                cv2.circle(img0, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                front_dist = USS["front"].distance * 100
                if front_dist <= 5 and front_dist != 0:
                    bottom_block_approach_counter += 5

                if "bottom" in contour_block["touching_sides"]:
                    m.run_tank(35, 35)
                    bottom_block_approach_counter += 1

                    if bottom_block_approach_counter > 30:
                        print("Finished approach")
                        m.run_tank_for_time(-40, -40, 1400)
                        time.sleep(0.1)
                        start_bearing = cmps.read_bearing_16bit()
                        align_to_bearing(start_bearing - 180, 10, debug_prefix="EVAC Align - ")
                        time.sleep(0.1)
                        m.run_tank_for_time(-35, -35, 1000)
                        servo["gate"].angle = 70
                        time.sleep(0.5)
                        for i in range(12):
                            m.run_tank_for_time(100, 100, 150)
                            m.run_tank_for_time(-100, -100, 250)
                        servo["gate"].angle = -90
                        m.run_tank_for_time(35, 35, 1000)
                        time.sleep(1)

                    m.run_tank_for_time(35, 35, 200)
                    print("Block on bottom - approach")
                else:
                    if front_dist <= 5 and front_dist != 0:
                        print("Block is nearby")
                    else:
                        bottom_block_approach_counter = 0
                    if (
                        ("left" in contour["near_sides"] and "right" in contour["near_sides"])
                        or (
                            abs(cx - (img0.shape[1] / 2)) < 50
                            and (("left" in contour_block["touching_sides"]) + ("right" in contour_block["touching_sides"])) != 1
                        )
                    ):
                        m.run_tank(35, 35)
                        print("Block in middle - approach")
                    elif cx < (img0.shape[1] / 2) or "left" in contour_block["touching_sides"]:
                        m.run_tank(10, 35)
                        print("Block on left - turn left")
                    else:
                        m.run_tank(35, 10)
                        print("Block on right - turn right")
            else:
                m.run_tank_for_time(60, -60, 100)

            servo["claw"].angle = -90

            if servo["lift"].angle > -60:
                # If the lift was down, give a bit of time before the camera moves
                servo["cam"].angle = -80
                servo["lift"].angle = -80
                time.sleep(0.5)
                servo["cam"].angle = evac_cam_angle
            else:
                servo["lift"].angle = -80
                servo["cam"].angle = evac_cam_angle

            if debug_state("rescue"):
                # cv2.imshow("img0_blurred", img0_blurred)
                # cv2.imshow("img0_gray_rescue_scaled", img0_gray_rescue_scaled)
                cv2.imshow("img0_binary_rescue_block", cv2.resize(img0_binary_rescue_block, (0, 0), fx=0.8, fy=0.7))
                cv2.imshow("img0_binary_rescue", cv2.resize(img0_binary_rescue, (0, 0), fx=0.8, fy=0.7))
                cv2.imshow("img0_binary_rescue_clean", cv2.resize(img0_binary_rescue_clean, (0, 0), fx=0.8, fy=0.7))
                cv2.imshow("img0", cv2.resize(img0, (0, 0), fx=0.8, fy=0.7))

                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break
# ------------------------
# WAIT FOR VISION TO START
# ------------------------
m.stop_all()
os.system("cat motd.txt")
for i in range(3, 0, -1):
    print(f"Starting in {i}...", end="\r")
    time.sleep(1)

# Clear the countdown line
print("\033[K")
print()

# Ensure frames have been processed at least once
if cam.read_stream_processed()["raw"] is None:
    print("Waiting for first frame")
    while cam.read_stream_processed()["raw"] is None:
        time.sleep(0.1)

# ---------
# MAIN LOOP
# ---------
while program_active:
    try:
        # ---------------
        # FRAME BALANCING
        # ---------------
        time.sleep(program_sleep_time)
        frames += 1

        if frames % 30 == 0 and frames != 0:
            fpsLoop = int(frames / (time.time() - fpsTime))
            fpsCamera = cam.get_fps()

            # Try to balance out the processing time and the camera FPS
            sleep_adjustment_amt = 0.001
            sleep_time_max = 0.02
            sleep_time_min = 0.005
            if fpsLoop > fpsCamera + 5 and program_sleep_time < sleep_time_max:
                program_sleep_time += sleep_adjustment_amt
            elif fpsLoop < fpsCamera and program_sleep_time > sleep_time_min:
                program_sleep_time -= sleep_adjustment_amt

            if frames > 500:
                fpsTime = time.time()
                frames = 0
            print(f"FPS: {fpsLoop}, {fpsCamera} \tDel: {int(program_sleep_time*1000)}")

        # ------------------
        # OBSTACLE AVOIDANCE
        # ------------------
        front_dist = USS["front"].distance * 100
        if front_dist < obstacle_treshold:
            print(f"Obstacle detected at {front_dist}cm... ", end="")
            m.stop_all()
            time.sleep(0.7)
            if USS["front"].distance * 100 < obstacle_treshold + 1:
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
        img0_clean = img0.copy() # Used for displaying the image without any overlays

        # img0_gray = frame_processed["gray"].copy()
        # img0_gray_scaled = frame_processed["gray_scaled"].copy()
        img0_binary = frame_processed["binary"].copy()
        img0_hsv = frame_processed["hsv"].copy()
        img0_green = frame_processed["green"].copy()
        img0_line = frame_processed["line"].copy()

        # -----------

        raw_white_contours, white_hierarchy = cv2.findContours(img0_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter white contours based on area
        white_contours = []
        for contour in raw_white_contours:
            if cv2.contourArea(contour) > 1000:
                white_contours.append(contour)

        if len(white_contours) == 0:
            print("No white contours found")
            continue

        # Find black contours
        # If there are no black contours, skip the rest of the loop
        img0_line_not = cv2.bitwise_not(img0_line)
        black_contours, black_hierarchy = cv2.findContours(img0_line_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(black_contours) == 0:
            print("No black contours found")
            continue

        # -----------
        # GREEN TURNS
        # -----------

        is_there_green = np.count_nonzero(img0_green == 0)
        black_contours_turn = None

        # print("Green: ", is_there_green)

        img0_line_new = img0_line.copy()

        # Check if there is a significant amount of green pixels
        if is_there_green > 4000: # and len(white_contours) > 2: #((is_there_green > 1000 or time.time() - last_green_found_time < 0.5) and (len(white_contours) > 2 or greenCenter is not None)):
            changed_img0_line = None

            unfiltered_green_contours, green_hierarchy = cv2.findContours(cv2.bitwise_not(img0_green), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                turning = "RIGHT" # Arbitrarily make it right for now... It is very possible this won't work
                continue
            else:
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
                new_black_contours, new_black_hierarchy = cv2.findContours(img0_line_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img0, new_black_contours, -1, (0, 0, 255), 2)
                if len(new_black_contours) > 0:
                    black_contours = new_black_contours
                    black_hierarchy = new_black_hierarchy
                else:
                    print("No black contours found after changing contour")

                changed_black_contour = False

            print("GREEN TURN STUFF")
        elif turning is not None and last_green_time + 1 < time.time():
            turning = None
            print("No longer turning")

        # -----------------
        # STOP ON RED CHECK
        # -----------------
        img0_red = cv2.inRange(img0_hsv, config_values["red_hsv_threshold"][0], config_values["red_hsv_threshold"][1])
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
                    cv2.imshow("img0_red", img0_red)
                    cv2.waitKey(1)

                if red_stop_check > 3:
                    print("DETECTED RED STOP 3 TIMES, STOPPING")
                    break

                continue # Don't run the rest of the follower, we don't really want to move forward in case we accidentally loose the red...
            else:
                red_stop_check = 0

        # -------------
        # INTERSECTIONS
        # -------------
        if not turning:
            # Filter white contours to have a minimum area before we accept them
            white_contours_filtered = [contour for contour in white_contours if cv2.contourArea(contour) > 500]

            if len(white_contours_filtered) == 2:
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
                    if sum([d < 80 for d in top_dists]) == 4:
                        # If none of the following are true, then make the top of the image white, anywhere above the lowest point
                        #   - All points at the top
                        #   - Left top points are at the top, right top points are close to the top (disabled for now, as it breaks entry to 3WC)
                        #   - Right top points are at the top, left top points are close to the top (disabled for now, as it breaks entry to 3WC)
                        if (
                            not sum([d < 3 for d in top_dists]) == 4
                            # and not (sum([d < 3 for d in top_dists[:2]]) == 2 and sum([12 < d and d < 80 for d in top_dists[2:]]) == 2)
                            # and not (sum([d < 3 for d in top_dists[2:]]) == 2 and sum([12 < d and d < 80 for d in top_dists[:2]]) == 2)
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
                    if sum([d < 3 for d in bottom_dists]) == 4:
                        current_linefollowing_state = None
                        print("Exited intersection")
                    # If all points are still near the bottom, since we already know we are existing an intersection, remove the bottom to prevent the robot from turning
                    elif sum([d < 120 for d in bottom_dists]) == 4:
                        current_linefollowing_state = "2-bottom-ng"
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

                intersection_state_debug = ["3-ng", time.time()]
                # Get the center of each contour
                white_contours_filtered_with_center = [(contour, ck.centerOfContour(contour)) for contour in white_contours_filtered]

                # Sort the contours from left to right - Based on the centre of the contour's horz val
                sorted_contours_horz = sorted(white_contours_filtered_with_center, key=lambda contour: contour[1][0])

                # Simplify the contours to get the corner points
                approx_contours = [ck.simplifiedContourPoints(contour[0], 0.03) for contour in sorted_contours_horz]

                # Middle of contour centres
                mid_point = (
                    int(sum([contour[1][0] for contour in sorted_contours_horz]) / len(sorted_contours_horz)),
                    int(sum([contour[1][1] for contour in sorted_contours_horz]) / len(sorted_contours_horz))
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

                # --------------
                # EVAC DETECTION
                # --------------
                # This is a janky solution to detecting evac entry... it should work for now, but definitely should be looked at.
                if len(black_contours) >= 1 and edges_big == ["left", "right", "top"]:
                    edges_black = sorted(ck.getTouchingEdges(ck.simplifiedContourPoints(black_contours[0], 0.03), img0_binary.shape))

                    if edges_black == ["bottom", "left", "right"]:
                        m.stop_all()
                        evac_detect_check += 1
                        print(f"EVACUATION ZONE DETECTED: {evac_detect_check}/3")

                        if evac_detect_check == 1:
                            time.sleep(0.1)
                            m.run_tank_for_time(-40, -40, 100)
                            time.sleep(0.1)

                        if evac_detect_check >= 3:
                            print("STARTING EVAC")
                            run_evac()

                        time.sleep(0.1)
                        continue

                evac_detect_check = 0

                # --- Rest of 3WC Intersections

                # Cut direction is based on the side of the line with the most contour center points (contour_center_point_sides)
                cut_direction = len(contour_center_point_sides[0]) > len(contour_center_point_sides[1])

                # If we are just entering a 3-way intersection, and the 'big contour' does not connect to the bottom,
                # we may be entering a 4-way intersection... so follow the vertical line
                if len(edges_big) >= 2 and "bottom" not in edges_big and "-en" in current_linefollowing_state:
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
                if closest_2_points_vert_sort[0][0][1] == closest_2_points_vert_sort[1][0][1]:
                    closest_2_points_vert_sort[0][0][1] += 1 # Move the first point up by 1 pixel

                img0_line_new = helper_intersections.CutMaskWithLine(closest_2_points_vert_sort[0][0], closest_2_points_vert_sort[1][0], img0_line_new, "left" if cut_direction else "right")
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

                img0_line_new = helper_intersections.CutMaskWithLine(closest_BL, closest_TL, img0_line_new, "left")
                img0_line_new = helper_intersections.CutMaskWithLine(closest_BR, closest_TR, img0_line_new, "right")

                current_linefollowing_state = "4-ng"
                changed_black_contour = cv2.bitwise_not(img0_line_new)

        if changed_black_contour is not False:
            print("Changed black contour, LF State: ", current_linefollowing_state)
            cv2.drawContours(img0, black_contours, -1, (0, 0, 255), 2)
            new_black_contours, new_black_hierarchy = cv2.findContours(changed_black_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(new_black_contours) > 0:
                black_contours = new_black_contours
                black_hierarchy = new_black_hierarchy
            else:
                print("No black contours found after changing contour")

            changed_black_contour = False

        # --------------------------
        # REST OF LINE LINE FOLLOWER
        # --------------------------

        #Find the black contours
        sorted_black_contours = ck.findBestContours(black_contours, black_contour_threshold, last_line_pos)
        if len(sorted_black_contours) == 0:
            print("No black contours found")

            # This is a botchy temp fix so that sometimes we can handle the case where we lose the line,
            # and other times we can handle gaps in the line
            # TODO: Remove this, and implement a proper line following fix
            if not no_black_contours:
                no_black_contours_mode = "straight" if no_black_contours_mode == "steer" else "steer"
                no_black_contours = True

            # We've lost any black contour, so it's possible we have encountered a gap in the line
            # Hence, go straight.
            #
            # Optimally, this should figure out if the line lost was in the centre and hence we haven't just fallen off the line.
            # Going forward, instead of using current_steering, means if we fall off the line, we have little hope of getting back on...
            new_steer = current_steering if no_black_contours_mode == "steer" else 0
            m.run_steer(follower_speed, 100, new_steer)

            preview_image_img0 = cv2.resize(img0, (0, 0), fx=0.8, fy=0.7)

            if debug_state():
                cv2.imshow("img0 - NBC", preview_image_img0)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    program_active = False
                    break
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
        black_bounding_box_TR = sorted(vert_sorted_black_bounding_points[2:], key=lambda point: point[0])[1]

        # Get the angle of the line between the bottom left and top right points
        black_contour_angle = int(math.degrees(math.atan2(black_bounding_box_TR[1] - black_bounding_box_BL[1], black_bounding_box_TR[0] - black_bounding_box_BL[0])))
        black_contour_angle_new = black_contour_angle + 80

        # The two top-most points, sorted from left to right
        horz_sorted_black_bounding_points_top_2 = sorted(vert_sorted_black_bounding_points[2:], key=lambda point: point[0])

        # If the angle of the contour is big enough and the contour is close to the edge of the image (within bigTurnSideMargin pixels)
        # Then, the line likely is a big turn and we will need to turn more
        # 0 if None, 1 if left, 2 if right
        isBigTurn = 0

        bigTurnAngleMargin = 30
        bigTurnSideMargin = 30
        if abs(black_contour_angle_new) > bigTurnAngleMargin:
            if horz_sorted_black_bounding_points_top_2[0][0] < bigTurnSideMargin:
                isBigTurn = 1
            elif horz_sorted_black_bounding_points_top_2[1][0] > img0.shape[1] - bigTurnSideMargin:
                isBigTurn = 2

        if isBigTurn == 1 and black_contour_angle_new > 0 or isBigTurn == 2 and black_contour_angle_new < 0:
            black_contour_angle_new = black_contour_angle_new * -1

        current_position = (black_contour_angle_new / max_angle) * angle_weight + (black_contour_error / max_error) * error_weight
        current_position *= 100

        # The closer the topmost point is to the bottom of the screen, the more we want to turn
        topmost_point = sorted(black_bounding_box, key=lambda point: point[1])[0]
        extra_pos = (topmost_point[1] / img0.shape[1]) * 10
        if isBigTurn and extra_pos > 1:
            current_position *= min(0.7 * extra_pos, 1)

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

        current_pitch = cmps.read_pitch()

        if current_pitch > 180 and current_pitch < 240:
            if time_since_ramp_start == 0:
                time_since_ramp_start = time.time()
            print(f"RAMP ({int(time.time() - time_since_ramp_start)})")
            if time.time() - time_since_ramp_start > 18:
                motor_vals = m.run_steer(100, 100, 0)
            if time.time() - time_since_ramp_start > 10:
                motor_vals = m.run_steer(80, 100, current_steering, ramp=True)
            else:
                motor_vals = m.run_steer(follower_speed, 100, current_steering, ramp=True)
        else:
            if time_since_ramp_start > 3:
                time_ramp_end = time.time() + 2

            time_since_ramp_start = 0
            if time.time() < time_ramp_end:
                print("END RAMP")
                last_significant_bearing_change = time.time()
                motor_vals = m.run_steer(follower_speed, 100, current_steering, ramp=True)
            else:
                new_bearing = cmps.read_bearing_16bit()

                if current_bearing is None:
                    last_significant_bearing_change = time.time()
                    current_bearing = new_bearing

                bearing_diff = abs(new_bearing - current_bearing)
                if frames % 7 == 0: current_bearing = new_bearing

                # Check if the absolute difference is within the specified range or if it wraps around 360
                bearing_min_err = 6
                if (bearing_diff <= bearing_min_err or bearing_diff >= (360 - bearing_min_err)) and int(time.time() - last_significant_bearing_change) > 10:
                    print("SAME BEARING FOR 10 SECONDS")
                    m.run_tank_for_time(100, 100, 400)
                    last_significant_bearing_change = time.time()
                elif not (bearing_diff <= bearing_min_err or bearing_diff >= (360 - bearing_min_err)):
                    last_significant_bearing_change = time.time()

                motor_vals = m.run_steer(follower_speed, 100, current_steering)

        # ----------
        # DEBUG INFO
        # ----------

        print(f"FPS: {fpsLoop}, {fpsCamera} \
                \tDel: {int(program_sleep_time*1000)}  \
                \tSteer: {int(current_steering)} \
                \t{str(motor_vals)} \
                \tUSS: {round(front_dist, 1)} \
                \tPit: {int(current_pitch)} \
                \tBear: {current_bearing} LSB: {int(time.time() - last_significant_bearing_change)}")
        if debug_state():
            # cv2.drawContours(img0, [chosen_black_contour[2]], -1, (0,255,0), 3) # DEBUG
            # cv2.drawContours(img0, [black_bounding_box], 0, (255, 0, 255), 2)
            # cv2.line(img0, black_leftmost_line_points[0], black_leftmost_line_points[1], (255, 20, 51, 0.5), 3)

            preview_image_img0 = cv2.resize(img0, (0, 0), fx=0.8, fy=0.7)
            cv2.imshow("img0", preview_image_img0)

            # preview_image_img0_clean = cv2.resize(img0_clean, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_clean", preview_image_img0_clean)

            # preview_image_img0_binary = cv2.resize(img0_binary, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_binary", preview_image_img0_binary)

            # preview_image_img0_line = cv2.resize(img0_line, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_line", preview_image_img0_line)

            # preview_image_img0_green = cv2.resize(img0_green, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_green", preview_image_img0_green)

            # preview_image_img0_gray = cv2.resize(img0_gray, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_gray", preview_image_img0_gray)

            # def mouseCallbackHSV(event, x, y, flags, param):
            #     if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            #         # Print HSV value only when the left mouse button is pressed and mouse is moving
            #         hsv_value = img0_hsv[y, x]
            #         print(f"HSV: {hsv_value}")
            # # Show HSV preview with text on hover to show HSV values
            # preview_image_img0_hsv = cv2.resize(img0_hsv, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_hsv", preview_image_img0_hsv)
            # cv2.setMouseCallback("img0_hsv", mouseCallbackHSV)

            # preview_image_img0_gray_scaled = cv2.resize(img0_gray_scaled, (0,0), fx=0.8, fy=0.7)
            # cv2.imshow("img0_gray_scaled", preview_image_img0_gray_scaled)

            # Show a preview of the image with the contours drawn on it, black as red and white as blue

            # if frames % 5 == 0:
            preview_image_img0_contours = img0_clean.copy()
            cv2.drawContours(preview_image_img0_contours, white_contours, -1, (255, 0, 0), 3)
            cv2.drawContours(preview_image_img0_contours, black_contours, -1, (0, 255, 0), 3)
            cv2.drawContours(preview_image_img0_contours, [chosen_black_contour[2]], -1, (0, 0, 255), 3)

            cv2.putText(preview_image_img0_contours, f"{black_contour_angle:4d} Angle Raw", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{black_contour_angle_new:4d} Angle", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{black_contour_error:4d} Error", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{int(current_position):4d} Position", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{int(current_steering):4d} Steering", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # DEBUG
            cv2.putText(preview_image_img0_contours, f"{int(extra_pos):4d} Extra", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # DEBUG

            if turning is not None:
                cv2.putText(preview_image_img0_contours, f"{turning} Turning", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 0, 255), 2) # DEBUG

            if isBigTurn:
                cv2.putText(preview_image_img0_contours, "Big Turn", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(preview_image_img0_contours, f"LF State: {current_linefollowing_state}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(preview_image_img0_contours, f"INT Debug: {intersection_state_debug[0]} - {int(time.time() - intersection_state_debug[1])}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(preview_image_img0_contours, f"FPS: {fpsLoop} | {fpsCamera}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            preview_image_img0_contours = cv2.resize(preview_image_img0_contours, (0, 0), fx=0.8, fy=0.7)
            cv2.imshow("img0_contours", preview_image_img0_contours)

            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                program_active = False
                break

            if not has_moved_windows:
                cv2.moveWindow("img0", 100, 100)
                # cv2.moveWindow("img0_binary", 100, 800)
                # cv2.moveWindow("img0_line", 100, 600)
                # cv2.moveWindow("img0_green", 600, 600)
                # cv2.moveWindow("img0_gray", 0, 0)
                # cv2.moveWindow("img0_hsv", 0, 0)
                # cv2.moveWindow("img0_gray_scaled", 0, 0)
                cv2.moveWindow("img0_contours", 700, 100)
                has_moved_windows = True
    except Exception:
        print("UNHANDLED EXCEPTION: ")
        traceback.print_exc()
        print("Returning to start of loop in 5 seconds...")
        m.stop_all()
        time.sleep(5)

exit_gracefully()
