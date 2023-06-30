import time
import cv2
import json
import math
import helper_camera
import helper_motorkit as m
import helper_intersections
import numpy as np
import threading
from gpiozero import AngularServo
from typing import List, Tuple
from helper_cmps14 import CMPS14

# Type aliases
Contour = List[List[Tuple[int, int]]]

PORT_SERVO_GATE = 12
PORT_SERVO_CLAW = 13
PORT_SERVO_LIFT = 18
PORT_SERVO_CAM = 19

servo = {
    "gate": AngularServo(PORT_SERVO_GATE, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-90),    # -90=Close, 90=Open
    "claw": AngularServo(PORT_SERVO_CLAW, min_pulse_width=0.0005, max_pulse_width=0.002, initial_angle=-80),    # 0=Open, -90=Close
    "lift": AngularServo(PORT_SERVO_LIFT, min_pulse_width=0.0005, max_pulse_width=0.0025, initial_angle=-80),   # -90=Up, 40=Down
    "cam": AngularServo(PORT_SERVO_CAM, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-83)       # -90=Down, 90=Up
}

#Configs

# Load the calibration map from the JSON file
with open("calibration.json", "r") as json_file:
    calibration_data = json.load(json_file)
calibration_map = 255 / np.array(calibration_data["calibration_map_w"])
calibration_map_rescue = 255 / np.array(calibration_data["calibration_map_rescue_w"])

with open("config.json", "r") as json_file:
    config_data = json.load(json_file)

black_contour_threshold = 5000
config_values = {
    "black_line_threshold": config_data["black_line_threshold"],
    "black_rescue_threshold": config_data["black_rescue_threshold"],
    "green_turn_hsv_threshold": [np.array(bound) for bound in config_data["green_turn_hsv_threshold"]],
    "rescue_block_hsv_threshold": [np.array(bound) for bound in config_data["rescue_block_hsv_threshold"]],
    "rescue_circle_conf": config_data["rescue_circle_conf"],
}

cam = helper_camera.CameraStream(0, {
    "calibration_map": calibration_map,
    "black_line_threshold": config_values["black_line_threshold"],
    "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
})
cam.start_stream()

cmps = CMPS14(1, 0x61)

frames = 0
start_time = time.time()
fpsTime = time.time()
fpsLoop = 0
fpsCamera = 0

hasMovedWindows = False

last_circle_pos = None
circle_check_counter = 0
bottom_block_approach_counter = 0

EVAC_CAM_ANGLE = 4

rescue_mode = "init"
ball_found_qty = [0, 0] # Silver, Black

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
    servo["cam"].angle = EVAC_CAM_ANGLE
    time.sleep(0.7)

    last_circle_pos = None

def check_found_ball():
    global ball_found_qty

    # TODO: Add an actual check if we have a ball or not, and what type it is. (Using VL6180X)
    ball_found_qty[0] += 1
    return True

def align_to_bearing(target_bearing: int, cutoff_error: int, timeout: int = 1000, debug_prefix: str = "") -> bool:
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

        rotate_speed = min_speed + ((max_speed - min_speed)/math.sqrt(180)) * math.sqrt(error)

        # Rotate in the direction closest to the bearing
        if (current_bearing - target_bearing) % 360 < 180:
            m.run_tank(-rotate_speed, rotate_speed)
        else:
            m.run_tank(rotate_speed, -rotate_speed)

        print(f"{debug_prefix}Bearing: {current_bearing}\tTarget: {target_bearing}\tError: {error}\tSpeed: {rotate_speed}")

# MAIN LOOP
program_sleep_time = 0.01 # Initial sleep time

servo["cam"].angle = -80
servo["lift"].angle = 40
servo["claw"].angle = 0
time.sleep(0.5)
servo["cam"].angle = EVAC_CAM_ANGLE
while True:
    # time.sleep(program_sleep_time)
    frames += 1

    if frames % 20 == 0 and frames != 0:
        fpsLoop = int(frames/(time.time()-fpsTime))
        fpsCamera = cam.get_fps()

        if frames > 500:
            fpsTime = time.time()
            frames = 0
        print(f"Processing FPS: {fpsLoop} | Camera FPS: {cam.get_fps()} | Sleep time: {int(program_sleep_time*1000)}")

    changed_black_contour = False
    frame_processed = cam.read_stream_processed()
    if (frame_processed is None or frame_processed["resized"] is None):
        print("Waiting for image...")
        time.sleep(0.1)
        fpsTime = time.time()
        frames = 0
        continue

    if cam.is_halted():
        print("Camera is halted... Waiting")
        m.stop_all()
        time.sleep(0.1)
        continue

    img0 = frame_processed["resized"]
    img0_clean = img0.copy() # Used for displaying the image without any overlays

    img0_gray = frame_processed["gray"]
    img0_binary = frame_processed["binary"]
    img0_hsv = frame_processed["hsv"]
    img0_green = frame_processed["green"]
    img0_line = frame_processed["line"]
    
    img0_binary_rescue = ((calibration_map_rescue * img0_gray > config_values["black_rescue_threshold"]) * 255).astype(np.uint8)
    img0_binary_rescue = cv2.morphologyEx(img0_binary_rescue, cv2.MORPH_OPEN, np.ones((13,13),np.uint8))

    img0_binary_rescue_clean = img0_binary_rescue.copy()

    img0_gray_rescue_scaled = img0_gray * config_values["rescue_circle_conf"]["grayScaleMultiplier"]
    img0_gray_rescue_scaled = np.clip(img0_gray_rescue_scaled, 0, 255).astype(np.uint8)

    img0_blurred = cv2.medianBlur(img0_gray_rescue_scaled, 9)

    segment_width = 40

    for x in range(0, img0_binary_rescue.shape[1], segment_width):
        segment = img0_binary_rescue[:, x:x+segment_width]

        # Find the lowest point in the segment that has a full column of white pixels at least 10 pixels high
        bottom_white_column = -1
        for y in range(0, img0_binary_rescue.shape[0] - 10):
            if np.all(segment[y:y+10, :] == 255):
                bottom_white_column = y
                break

        # Set all pixels below this column to black
        if bottom_white_column != -1:
            img0_binary_rescue[:bottom_white_column:, x:x+segment_width] = 255
            img0_blurred[:bottom_white_column:, x:x+segment_width] = 255

    # -------------
    # INITIAL ENTER
    # -------------
    if rescue_mode == "init":
        # TODO: Optimally, this should try and use the ultrasonic sensor 
        #       to get to a decent spot in the middle of the evac zone
        # For now, just drive forward for a bit
        m.run_tank_for_time(60, 60, 2000)

        ball_found_qty = [0, 0]
        rescue_mode = "victim"
        continue

    # ------------
    # VICTIM STUFF
    # ------------
    elif rescue_mode == "victim":
        if sum(ball_found_qty) >= 3:
            print("Found all victims")
            rescue_mode = "block"

        servo["cam"].angle = EVAC_CAM_ANGLE
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
            detected_circles = sorted(detected_circles , key = lambda v: [v[1], v[1]],reverse=True)
            
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
            height_bar_maxRadius = [b + 20 for b in height_bar_minRadius]

            height_bars = [(img0.shape[0] / height_bar_qty) * i for i in range(height_bar_qty - 1, -1, -1)]
            height_bar_circles = [[] for i in range(height_bar_qty)]
            for (x, y, r) in detected_circles:
                for i in range(height_bar_qty):
                    if y >= height_bars[i]:
                        if r >= height_bar_minRadius[i] and r <= height_bar_maxRadius[i]: 
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
            for i, (x, y, r, bar) in enumerate(sorted_circles):
                cv2.circle(img0, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img0, (x, y), 2, (0, 255, 0), 3)
                cv2.putText(img0, f"{bar}-{i}-{r}", (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 125, 255), 2)

        # cv2.imshow("img0_blurred", img0_blurred)
        # cv2.imshow("img0_gray_rescue_scaled", img0_gray_rescue_scaled)
        cv2.imshow("img0_binary_rescue", img0_binary_rescue)
        cv2.imshow("img0_binary_rescue_clean", img0_binary_rescue_clean)
        cv2.imshow("img0", img0)

        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            # pr.print_stats(SortKey.TIME)
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

                THRESH_FINAL_APPROACH = 380
                if lowest_circle[1] > THRESH_FINAL_APPROACH:
                    print("Approaching")
                    approach_victim(1)
                m.run_tank_for_time(40, 40, 200)
        else:
            if last_circle_pos is not None and last_circle_pos[1] > 340:
                print("Approaching with extra distance")
                approach_victim(1.5)
                continue
            elif last_circle_pos is not None and last_circle_pos[1] <= 340 and abs(last_circle_pos[0] - (img0.shape[1] / 2)) < 250:
                # Did the circle vanish? Lets double check...
                if circle_check_counter < 3:
                    print("Circle may have vanished, double checking")
                    circle_check_counter += 1
                    time.sleep(0.3)
                    continue
                else:
                    print("Circle vanished")
            last_circle_pos = None
            print("Rotating")
            m.run_tank_for_time(40, -40, 300)
            time.sleep(0.3)

    # ------------------------
    # BLOCK FINDING AND RESCUE
    # ------------------------
    elif rescue_mode == "block":
        # Find the contours of the rescue blocks
        img0_block_mask = cv2.inRange(img0_hsv, config_values["rescue_block_hsv_threshold"][0], config_values["rescue_block_hsv_threshold"][1])
        img0_binary_rescue_block = cv2.bitwise_and(cv2.bitwise_not(img0_binary_rescue), img0_block_mask)
        img0_binary_rescue_block = cv2.morphologyEx(img0_binary_rescue_block, cv2.MORPH_OPEN, np.ones((13,13),np.uint8))

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
                bottom_block_approach_counter = 0
                if (
                    ("left" in contour["near_sides"] and "right" in contour["near_sides"])
                    or (
                        abs(cx - (img0.shape[1]/2)) < 50
                        and (("left" in contour_block["touching_sides"]) + ("right" in contour_block["touching_sides"])) != 1
                    )
                ):
                    m.run_tank(35, 35)
                    print("Block in middle - approach")
                elif cx < (img0.shape[1]/2) or "left" in contour_block["touching_sides"]:
                    m.run_tank(10, 35)
                    print("Block on left - turn left")
                else:
                    m.run_tank(35, 10)
                    print("Block on right - turn right")
        else:
            m.run_tank_for_time(35, -35, 100)
        
        servo["claw"].angle = -90

        if servo["lift"].angle > -60:
            # If the lift was down, give a bit of time before the camera moves
            servo["cam"].angle = -80
            servo["lift"].angle = -80
            time.sleep(0.5)
            servo["cam"].angle = EVAC_CAM_ANGLE
        else:
            servo["lift"].angle = -80
            servo["cam"].angle = EVAC_CAM_ANGLE

        # cv2.imshow("img0_blurred", img0_blurred)
        # cv2.imshow("img0_gray_rescue_scaled", img0_gray_rescue_scaled)
        cv2.imshow("img0_binary_rescue_block", img0_binary_rescue_block)
        cv2.imshow("img0_binary_rescue", img0_binary_rescue)
        cv2.imshow("img0_binary_rescue_clean", img0_binary_rescue_clean)
        cv2.imshow("img0", img0)

        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            # pr.print_stats(SortKey.TIME)
            program_active = False
            break

m.stop_all()
cam.stop()
cv2.destroyAllWindows()