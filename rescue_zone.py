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
    "rescue_circle_conf": config_data["rescue_circle_conf"],
}

cam = helper_camera.CameraStream(0, {
    "calibration_map": calibration_map,
    "black_line_threshold": config_values["black_line_threshold"],
    "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
})
cam.start_stream()

frames = 0
start_time = time.time()
fpsTime = time.time()
fpsLoop = 0
fpsCamera = 0

hasMovedWindows = False

last_circle_pos = None
circle_check_counter = 0

EVAC_CAM_ANGLE = 4

def approach_victim(time_to_approach):
    global last_circle_pos
    time.sleep(0.2)
    m.run_tank(40, 40)
    time.sleep(time_to_approach - 0.1)
    servo["claw"].angle = -90
    time.sleep(0.1)
    m.stop_all()
    time.sleep(0.2)
    servo["cam"].angle = -80 # Ensure cam is out of the way before we do lifting actions
    m.run_tank_for_time(-40, -40, 800)
    servo["lift"].angle = -80
    time.sleep(0.8)
    servo["claw"].angle = -45
    time.sleep(0.5)
    servo["lift"].angle = 40
    servo["claw"].angle = 0
    time.sleep(0.2)
    servo["cam"].angle = EVAC_CAM_ANGLE
    time.sleep(0.7)
    last_circle_pos = None
    
# MAIN LOOP
program_sleep_time = 0.01 # Initial sleep time

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
    img0_green = frame_processed["green"]
    img0_line = frame_processed["line"]

    img0_binary_rescue = ((calibration_map_rescue * img0_gray > config_values["black_rescue_threshold"]) * 255).astype(np.uint8)
    img0_binary_rescue = cv2.morphologyEx(img0_binary_rescue, cv2.MORPH_OPEN, np.ones((13,13),np.uint8))

    servo["cam"].angle = EVAC_CAM_ANGLE
    servo["lift"].angle = 40
    servo["claw"].angle = 0

    # minDist = min distance between circles
    # param1 = high threshold for canny edge detection (sensitivity) - lower = more circles
    # param2 = accumulator threshold for circle detection - Higher = more reliable circles, but may miss some
    # minRadius = minimum radius of circle
    # maxRadius = maximum radius of circle
    img0_gray_rescue_scaled = img0_gray * config_values["rescue_circle_conf"]["grayScaleMultiplier"]
    img0_gray_rescue_scaled = np.clip(img0_gray_rescue_scaled, 0, 255).astype(np.uint8)

    img0_blurred = cv2.medianBlur(img0_gray_rescue_scaled, 9)

    # # Find the highest point of the image which has a full row of white pixels at least 10 pixels high
    # top_white_row = -1
    # for y in range(0, img0_binary_rescue.shape[0] - 10):
    #     if np.all(img0_binary_rescue[y:y+10, :] == 255):
    #         top_white_row = y
    #         break

    # # Set all pixels above this row to black
    # if top_white_row != -1:
    #     img0_binary_rescue[:top_white_row, :] = 0
    #     img0_blurred[:top_white_row, :] = 0

    segment_width = 30

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

    circles = cv2.HoughCircles(img0_blurred, cv2.HOUGH_GRADIENT, **{
        "dp": config_values["rescue_circle_conf"]["dp"],
        "minDist": config_values["rescue_circle_conf"]["minDist"],
        "param1": config_values["rescue_circle_conf"]["param1"],
        "param2": config_values["rescue_circle_conf"]["param2"],
        "minRadius": config_values["rescue_circle_conf"]["minRadius"],
        "maxRadius": config_values["rescue_circle_conf"]["maxRadius"],
    })

    img0_circles = img0_clean.copy()
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
            cv2.circle(img0_circles, (x, y), r, (0, 0, 255), 2)
            cv2.circle(img0_circles, (x, y), 2, (0, 0, 255), 3)

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
            cv2.line(img0_circles, (0, int(height_bars[i])), (img0.shape[1], int(height_bars[i])), (255, 255, 255), 1)
        
        # Draw the sorted circles on the original image
        for i, (x, y, r, bar) in enumerate(sorted_circles):
            cv2.circle(img0_circles, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img0_circles, (x, y), 2, (0, 255, 0), 3)
            cv2.putText(img0_circles, f"{bar}-{i}-{r}", (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 125, 255), 2)

    cv2.imshow("img0_circles", img0_circles)
    cv2.imshow("img0_blurred", img0_blurred)
    cv2.imshow("img0_gray_rescue_scaled", img0_gray_rescue_scaled)
    cv2.imshow("img0_binary_rescue", img0_binary_rescue)

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

m.stop_all()
cam.stop()
cv2.destroyAllWindows()