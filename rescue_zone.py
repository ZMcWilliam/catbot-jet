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

# MAIN LOOP
program_sleep_time = 0.01 # Initial sleep time
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

    img0 = frame_processed["resized"]
    img0_clean = img0.copy() # Used for displaying the image without any overlays

    img0_gray = frame_processed["gray"]
    img0_binary = frame_processed["binary"]
    img0_green = frame_processed["green"]
    img0_line = frame_processed["line"]

    img0_binary_rescue = ((calibration_map_rescue * img0_gray > config_values["black_rescue_threshold"]) * 255).astype(np.uint8)
    img0_binary_rescue = cv2.morphologyEx(img0_binary_rescue, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

    servo["cam"].angle = 4

    # minDist = min distance between circles
    # param1 = high threshold for canny edge detection (sensitivity) - lower = more circles
    # param2 = accumulator threshold for circle detection - Higher = more reliable circles, but may miss some
    # minRadius = minimum radius of circle
    # maxRadius = maximum radius of circle
    img0_gray_rescue_scaled = img0_gray * config_values["rescue_circle_conf"]["grayScaleMultiplier"]
    img0_gray_rescue_scaled = np.clip(img0_gray_rescue_scaled, 0, 255).astype(np.uint8)

    img0_blurred = cv2.medianBlur(img0_gray_rescue_scaled, 9)
    circles = cv2.HoughCircles(img0_blurred, cv2.HOUGH_GRADIENT, **{
        "dp": config_values["rescue_circle_conf"]["dp"],
        "minDist": config_values["rescue_circle_conf"]["minDist"],
        "param1": config_values["rescue_circle_conf"]["param1"],
        "param2": config_values["rescue_circle_conf"]["param2"],
        "minRadius": config_values["rescue_circle_conf"]["minRadius"],
        "maxRadius": config_values["rescue_circle_conf"]["maxRadius"],
    })

    img0_circles = img0_clean.copy()
    if circles is not None:
        # Round the circle parameters to integers
        detected_circles = np.round(circles[0, :]).astype(int)
        detected_circles = sorted(detected_circles , key = lambda v: [v[1], v[1]],reverse=True)
        
        # If the circle is in the bottom half of the image, check that the radius is greater than 40
        valid_circles = []
        for (x, y, r) in detected_circles:
            if y > config_values["rescue_circle_conf"]["heightBuffer"] and r > config_values["rescue_circle_conf"]["lowHeightMinRadius"]:
                valid_circles.append([x, y, r])
            elif y <= config_values["rescue_circle_conf"]["heightBuffer"]:
                valid_circles.append([x, y, r])

        # Draw the detected circles on the original image
        for (x, y, r) in detected_circles:
            cv2.circle(img0_circles, (x, y), r, (0, 0, 255), 2)
            cv2.circle(img0_circles, (x, y), 2, (0, 0, 255), 3)

        for i, (x, y, r) in enumerate(valid_circles):
            cv2.circle(img0_circles, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img0_circles, (x, y), 2, (0, 255, 0), 3)
            cv2.putText(img0_circles, "#{}".format(i), (x , y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 255, 255), 2)

        print(f"{len(valid_circles)}/{len(detected_circles)} circles")
                                        
    cv2.imshow("img0_circles", img0_circles)
    cv2.imshow("img0_gray_rescue_scaled", img0_gray_rescue_scaled)
    cv2.imshow("img0_binary_rescue", img0_binary_rescue)

    k = cv2.waitKey(1)
    if (k & 0xFF == ord('q')):
        # pr.print_stats(SortKey.TIME)
        program_active = False
        break
    
    

m.stop_all()
cam.stop()
cv2.destroyAllWindows()