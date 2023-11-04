import time
import cv2
import json
import helper_camera
import numpy as np
from threading import Thread
from picamera2 import Picamera2

cam = helper_camera.CameraStream()
cam.start_stream()
time.sleep(1)

# Load the calibration map from the JSON file
with open("calibration.json", "r") as json_file:
    calibration_data = json.load(json_file)
calibration_map = np.array(calibration_data["calibration_map_w"], dtype=np.float32)

while True:
    img0 = cam.read_stream()
    # img0 = img0[0:img0.shape[0]-38, 0:img0.shape[1]]
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    # img0_gray = cv2.equalizeHist(img0_gray)
    img0_gray = cv2.GaussianBlur(img0_gray, (5, 5), 0)

    # print(img0_gray[100][100], calibration_map[100][100])

    # img0_gray_scaled = 240 / np.clip(calibration_map, a_min=1, a_max=None) * img0_gray  # Scale white values based on the inverse of the calibration map
    # img0_gray_scaled = np.clip(img0_gray_scaled, 0, 255)    # Clip the scaled image to ensure values are within the valid range
    # img0_gray_scaled = img0_gray_scaled.astype(np.uint8)    # Convert the scaled image back to uint8 data type
    
    cv2.imshow(f"img0", img0)
    cv2.imshow(f"img0_gray", img0_gray)
    # cv2.imshow(f"img0_gray_scaled", img0_gray_scaled)

    k = cv2.waitKey(1)
    if (k & 0xFF == ord('q')):
        break

cam.stop()
