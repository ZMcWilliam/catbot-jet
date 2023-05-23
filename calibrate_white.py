import json
import cv2
import helper_camera
import numpy as np

cams = helper_camera.CameraController()
cams.start_stream(0)

calibration_images = []
calibration_value = 0

NUM_CALIBRATION_IMAGES = 10

while NUM_CALIBRATION_IMAGES > len(calibration_images):
    img = cams.read_stream(0)
    if img is None:
        continue
    img = img[0:img.shape[0]-38, 0:img.shape[1]-70]
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    calibration_images.append(img_gray)
    print(f"Calibration image {len(calibration_images)} of {NUM_CALIBRATION_IMAGES} captured.")
    
# Calculate the average grayscale value across all calibration images
calibration_value = np.mean([np.mean(img_gray) for img_gray in calibration_images])
# Create an empty calibration map
calibration_map = np.zeros_like(calibration_images[0], dtype=np.float32)
# Calculate the calibration map
for img_gray in calibration_images:
    calibration_map += img_gray

calibration_map //= len(calibration_images)

calibration_data = {"calibration_value": calibration_value, "calibration_map": calibration_map.tolist()}
with open("calibration.json", "w") as json_file:
    json.dump(calibration_data, json_file)

print(f"Calibration value: {calibration_value}")

cams.stop()
