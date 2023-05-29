import json
import time
import cv2
import helper_camera
import numpy as np

cams = helper_camera.CameraController()
cams.start_stream(0)

calibration_images = {
    "w": [],
}

NUM_CALIBRATION_IMAGES = 50

time.sleep(1)

# Load the calibration map from the JSON file if it exists
calibration_data = {
    "calibration_value_w": 0, 
    "calibration_map_w": [],
}
try:
    with open("calibration.json", "r") as json_file:
        calibration_data = json.load(json_file)
except:
    pass # If the file doesn't exist, we'll create it later

while True:
    requested = input("Enter 'w' for white calibration, or 'q' to quit: ")
    
    if requested == "q":
        break

    if requested not in ["w", "b"]:
        continue

    calibration_images[requested] = []

    while NUM_CALIBRATION_IMAGES > len(calibration_images[requested]):
        img = cams.read_stream(0)
        if img is None:
            continue
        img = img[0:img.shape[0]-38, 0:img.shape[1]-70]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        print(img_gray[100][100])
        
        calibration_images[requested].append(img_gray)
        print(f"Calibration image {len(calibration_images[requested])} of {NUM_CALIBRATION_IMAGES} captured.")

        time.sleep(0.01)
        cv2.imshow("Calibration Image", img_gray)
        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            break

    # Calculate the average grayscale value across all calibration images
    calibration_data["calibration_value_" + requested] = np.mean([np.mean(img_gray) for img_gray in calibration_images[requested]])
    # Create an empty calibration map
    calibration_map = np.zeros_like(calibration_images[requested][0], dtype=np.float32)
    # Calculate the calibration map
    for img_gray in calibration_images[requested]:
        calibration_map += img_gray

    calibration_map //= len(calibration_images[requested])

    calibration_data["calibration_map_" + requested] = calibration_map.tolist()

    # Save the calibration map to the JSON file
    with open("calibration.json", "w") as json_file:
        json.dump(calibration_data, json_file, indent=4)

    print("Calibration images captured, updated calibration.json")
    
cams.stop()
