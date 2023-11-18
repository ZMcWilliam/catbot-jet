import time
import cv2
from helpers import camera

cam = camera.CameraStream(
    camera_num = 0,
)
cam.wait_for_image()

# Get max width and height
img0 = cam.read_stream()
max_width = img0.shape[1]
max_height = img0.shape[0]

cam_crop_w = 290
cam_crop_h = 264
cam_crop_offset_x = int(max_width / 2) -2
cam_crop_offset_y = 104

cv2.namedWindow("Config")

def update_width(val):
    global cam_crop_w
    cam_crop_w = val

def update_height(val):
    global cam_crop_h
    cam_crop_h = val

def update_offset_x(val):
    global cam_crop_offset_x
    cam_crop_offset_x = int(val - (max_width / 2))

def update_offset_y(val):
    global cam_crop_offset_y
    cam_crop_offset_y = val

cv2.createTrackbar("Crop Width", "Config", cam_crop_w, max_width, update_width)
cv2.createTrackbar("Crop Height", "Config", cam_crop_h, max_height, update_height)

cv2.createTrackbar("Crop Offset X", "Config", cam_crop_offset_x, max_width, update_offset_x)
cv2.createTrackbar("Crop Offset Y", "Config", cam_crop_offset_y, max_height, update_offset_y)

try:
    start_time = time.time()
    frames_count = 0
    while True:
        frames_count += 1
        if frames_count % 1000 == 0:
            elapsed_time = time.time() - start_time
            fps = frames_count / elapsed_time

            print(f"FPS: Cam {cam.get_fps():.2f}\tLoop {fps:.2f}\tElapsed {elapsed_time:.2f}")

        # img0 = cam.read_stream_processed()
        img0 = cam.read_stream()
        cv2.imshow("Raw", img0)

        resized = cam.resize_image(img0, cam_crop_w, cam_crop_h, cam_crop_offset_x, cam_crop_offset_y)
        cv2.imshow("Resized", resized)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break

        time.sleep(0.001)
except KeyboardInterrupt:
    print("Main loop interrupted. Exiting...")

cam.stop()
