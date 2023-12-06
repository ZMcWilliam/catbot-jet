import os
import time
import cv2
from helpers import camera as c
from helpers import config

TARGET_OUTPUT_PATH = "ml/data_in/silver/with"

# Ensure the output path exists
os.makedirs(TARGET_OUTPUT_PATH, exist_ok=True)

cam = c.CameraStream(
    camera_num=0,
    processing_conf=config.processing_conf
)
cam.wait_for_image()

try:
    start_time = time.time()
    frames_count = 0
    total_saved = 0
    while True:
        frames_count += 1
        if frames_count % 1000 == 0:
            elapsed_time = time.time() - start_time
            fps = frames_count / elapsed_time

            print(f"FPS: Cam {cam.get_fps():.2f}\tLoop {fps:.2f}\tElapsed {elapsed_time:.2f}")

        frame_processed = cam.read_stream_processed()
        img0 = frame_processed["silver"]
        img0 = cv2.resize(img0, (72, 66))

        cv2.imshow("img0", img0)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break

        if k != -1:
            total_saved += 1
            print(f"Saving image {total_saved} ({frames_count})")
            cv2.imwrite(os.path.join(TARGET_OUTPUT_PATH, f"{int(time.time())}-{total_saved}.png"), img0)

        time.sleep(0.001)
except KeyboardInterrupt:
    print("Main loop interrupted. Exiting...")

cam.stop()
