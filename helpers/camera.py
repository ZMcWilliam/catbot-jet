import os
import sys
import time
import signal
import atexit
import numpy as np
import cv2
from threading import Thread

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                       display_width=640, display_height=480, framerate=60, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true"
    )

def get_camera(num):
    cam = cv2.VideoCapture(gstreamer_pipeline(num), cv2.CAP_GSTREAMER)
    return cam

class CameraStream:
    def __init__(self, camera_num=0, processing_conf=None):
        self.num = camera_num
        self.cam = get_camera(self.num)
        self.processing_conf = processing_conf

        self.img = None
        self.processed = {
            "raw": None,
            "resized": None,
            "gray": None,
            "gray_scaled": None,
            "binary": None,
            "hsv": None,
            "green": None,
            "line": None
        }

        self.frames = 0
        self.start_time = 0

        self.capture_fails = 0

        self.stream_running = True
        self.capture_thread = Thread(target=self.update_stream, args=())
        self.capture_thread.start()

        atexit.register(self.atexit)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print(f"[CAMERA] C{self.num} Initialized")

    def restart_nvargus_daemon(self):
        print(f"[CAMERA] C{self.num} Attempting to restart nvargus-daemon...")
        os.system('sudo service nvargus-daemon restart')
        time.sleep(2) # Wait a bit for the service to restart
        print(f"[CAMERA] C{self.num} nvargus-daemon restarted. Trying to initialize camera again.")
        self.cam = get_camera(self.num) # Attempt to re-initialize the camera

    def take_image(self):
        ret, new_img = self.cam.read()
        if not ret:
            raise Exception("[CAMERA] Failed to capture an image.")
        print(f"[CAMERA] C{self.num} Image captured")
        self.img = new_img
        return self.img

    def read_stream(self):
        return self.img

    def read_stream_processed(self):
        return self.processed

    def wait_for_image(self):
        waitingText = f"[CAMERA] C{self.num} Waiting for first image..."
        time.sleep(0.5)
        while self.img is None or (self.processing_conf is not None and self.processed["raw"] is None):
            time.sleep(0.1)
            print(waitingText, end="\r")
            waitingText += "."
        print(f"{waitingText} Ready!")

    def update_stream(self):
        self.start_time = time.time()

        while self.stream_running:
            self.frames += 1
            ret, self.img = self.cam.read()
            if not ret:
                self.capture_fails += 1
                print(f"[CAMERA] C{self.num} WARNING: Failed to capture an image, retrying... ({self.capture_fails}/5)")
                if self.capture_fails >= 5:
                    self.restart_nvargus_daemon()
                    self.capture_fails = 0
                time.sleep(0.1)
                continue

            if self.processing_conf is not None:
                self.process_image()

        if self.cam is not None:
            self.cam.release()
            self.cam = None

    def resize_image(self, img, target_w=290, target_h=264, offset_x=-2, offset_y=104):
        start_x = (img.shape[1] // 2 - target_w // 2) + offset_x
        start_y = 0 + offset_y

        resized = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return resized
    
    def resize_image_obstacle(self, img, target_w=525, target_h=200, offset_x=-2, offset_y=65):
        start_x = (img.shape[1] // 2 - target_w // 2) + offset_x
        start_y = 0 + offset_y

        resized = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return resized

    def process_image(self):
        image = self.img
        if image is None:
            raise Exception(f"[CAMERA] C{self.num} has no image to process, run .take_image() first")

        if self.processing_conf is None:
            raise Exception(f"[CAMERA] C{self.num} has no conf for processing")

        resized = self.resize_image(image)
        resized_silver = self.resize_image(image, offset_y=0)
        # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert image to grayscale/HSV
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_silver = cv2.cvtColor(resized_silver, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Blur the grayscale image slightly
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_silver = cv2.GaussianBlur(gray_silver, (5, 5), 0)

        # Scale white values based on the inverse of the calibration map
        gray_scaled = self.processing_conf["calibration_map"] * gray
        gray_silver_scaled = self.processing_conf["calibration_map_silver"] * gray_silver

        # Get the binary image
        black_line_threshold = self.processing_conf["black_line_threshold"]
        binary = ((gray_scaled > black_line_threshold) * 255).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        silver_binary = ((gray_silver_scaled > black_line_threshold) * 255).astype(np.uint8)
        silver_binary = cv2.morphologyEx(silver_binary, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        # Find green in the image
        green_turn_hsv_threshold = self.processing_conf["green_turn_hsv_threshold"]
        green = cv2.bitwise_not(cv2.inRange(hsv, green_turn_hsv_threshold[0], green_turn_hsv_threshold[1]))
        green = cv2.erode(green, np.ones((5, 5), np.uint8), iterations=1)

        # Find the line, by removing the green from the image (since green looks like black when grayscaled)
        line = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=2)
        line = cv2.bitwise_or(line, cv2.bitwise_not(green))

        # Only set the processed data once it is all populated, to avoid partial data being read
        self.processed = {
            "raw": image,
            "resized": resized,
            "gray": gray,
            "gray_scaled": gray_scaled,
            "silver": gray_silver,
            "silver_binary": silver_binary,
            "binary": binary,
            "hsv": hsv,
            "green": green,
            "line": line,
        }

    def stop(self):
        print(f"[CAMERA] C{self.num} Stopping stream")
        self.stream_running = False
        
        # Explicitly set the GStreamer elements to the NULL state
        if self.cam is not None:
            self.cam.release()
            self.cam = None

    def atexit(self):
        print(f"\n[CAMERA] C{self.num} Atexit called. Stopping stream...")
        time.sleep(1)
        self.stop()

    def signal_handler(self, signum, frame):
        print(f"\n[CAMERA] C{self.num} Signal received: {signum}. Initiating graceful shutdown...")
        time.sleep(1)
        self.stop()
        print(f"\n[CAMERA] C{self.num} Stream stopped. Exiting...")

    def set_processing_conf(self, conf):
        self.processing_conf = conf

    def get_fps(self):
        duration = time.time() - self.start_time
        return self.frames / duration
