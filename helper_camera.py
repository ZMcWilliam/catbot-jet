import time
import cv2
import numpy as np
from picamera2 import Picamera2
from threading import Thread

def get_camera(num):
    cam = Picamera2(num)
    if num == 0: # Pi camera
        available_modes = cam.sensor_modes
        available_modes.sort(key=lambda x: x["fps"], reverse=True)
        chosen_mode = available_modes[0]
        cam.video_configuration = cam.create_video_configuration(
                raw={"size": chosen_mode["size"], "format": chosen_mode["format"].format},
                main={"size": (640,480)},
        )
        cam.configure("video")
        cam.set_controls({"FrameRate": 120,"ExposureTime": 10000})

    return cam

class CameraController:
    def __init__(self, processing_conf=None):
        self.cameras = [
            CameraStream(0, processing_conf)
        ]

    def stop(self):
        for cam in self.cameras:
            cam.stop_stream()

    def read_stream(self, num):
        return self.cameras[num].read_stream()
    
    def read_stream_processed(self, num):
        return self.cameras[num].read_stream_processed()

    def start_stream(self, num):
        self.cameras[num].start_stream()

    def stop_stream(self, num):
        self.cameras[num].stop_stream()

    def get_fps(self, num):
        return self.cameras[num].get_fps()

class CameraStream:
    def __init__(self, num=0, processing_conf=None):
        self.num = num
        self.cam = get_camera(self.num)
        self.processing_conf = processing_conf

        if self.processing_conf is None:
            print("Warning: No processing configuration provided, images will not be pre-processed")

        self.stream_running = False
        
        self.frame = None


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
        self.stop_time = 0

    def take_image(self):
        self.cam.start()
        new_frame = self.frame
        while new_frame == self.frame:
            new_frame = self.cam.helpers.make_array(self.cam.capture_buffer(), self.cam.camera_configuration()["main"])
            print(f"C{self.num} Frame Captured")
        self.cam.stop()
        self.frame = new_frame
        return self.frame
        
    def read_stream(self):
        if not self.stream_running: 
            raise Exception(f"Camera {self.num} is not active, run .start_stream() first")

        return self.frame

    def read_stream_processed(self):
        if not self.stream_running: 
            raise Exception(f"Camera {self.num} is not active, run .start_stream() first")

        return self.processed

    def start_stream(self):
        print(f"[CAMERA] Starting stream for Camera {self.num}")
        self.thread = Thread(target=self.update_stream, args=())
        self.stream_running = True
        self.thread.start()
        
    def update_stream(self):
        self.cam.start()
        self.start_time = time.time()

        while self.stream_running:
            self.frames += 1
            self.frame = self.cam.helpers.make_array(self.cam.capture_buffer(), self.cam.camera_configuration()["main"])

            if self.processing_conf is not None:
                self.process_frame()

        self.cam.stop()
        self.stop_time = time.time()

    def process_frame(self):
        frame = self.frame
        if frame is None:
            raise Exception(f"Camera {self.num} has no frame to process, run .take_image() first")
        
        if self.processing_conf is None:
            raise Exception(f"Camera {self.num} has no conf for processing")

        resized = frame[0:frame.shape[0]-38, 0:frame.shape[1]]
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Find the black in the image
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Scale white values based on the inverse of the calibration map
        gray_scaled = self.processing_conf["calibration_map"] * gray

        # Get the binary image
        black_line_threshold = self.processing_conf["black_line_threshold"]
        binary = ((gray_scaled > black_line_threshold) * 255).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

        # Find green in the image
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        green_turn_hsv_threshold = self.processing_conf["green_turn_hsv_threshold"]
        green = cv2.bitwise_not(cv2.inRange(hsv, green_turn_hsv_threshold[0], green_turn_hsv_threshold[1]))
        green = cv2.erode(green, np.ones((5,5),np.uint8), iterations=1)

        # Find the line, by removing the green from the image (since green looks like black when grayscaled)
        line = cv2.dilate(binary, np.ones((5,5),np.uint8), iterations=2)
        line = cv2.bitwise_or(line, cv2.bitwise_not(green))

        # Only set the processed data once it is all populated, to avoid partial data being read
        self.processed = {
            "raw": frame,
            "resized": resized,
            "gray": gray,
            "gray_scaled": gray_scaled,
            "binary": binary,
            "hsv": hsv,
            "green": green,
            "line": line
        }

    def stop_stream(self):
        print(f"[CAMERA] Stopping stream for Camera {self.num}")
        self.stream_running = False

    def get_fps(self):
        return int(self.frames/(time.time() - self.start_time))

if __name__ == "__main__":
    camNum = 0
    cams = CameraController()
    cams.start_stream(camNum)
    time.sleep(1)
    
    while True:
        img = cams.read_stream(camNum)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

        gamma = 1.3
        gammatable = [((i / 255) ** (1 / gamma)) * 255 for i in range(256)]
        gammatable = np.array(gammatable, np.uint8)
        img = cv2.LUT(img, gammatable)

        cv2.imshow(f"Camera {camNum}", img)

        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            break
    
    print(f"Average FPS: {cams.get_fps(camNum)}")
    cams.stop()
