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
    def __init__(self):
        self.cameras = [CameraStream(0)]

    def stop(self):
        for cam in self.cameras:
            cam.stop_stream()

    def read_stream(self, num):
        return self.cameras[num].read_stream()

    def start_stream(self, num):
        self.cameras[num].start_stream()

    def stop_stream(self, num):
        self.cameras[num].stop_stream()

    def get_fps(self, num):
        return self.cameras[num].get_fps()

class CameraStream:
    def __init__(self, num=0):
        self.num = num
        self.cam = get_camera(self.num)

        self.stream_running = False
        self.frame = None
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
        
        self.cam.stop()
        self.stop_time = time.time()

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
