from picamera2 import Picamera2
import time
import cv2
from threading import Thread

def get_camera(num):
    cam = Picamera2(num)
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

class CameraStream:
    def __init__(self, num=0):
        self.num = num
        self.cam = get_camera(self.num)

        self.stream_running = False
        self.frame = None
        
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

        while self.stream_running:
            self.frame = self.cam.helpers.make_array(self.cam.capture_buffer(), self.cam.camera_configuration()["main"])
        
        self.cam.stop()

    def stop_stream(self):
        print(f"[CAMERA] Stopping stream for Camera {self.num}")
        self.stream_running = False

if __name__ == "__main__":
    camNum = 0
    cams = CameraController()
    cams.start_stream(camNum)
    time.sleep(1)
    
    while True:
        img1 = cams.read_stream(camNum)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)

        cv2.imshow(f"Camera {camNum}", img1)

        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            break
    
    cams.stop()