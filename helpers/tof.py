import threading
import time
import board
import busio
import adafruit_vl53l1x

class RangeSensorMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l1x.VL53L1X(self.i2c)
        self.range_mm = 0
        self._stop_event = threading.Event()

        self.start_time = time.time()
        self.frames = 0

    def run(self):
        self.sensor.start_ranging()
        while not self._stop_event.is_set():
            if self.sensor.data_ready:
                new_dist = self.sensor.distance
                if new_dist is not None:
                    self.range_mm = new_dist * 10
                self.sensor.clear_interrupt()
                self.frames += 1
        self.sensor.stop_ranging()

    def stop(self):
        self._stop_event.set()
    
    @property
    def fps(self):
        return int(self.frames / (time.time() - self.start_time))
