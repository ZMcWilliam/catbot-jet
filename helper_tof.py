import threading
import time
import board
import busio
import adafruit_vl6180x

class RangeSensorMonitor(threading.Thread):
    def __init__(self):
        super().__init__()

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl6180x.VL6180X(self.i2c)
        self.range_mm = None
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            self.range_mm = self.sensor.range
            time.sleep(0.1)

    def stop(self):
        self._stop_event.set()
