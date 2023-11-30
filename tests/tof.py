import time
from helpers.tof import RangeSensorMonitor

tof = RangeSensorMonitor()
tof.start()

try:
    while True:
        range_mm = tof.range_mm
        print(f"{range_mm:4.0f}mm @ {tof.fps:3.0f}fps")
        time.sleep(0.1)
except KeyboardInterrupt:
    tof.stop()
    tof.join()
