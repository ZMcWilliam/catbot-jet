import adafruit_vl6180x
from helpers.tof import RangeSensorMonitor

tof = RangeSensorMonitor()
tof.start()

try:
    while True:
        # - adafruit_vl6180x.ALS_GAIN_1 = 1x
        # - adafruit_vl6180x.ALS_GAIN_1_25 = 1.25x
        # - adafruit_vl6180x.ALS_GAIN_1_67 = 1.67x
        # - adafruit_vl6180x.ALS_GAIN_2_5 = 2.5x
        # - adafruit_vl6180x.ALS_GAIN_5 = 5x
        # - adafruit_vl6180x.ALS_GAIN_10 = 10x
        # - adafruit_vl6180x.ALS_GAIN_20 = 20x
        # - adafruit_vl6180x.ALS_GAIN_40 = 40x
        light_lux = tof.sensor.read_lux(adafruit_vl6180x.ALS_GAIN_1)
        range_mm = tof.range_mm
        print(f"{range_mm}mm {light_lux}lux")
except KeyboardInterrupt:
    tof.stop()
    tof.join()
