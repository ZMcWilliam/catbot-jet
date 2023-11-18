import time
from helpers.cmps14 import CMPS14

cmps14 = CMPS14(7, 0x61)

while True:
    bearing_8bit = cmps14.read_bearing_8bit()
    bearing_16bit = cmps14.read_bearing_16bit()
    pitch = cmps14.read_pitch()
    roll = cmps14.read_roll()

    print(f"{bearing_16bit}°"
          + f"\tPitch: {pitch}°"
          + f"\tRoll: {roll}°")

    time.sleep(0.02)
