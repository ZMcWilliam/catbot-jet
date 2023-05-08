import board
import time
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C(board.SCL, board.SDA)

ads_1 = ADS.ADS1015(i2c, address=0x48)
ads_1.gain = 1
ads_1.mode = ADS.Mode.SINGLE

ads_2 = ADS.ADS1015(i2c, address=0x49)
ads_1.gain = 1
ads_2.mode = ADS.Mode.SINGLE

ads_3 = ADS.ADS1015(i2c, address=0x4B)
ads_1.gain = 1
ads_3.mode = ADS.Mode.SINGLE

chan_1_0 = AnalogIn(ads_1, ADS.P0)
chan_1_1 = AnalogIn(ads_1, ADS.P1)
chan_1_2 = AnalogIn(ads_1, ADS.P2)
chan_1_3 = AnalogIn(ads_1, ADS.P3)

chan_2_0 = AnalogIn(ads_2, ADS.P0)
chan_2_1 = AnalogIn(ads_2, ADS.P1)
chan_2_2 = AnalogIn(ads_2, ADS.P2)
chan_2_3 = AnalogIn(ads_2, ADS.P3)

chan_3_0 = AnalogIn(ads_3, ADS.P0)
chan_3_1 = AnalogIn(ads_3, ADS.P1)
chan_3_2 = AnalogIn(ads_3, ADS.P2)
chan_3_3 = AnalogIn(ads_3, ADS.P3)

frames = 0
start = time.time()
while True:
    frames += 1
    print(f"{int(chan_1_0.value)} "
        + f"{int(chan_1_1.value)} "
        + f"{int(chan_1_2.value)} "
        + f"{int(chan_1_3.value)} "
        + f"{int(chan_2_0.value)} "
        + f"{int(chan_2_1.value)} "
        + f"{int(chan_2_2.value)} "
        + f"{int(chan_2_3.value)} "
        + f"{int(chan_3_0.value)} "
        + f"{int(chan_3_1.value)} "
        + f"{int(chan_3_2.value)} "
        + f"{int(chan_3_3.value)} "
        + f"\tFPS: {int(frames / (time.time() - start))}")
