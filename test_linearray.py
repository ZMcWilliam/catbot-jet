import board
import time
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C(board.SCL, board.SDA)
ads_addresses = [0x49, 0x48, 0x4B]
ads_gain = 1
ads_mode = ADS.Mode.SINGLE

PORT_ADS_LINE = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "C0", "C1", "C2", "C3"]

ads = []
for address in ads_addresses:
    ads_instance = ADS.ADS1015(i2c, address=address)
    ads_instance.gain = ads_gain
    ads_instance.mode = ads_mode
    ads.append(ads_instance)

channels = []
for ads_port in PORT_ADS_LINE:
    ads_instance = ads[ord(ads_port[0]) - ord("A")]
    channel = AnalogIn(ads_instance, getattr(ADS, f"P{ads_port[1]}"))
    channels.append(channel)
    
# cal_min = None
# cal_max = None

# while True:
#     vals = [channel.value for channel in channels]

#     if cal_min is None:
#         cal_min = vals[:]
#         cal_max = vals[:]

#     for i in range(len(vals)):
#         if vals[i] < cal_min[i]:
#             cal_min[i] = vals[i]
#         if vals[i] > cal_max[i]:
#             cal_max[i] = vals[i]
        
#     print(vals, cal_min, cal_max)

las_min = [1312, 1248, 1280, 1280, 1248, 1280, 1264, 1312, 1296, 1248, 1264, 1344] 
las_max = [10352, 9600, 10736, 10944, 8544, 11264, 10480, 11952, 10848, 8080, 9584, 12800]

frames = 0
start = time.time()
while True:
    frames += 1
    vals = [channel.value for channel in channels]
    vals = [int((vals[i] - las_min[i]) / (las_max[i] - las_min[i]) * 100) for i in range(len(vals))]

    values_str = "\t".join([f"{val:4d}" for val in vals])

    fps_str = f"\tFPS: {int(frames / (time.time() - start))}"
    print(values_str + fps_str)
