from ADCPi import ADCPi

adc = ADCPi(0x68, 0x69, 12)

# cal_min = None
# cal_max = None

# while True:
#     vals = [adc.read_raw(i) for i in range(1, 9)]

#     if cal_min is None:
#         cal_min = vals[:]
#         cal_max = vals[:]

#     for i in range(8):
#         if vals[i] < cal_min[i]:
#             cal_min[i] = vals[i]
#         if vals[i] > cal_max[i]:
#             cal_max[i] = vals[i]
        
#     print(vals, cal_min, cal_max)


las_min = [67, 68, 66, 68, 68, 70, 66, 69]
las_max = [127, 143, 124, 145, 135, 150, 114, 172]

while True:
    vals = [adc.read_raw(i) for i in range(1, 9)]
    vals = [int((vals[i] - las_min[i]) / (las_max[i] - las_min[i]) * 100) for i in range(8)]
    print(vals)
