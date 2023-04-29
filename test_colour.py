import time
import board
import adafruit_tca9548a
from PiicoDev_VEML6040 import PiicoDev_VEML6040

PORT_COL_L = 5
PORT_COL_R = 4

i2c = board.I2C()
tca = adafruit_tca9548a.TCA9548A(i2c)

mx_locks = [False] * 8
def mx_remove_locks():
    for i in range(8):
        if mx_locks[i] == True:
            tca[i].unlock()
            mx_locks[i] = False

def mx_select(port):
    mx_remove_locks()
    tca[port].try_lock()
    mx_locks[port] = True

mx_select(PORT_COL_L)
col_l = PiicoDev_VEML6040()

mx_select(PORT_COL_R)
col_r = PiicoDev_VEML6040()

def read_col(port):
    mx_select(port)
    data = {
        "hsv": col_l.readHSV(),
        "hue": col_l.classifyHue({"red":0,"yellow":60,"green":120,"cyan":180,"blue":240,"magenta":300})
    }
    mx_remove_locks()
    return data

while True:
    l = read_col(PORT_COL_L)
    r = read_col(PORT_COL_R)
    
    print(f"COL LEFT: {l['hue']} ({int(l['hsv']['hue'])})\tCOL RIGHT: {r['hue']} ({int(r['hsv']['hue'])})")