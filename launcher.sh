#!/bin/bash
cd /home/pi/Desktop/CatBot/
source env/bin/activate
cat /home/pi/Desktop/CatBot/motd.txt
export GPIOZERO_PIN_FACTORY=pigpio
DISPLAY=:0 xterm -fs 9 -geometry 115x30 -T 'CatBot NEO' -fa 'Monospace' -e 'bash -c "python3 runner.py; read x"'
read x
q
