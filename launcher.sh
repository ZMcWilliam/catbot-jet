#!/bin/bash
cd /home/pi/Desktop/CatBot/
source env/bin/activate
DISPLAY=:0 xterm -T 'CatBot NEO' -fa 'Monospace' -e 'python3 runner.py; read x'
read x
q
