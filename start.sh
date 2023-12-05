#!/bin/bash
xrandr --fb 1600x960
xrandr --output DP-1 --scale 2x2
sleep 1

wmctrl -r "$(xdotool getwindowname $(xdotool getactivewindow))" -e 0,100,350,-1,-1

export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1 /lib/aarch64-linux-gnu/libGLdispatch.so"
python3 runner.py
