import time
import os
import sys
import signal
import subprocess
import Jetson.GPIO as GPIO
from colorama import init
# from helpers import motorkit as m
init()

RUN_PIN = "GP167"

# Motorkit forces GPIO mode to TEGRA_SOC, so we must use that
GPIO.setmode(GPIO.TEGRA_SOC)
GPIO.setup(RUN_PIN, GPIO.IN)

print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStarting CatBot...")

start_time = time.time()
stopCheck = 10

if not GPIO.input(RUN_PIN):
    print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower is disabled, waiting to start...")
p = None
state = 0

follower_process = None

def start_follower():
    global follower_process
    if follower_process is None:
        print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStarting Follower...")
        preload = 'LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1 /lib/aarch64-linux-gnu/libGLdispatch.so"'
        follower_process = subprocess.Popen(["bash", "-c", preload + " && sudo service nvargus-daemon restart && python3 follower.py"])

# def stop_follower():
#     global follower_process
#     if follower_process is not None:
#         print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStopping Follower...")
#         follower_process.send_signal(signal.SIGINT)
#         try:
#             follower_process.wait(timeout=10)
#         except subprocess.TimeoutExpired:
#             print("\033[1;33m[RUNNER]\033[1;m \033[1;37mSurpassed Timeout Terminating...")
#             follower_process.terminate()
#         follower_process = None
#         print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStopped Follower.")

def stop_follower():
    global follower_process
    if follower_process is not None:
        print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStopping Follower...")
        try:
            os.kill(follower_process.pid, signal.SIGINT)
            follower_process.wait(timeout=10)
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower stopped.")
        except subprocess.TimeoutExpired:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mSurpassed Timeout Terminating...")
            follower_process.terminate()
        except Exception as e:
            print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37mError stopping Follower: {e}")
        follower_process = None

while True:
    try:
        input_state = GPIO.input(RUN_PIN)
        if p is not None:
            process_state = follower_process.poll()
            if input_state and process_state is not None:
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mResurrecting Follower...")
                state = 0
        if input_state and state == 0:
            time.sleep(0.4)
            state = 1
            start_follower()
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower started")
        if not input_state and state == 0:
            time.sleep(0.1)
        if not input_state and state == 1:
            stopCheck -= 1
            if stopCheck == 0:
                stop_follower()
                time.sleep(0.4)
                state = 0
                stopCheck = 10
                # m.stop_all() # Stop all motors
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower stopped")
            else:
                time.sleep(0.01)
        elif stopCheck != 10:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;40mAborting possible false stop\033[1;37m")
            stopCheck = 10
        else:
            stopCheck = 10
    except KeyboardInterrupt:
        try:
            resp = input("\033[1;33m[RUNNER]\033[1;m \033[1;37mAre you sure you want to quit? (y/n) ")
        except KeyboardInterrupt:
            resp = "n"
        if resp == "y":
            try:
                if state == 1:
                    stop_follower()
            except Exception:
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFailed to stop follower, is it running?")
            # m.stop_all() # Stop all motors
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower stopped")
            sys.exit()
        else:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mAborted quit")
            continue

