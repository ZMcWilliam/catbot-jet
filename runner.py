import time
import os
import sys
import psutil
import signal
import subprocess
import threading
import Jetson.GPIO as GPIO
from colorama import init
from helpers import motorkit as m
init()

RUN_PIN = "GP167"
LED_R_PIN = "GP125"
LED_G_PIN = "GP123"

# Motorkit forces GPIO mode to TEGRA_SOC, so we must use that
GPIO.setmode(GPIO.TEGRA_SOC)
GPIO.setup(RUN_PIN, GPIO.IN)

GPIO.setup([LED_R_PIN, LED_G_PIN], GPIO.OUT)
GPIO.output([LED_R_PIN, LED_G_PIN], GPIO.LOW)

print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37mStarting CatBot... (pid: {os.getpid()})")

start_time = time.time()
stopCheck = 10

if not GPIO.input(RUN_PIN):
    print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower is disabled, waiting to start...")

state = 0

runner_active = True
follower_stopping = False
follower_process = None
follower_process_id = -1

def led_thread_loop():
    global runner_active
    global start_time
    global follower_process
    while True:
        if not runner_active: break

        GPIO.output(LED_G_PIN, GPIO.HIGH if follower_process else GPIO.LOW)
        
        # Process is stopping
        if follower_stopping:
            if (time.time() - start_time) % 1 < 0.5:
                GPIO.output(LED_R_PIN, GPIO.HIGH)
            else:
                GPIO.output(LED_R_PIN, GPIO.LOW)
        # Process is not running, but PID exists...
        elif follower_process is None and psutil.pid_exists(follower_process_id):
            if (time.time() - start_time) % 0.8 < 0.4:
                GPIO.output(LED_R_PIN, GPIO.HIGH)
            else:
                GPIO.output(LED_R_PIN, GPIO.LOW)
        # Valid state
        else:
            GPIO.output(LED_R_PIN, GPIO.HIGH if follower_process is None else GPIO.LOW)
        
        time.sleep(0.1)

led_thread = threading.Thread(target=led_thread_loop)
led_thread.start()

def start_follower():
    global follower_process
    global follower_process_id
    if follower_process is None:
        print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStarting Follower...")
        preload = 'LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1 /lib/aarch64-linux-gnu/libGLdispatch.so"'
        follower_process = subprocess.Popen(["bash", "-c", preload + " && sudo service nvargus-daemon restart && python3 follower.py"])
        follower_process_id = follower_process.pid

def stop_follower():
    global follower_stopping
    global follower_process
    if follower_process is not None:
        print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37mStopping Follower (pid: {follower_process.pid})...")
        follower_stopping = True
        try:
            follower_process.send_signal(signal.SIGINT)
            follower_process.wait(timeout=20)
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStop: Successful")
        except subprocess.TimeoutExpired:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStop: Surpassed Timeout. Terminating...")
            follower_process.kill()
        except Exception as e:
            print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37mStop: Error: {e}")
        follower_stopping = False
        follower_process = None

while runner_active:
    try:
        input_state = GPIO.input(RUN_PIN)
        GPIO.output(LED_R_PIN, GPIO.LOW if input_state else GPIO.HIGH)
        if follower_process is not None:
            process_state = psutil.pid_exists(follower_process_id)
            if input_state and not process_state:
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mResurrecting Follower...")
                state = 0
        if input_state and state == 0:
            time.sleep(0.4)
            state = 1
            start_follower()
            sys.stdout.write(f"\x1b]2;CatBot Follower (pid: {follower_process.pid})\x07")
            print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower started (pid: {follower_process.pid})")
        if not input_state and state == 0:
            sys.stdout.write("\x1b]2;Stopped: CatBot Follower\x07")
            time.sleep(0.1)
        if not input_state and state == 1:
            stopCheck -= 1
            if stopCheck == 0:
                sys.stdout.write(f"\x1b]2;Stopping: CatBot Follower (pid: {follower_process.pid})\x07")
                stop_start_time = time.time()
                m.stop_all()
                stop_follower()
                m.stop_all()
                time.sleep(0.4)
                state = 0
                stopCheck = 10
                print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower stopped, took {time.time() - stop_start_time:.2f}s")
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
                    print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower stopped")
            except Exception:
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFailed to stop follower, is it running?")
            
            runner_active = False
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mExiting CatBot")
            break
        else:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mAborted quit")
            continue

GPIO.cleanup()
led_thread.join()
