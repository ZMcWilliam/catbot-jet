import time
import os
import sys
import signal
import subprocess
from git import Repo
from Jetson import GPIO
from colorama import init
from helpers import motorkit as m

init()

RUN_PIN = 21
ENABLE_GIT = False

GPIO.setmode(GPIO.BCM)
GPIO.setup(RUN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("\033[1;33m[RUNNER]\033[1;m \033[1;37mStarting CatBot...")

start_time = time.time()
stopCheck = 10

has_pushed = False
def pushToGit():
    print("\033[1;33m[RUNNER]\033[1;m \033[1;37mSaving to git...")
    try:
        full_local_path = "/home/pi/Desktop/CatBot"

        repo = Repo(full_local_path)
        repo.git.add(all=True)

        diff_list = repo.head.commit.diff()
        if len(diff_list) > 0:
            for diff in diff_list:
                print(f"\033[1;33m[RUNNER]\033[1;m \033[1;37m   {diff.change_type} -> {diff.new_file}")
            repo.index.commit("CatBot Auto Save")
            repo.git.push("origin", "dev")
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mSaved to git successfully")
        else:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mNo changes to save to git")
    except Exception as e:
        print("\033[1;33m[RUNNER]\033[1;m \033[1;31mFailed to save to git\033[1;37m")
        print(e)

if GPIO.input(RUN_PIN):
    print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower is disabled, waiting to start...")
p = None
state = 0

while True:
    try:
        input_state = not GPIO.input(RUN_PIN)
        if p is not None:
            process_state = p.poll()
            if input_state and process_state is not None:
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mResurrecting Follower...")
                state = 0
        if input_state and state == 0:
            p = subprocess.Popen("/home/pi/Desktop/CatBot/start.sh", shell=True, preexec_fn=os.setsid)
            time.sleep(0.4)
            state = 1
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower started")
        if not input_state and state == 0:
            time.sleep(0.1)
        if not input_state and state == 1:
            stopCheck -= 1
            if stopCheck == 0:
                try: os.killpg(p.pid, signal.SIGTERM)
                except Exception: print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFailed to kill follower instance, ignoring...")
                time.sleep(0.4)
                state = 0
                stopCheck = 10
                m.stop_all() # Stop all motors
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
                if p is not None and state == 1:
                    os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFailed to stop follower, is it running?")
            m.stop_all() # Stop all motors
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mFollower stopped")
            sys.exit()
        else:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;37mAborted quit")
            continue
    if not has_pushed and time.time() - start_time > 10:
        if ENABLE_GIT:
            pushToGit()
        else:
            print("\033[1;33m[RUNNER]\033[1;m \033[1;31mGit is disabled, not saving\033[1;37m")
        has_pushed = True
