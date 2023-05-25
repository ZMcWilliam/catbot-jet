import time
import cv2
import json
import math
import helper_camera
import helper_motorkit as m
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from gpiozero import AngularServo

PORT_SERVO_GATE = 12
PORT_SERVO_CLAW = 13
PORT_SERVO_LIFT = 18
PORT_SERVO_CAM = 19

servo = {
    "gate": AngularServo(PORT_SERVO_GATE, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-90),    # -90=Close, 90=Open
    "claw": AngularServo(PORT_SERVO_CLAW, min_pulse_width=0.0005, max_pulse_width=0.002, initial_angle=-80),    # 0=Open, -90=Close
    "lift": AngularServo(PORT_SERVO_LIFT, min_pulse_width=0.0005, max_pulse_width=0.0025, initial_angle=-80),   # -90=Up, 40=Down
    "cam": AngularServo(PORT_SERVO_CAM, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-80)       # -90=Down, 90=Up
}

cams = helper_camera.CameraController()
cams.start_stream(0)

#System variables
changed_angle = False
last_line_pos = np.array([100,100])
last_ang = 0
current_linefollowing_state = None
white_intersection_cooldown = 0
changed_black_contour = False

# max_error_and_angle = 285 + 90
max_error = 285
max_angle = 90
error_weight = 0.5
angle_weight = 1-error_weight

#Configs

# Load the calibration map from the JSON file
with open("calibration.json", "r") as json_file:
    calibration_data = json.load(json_file)
calibration_map = np.array(calibration_data["calibration_map"])

# Camera stuff
black_contour_threshold = 5000
config_values = {
    "black_line_threshold": [178, 255],
    "green_turn_hsv_threshold": [
        np.array([26, 31, 81]),
        np.array([69, 234, 229]),
    ],
}

# Constants for PID control
KP = 0.07 # Proportional gain
KI = 0  # Integral gain
KD = 0.1  # Derivative gain
follower_speed = 40

lastError = 0
integral = 0
# Motor stuff
max_motor_speed = 100

greenCenter = None


# Jank functions

# Calculate the distance between a point, and the last line position
def distToLastLine(point):
    if (point[0][0] > last_line_pos[0]):
        return np.linalg.norm(np.array(point[0]) - last_line_pos)
    else:
        return np.linalg.norm(last_line_pos - point[0])
    
# Vectorize the distance function so it can be applied to a numpy array
# This helps speed up calculations when calculating the distance of many points
distToLastLineFormula = np.vectorize(distToLastLine)

# Processes a set of contours to find the best one to follow
# Filters out contours that are too small, 
# then, sorts the remaining contours by distance from the last line position
def FindBestContours(contours):
    # Create a new array with the contour area, contour, and distance from the last line position (to be calculated later)
    contour_values = np.array([[cv2.contourArea(contour), cv2.minAreaRect(contour), contour, 0] for contour in contours ], dtype=object)

    # In case we have no contours, just return an empty array instead of processing any more
    if len(contour_values) == 0:
        return []
    
    # Filter out contours that are too small
    contour_values = contour_values[contour_values[:, 0] > black_contour_threshold]
    
    # No need to sort if there is only one contour
    if len(contour_values) <= 1:
        return contour_values

    # Sort contours by distance from the last known optimal line position
    contour_values[:, 3] = distToLastLineFormula(contour_values[:, 1])
    contour_values = contour_values[np.argsort(contour_values[:, 3])]
    return contour_values

current_time = time.time()

def pid(error): # Calculate error beforehand
    global current_time, integral, lastError
    timeDiff = time.time() - current_time
    if (timeDiff == 0):
        timeDiff = 1/10
    proportional = KP*(error)
    integral += KI*error*timeDiff
    derivative = KD*(error-lastError)/timeDiff
    PIDOutput = -(proportional + integral + derivative)
    lastError = error
    current_time = time.time()
    return PIDOutput

def centerOfContour(contour):
    center = cv2.boundingRect(contour)
    return (int(center[0]+(center[2]/2)), int(center[1]+(center[3]/3)))
def pointDistance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)


frames = 0
# check_thing = False
start_time = time.time()
# last_green_found_time = start_time - 1000
last_intersection_time = time.time() - 100
fpsTime = time.time()
delay = time.time()

double_check = 0
gzDetected = False


def simplifiedContourPoints(contour, epsilon=0.01):
    epsilonBL = epsilon * cv2.arcLength(contour, True)
    return [pt[0] for pt in cv2.approxPolyDP(contour, epsilonBL, True)]

smallKernel = np.ones((5,5),np.uint8)


# TKINTER STUFF

# Create the main application window
root = tk.Tk()
root.title("Configuration Settings")

# Create a frame to hold the sliders
frame = ttk.Frame(root, padding="20")
frame.pack()

# Create sliders for black_line_threshold
black_line_label = ttk.Label(frame, text="black_line_threshold")
black_line_label.pack()

black_line_thresh_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
black_line_thresh_slider.set(config_values["black_line_threshold"][0])  # Set initial value
black_line_thresh_slider.pack()

black_line_thresh_value = tk.StringVar()
black_line_thresh_label = ttk.Label(frame, textvariable=black_line_thresh_value)
black_line_thresh_label.pack()

black_line_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
black_line_max_slider.set(config_values["black_line_threshold"][1])  # Set initial value
black_line_max_slider.pack()

black_line_max_value = tk.StringVar()
black_line_max_label = ttk.Label(frame, textvariable=black_line_max_value)
black_line_max_label.pack()

# Create sliders for green_turn_hsv_threshold
green_turn_label = ttk.Label(frame, text="green_turn_hsv_threshold")
green_turn_label.pack()

green_h_min_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
green_h_min_slider.set(config_values["green_turn_hsv_threshold"][0][0])  # Set initial value
green_h_min_slider.pack()

green_h_min_value = tk.StringVar()
green_h_min_label = ttk.Label(frame, textvariable=green_h_min_value)
green_h_min_label.pack()

green_s_min_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
green_s_min_slider.set(config_values["green_turn_hsv_threshold"][0][1])  # Set initial value
green_s_min_slider.pack()

green_s_min_value = tk.StringVar()
green_s_min_label = ttk.Label(frame, textvariable=green_s_min_value)
green_s_min_label.pack()

green_v_min_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
green_v_min_slider.set(config_values["green_turn_hsv_threshold"][0][2])  # Set initial value
green_v_min_slider.pack()

green_v_min_value = tk.StringVar()
green_v_min_label = ttk.Label(frame, textvariable=green_v_min_value)
green_v_min_label.pack()

green_h_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
green_h_max_slider.set(config_values["green_turn_hsv_threshold"][1][0])  # Set initial value
green_h_max_slider.pack()

green_h_max_value = tk.StringVar()
green_h_max_label = ttk.Label(frame, textvariable=green_h_max_value)
green_h_max_label.pack()

green_s_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
green_s_max_slider.set(config_values["green_turn_hsv_threshold"][1][1])  # Set initial value
green_s_max_slider.pack()

green_s_max_value = tk.StringVar()
green_s_max_label = ttk.Label(frame, textvariable=green_s_max_value)
green_s_max_label.pack()

green_v_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
green_v_max_slider.set(config_values["green_turn_hsv_threshold"][1][2])  # Set initial value
green_v_max_slider.pack()

green_v_max_value = tk.StringVar()
green_v_max_label = ttk.Label(frame, textvariable=green_v_max_value)
green_v_max_label.pack()

# Function to handle slider value changes
def on_slider_change(event):
    config_values["black_line_threshold"] = [int(black_line_thresh_slider.get()), int(black_line_max_slider.get())]

    green_h_min = int(green_h_min_slider.get())
    green_s_min = int(green_s_min_slider.get())
    green_v_min = int(green_v_min_slider.get())
    green_h_max = int(green_h_max_slider.get())
    green_s_max = int(green_s_max_slider.get())
    green_v_max = int(green_v_max_slider.get())

    # Convert the green threshold values to numpy arrays
    green_turn_hsv_threshold = [
        np.array([green_h_min, green_s_min, green_v_min]),
        np.array([green_h_max, green_s_max, green_v_max])
    ]

    config_values["green_turn_hsv_threshold"] = green_turn_hsv_threshold

    # Update the value labels
    black_line_thresh_value.set(str(int(black_line_thresh_slider.get())))
    black_line_max_value.set(str(int(black_line_max_slider.get())))
    green_h_min_value.set(str(int(green_h_min_slider.get())))
    green_s_min_value.set(str(int(green_s_min_slider.get())))
    green_v_min_value.set(str(int(green_v_min_slider.get())))
    green_h_max_value.set(str(int(green_h_max_slider.get())))
    green_s_max_value.set(str(int(green_s_max_slider.get())))
    green_v_max_value.set(str(int(green_v_max_slider.get())))

# Bind the slider event to the on_slider_change function
black_line_thresh_slider.bind("<ButtonRelease-1>", on_slider_change)
black_line_max_slider.bind("<ButtonRelease-1>", on_slider_change)
green_h_min_slider.bind("<ButtonRelease-1>", on_slider_change)
green_s_min_slider.bind("<ButtonRelease-1>", on_slider_change)
green_v_min_slider.bind("<ButtonRelease-1>", on_slider_change)
green_h_max_slider.bind("<ButtonRelease-1>", on_slider_change)
green_s_max_slider.bind("<ButtonRelease-1>", on_slider_change)
green_v_max_slider.bind("<ButtonRelease-1>", on_slider_change)
    
# MAIN LOOP
# def main_program():
while True:
    changed_black_contour = False
    if frames % 20 == 0 and frames != 0:
        print(f"Processing FPS: {20/(time.time()-fpsTime)}")
        fpsTime = time.time()
    # if frames % 100 == 0:
    #     print(f"Camera 0 average FPS: {cams.get_fps(0)}")
    img0 = cams.read_stream(0)
    # cv2.imwrite("testImg.jpg", img0)
    if (img0 is None):
        continue
    img0 = img0.copy()

    img0 = img0[0:img0.shape[0]-38, 0:img0.shape[1]-70]
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # #Find the black in the image
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    # img0_gray = cv2.equalizeHist(img0_gray)
    img0_gray = cv2.GaussianBlur(img0_gray, (5, 5), 0)

    img0_gray_scaled = 255 / np.clip(calibration_map, a_min=1, a_max=None) * img0_gray  # Scale white values based on the inverse of the calibration map
    img0_gray_scaled = np.clip(img0_gray_scaled, 0, 255)    # Clip the scaled image to ensure values are within the valid range
    img0_gray_scaled = img0_gray_scaled.astype(np.uint8)    # Convert the scaled image back to uint8 data type

    img0_binary = cv2.threshold(img0_gray_scaled, config_values["black_line_threshold"][0], config_values["black_line_threshold"][1], cv2.THRESH_BINARY)[1]
    img0_binary = cv2.morphologyEx(img0_binary, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

    img0_hsv = cv2.cvtColor(img0, cv2.COLOR_RGB2HSV)

    #Find the green in the image
    img0_green = cv2.bitwise_not(cv2.inRange(img0_hsv, config_values["green_turn_hsv_threshold"][0], config_values["green_turn_hsv_threshold"][1]))
    img0_green = cv2.erode(img0_green, np.ones((5,5),np.uint8), iterations=1)

    # #Remove the green from the black (since green looks like black when grayscaled)
    img0_line = cv2.dilate(img0_binary, np.ones((5,5),np.uint8), iterations=2)
    img0_line = cv2.bitwise_or(img0_binary, cv2.bitwise_not(img0_green))

    # -----------

    raw_white_contours, white_hierarchy = cv2.findContours(img0_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter white contours based on area
    white_contours = []
    for contour in raw_white_contours:
        if (cv2.contourArea(contour) > 1000):
            white_contours.append(contour)
    img0_binary_not = cv2.bitwise_not(img0_binary)
    
    # Find black contours
    # If there are no black contours, skip the rest of the loop
    black_contours, black_hierarchy = cv2.findContours(img0_binary_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(black_contours) == 0):
        print("No black contours found")
        continue
    
    # -----------
    # GREEN TURNS
    # -----------

    is_there_green = np.count_nonzero(img0_green == 0)
    turning = False
    black_contours_turn = None

    # print("Green: ", is_there_green)
    
    # Check if there is a significant amount of green pixels
    if is_there_green > 2000: #and len(white_contours) > 2: #((is_there_green > 1000 or time.time() - last_green_found_time < 0.5) and (len(white_contours) > 2 or greenCenter is not None)):
        unfiltered_green_contours, green_hierarchy = cv2.findContours(cv2.bitwise_not(img0_green), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img0, green_contours[0], -1, (255,255,0), 3)

        # TODO

        print("GREEN TURN STUFF")
    else:
        greenCenter = None # Reset greenturn memory if no green found!


    # -----------
    # INTERSECTIONS
    # -----------

    # TODO

    # -----------
    # REST OF LINE LINE FOLLOWER
    # -----------

    #Find the black contours
    sorted_black_contours = FindBestContours(black_contours)
    if (len(sorted_black_contours) == 0):
        print("No black contours found")

        print("STEER TEMP: GO FORWARD")
        continue
    chosen_black_contour = sorted_black_contours[0]

    cv2.drawContours(img0, [chosen_black_contour[2]], -1, (0,255,0), 3) # DEBUG
    
    # Update the reference position for subsequent calculations
    last_line_pos = np.array([chosen_black_contour[1][0][0], chosen_black_contour[1][0][1]])

    preview_image_img0 = cv2.resize(img0, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0", preview_image_img0)

    preview_image_img0_binary = cv2.resize(img0_binary, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_binary", preview_image_img0_binary)

    preview_image_img0_line = cv2.resize(img0_line, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_line", preview_image_img0_line)

    preview_image_img0_green = cv2.resize(img0_green, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_green", preview_image_img0_green)

    # preview_image_img0_gray = cv2.resize(img0_gray, (0,0), fx=0.8, fy=0.7)
    # cv2.imshow("img0_gray", preview_image_img0_gray)

    # def mouseCallbackHSV(event, x, y, flags, param):
    #     if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
    #         # Print HSV value only when the left mouse button is pressed and mouse is moving
    #         hsv_value = img0_hsv[y, x]
    #         print(f"HSV: {hsv_value}")
    # # Show HSV preview with text on hover to show HSV values
    # preview_image_img0_hsv = cv2.resize(img0_hsv, (0,0), fx=0.8, fy=0.7)
    # cv2.imshow("img0_hsv", preview_image_img0_hsv)
    # cv2.setMouseCallback("img0_hsv", mouseCallbackHSV)

    preview_image_img0_gray_scaled = cv2.resize(img0_gray_scaled, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_gray_scaled", preview_image_img0_gray_scaled)

    # Show a preview of the image with the contours drawn on it, black as red and white as blue

    preview_image_img0_contours = img0_clean.copy()
    cv2.drawContours(preview_image_img0_contours, white_contours, -1, (255,0,0), 3)
    cv2.drawContours(preview_image_img0_contours, black_contours, -1, (0,0,255), 3)
    preview_image_img0_contours = cv2.resize(preview_image_img0_contours, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_contours", preview_image_img0_contours)

    # frames += 1

    k = cv2.waitKey(1)
    if (k & 0xFF == ord('q')):
        # pr.print_stats(SortKey.TIME)
        program_active = False
        break

cams.stop()

# Start the main program logic in a new thread
# program_thread = threading.Thread(target=main_program)
# program_thread.daemon = True
# program_thread.start()

# Start the Tkinter UI main loop
# root.mainloop()