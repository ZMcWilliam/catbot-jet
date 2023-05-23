import time
import cv2
import json
import math
import helper_camera
import helper_motorkit as m
import numpy as np
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
    "black_line_threshold": [180, 255],
    "green_turn_hsv_threshold": [
        np.array([43, 93, 90]),
        np.array([84, 234, 229]),
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
MOTOR_ENABLE = False # For debugging
max_motor_speed = 100

greenCenter = None



def distance(point):
    if (point[0][0] > last_line_pos[0]):
        return np.linalg.norm(np.array(point[0]) - last_line_pos)
    else:
        return np.linalg.norm(last_line_pos - point[0])
distanceFormula = np.vectorize(distance)
def FindBestContours(contours):
    contour_values = np.array([[cv2.contourArea(contour), cv2.minAreaRect(contour), contour, 0] for contour in contours ], dtype=object)
    if len(contour_values) == 0:
        return []
    contour_values = contour_values[contour_values[:, 0] > black_contour_threshold]
    if (len(contour_values) < 2):
        return contour_values
    contour_values[:, 3] = distanceFormula(contour_values[:, 1])
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
    # img0_binary = cv2.dilate(img0_binary, np.ones((5,5),np.uint8), iterations=2)
    # img0_binary = cv2.bitwise_or(img0_binary, cv2.bitwise_not(img0_green))


    preview_image_img0 = cv2.resize(img0, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0", preview_image_img0)

    preview_image_img0_binary = cv2.resize(img0_binary, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_binary", preview_image_img0_binary)

    preview_image_img0_green = cv2.resize(img0_green, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_green", preview_image_img0_green)

    preview_image_img0_gray = cv2.resize(img0_gray, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_gray", preview_image_img0_gray)

    def mouseCallbackHSV(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # Print HSV value only when the left mouse button is pressed and mouse is moving
            hsv_value = img0_hsv[y, x]
            print(f"HSV: {hsv_value}")
    # Show HSV preview with text on hover to show HSV values
    preview_image_img0_hsv = cv2.resize(img0_hsv, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_hsv", preview_image_img0_hsv)
    cv2.setMouseCallback("img0_hsv", mouseCallbackHSV)


    preview_image_img0_gray_scaled = cv2.resize(img0_gray_scaled, (0,0), fx=0.8, fy=0.7)
    cv2.imshow("img0_gray_scaled", preview_image_img0_gray_scaled)

    # frames += 1

    k = cv2.waitKey(1)
    if (k & 0xFF == ord('q')):
        # pr.print_stats(SortKey.TIME)
        program_active = False
        break

cams.stop()