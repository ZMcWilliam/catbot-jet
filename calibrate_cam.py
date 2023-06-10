import time
import cv2
import json
import helper_camera
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk

cams = helper_camera.CameraController()
cams.start_stream(0)

# Load the calibration map from the JSON file
with open("calibration.json", "r") as json_file:
    calibration_data = json.load(json_file)
calibration_map = 255 / np.array(calibration_data["calibration_map_w"])

with open("config.json", "r") as json_file:
    config_data = json.load(json_file)

black_contour_threshold = 5000
config_values = {
    "black_line_threshold": config_data["black_line_threshold"],
    "green_turn_hsv_threshold": [np.array(bound) for bound in config_data["green_turn_hsv_threshold"]],
    "rescue_circle_conf": config_data["rescue_circle_conf"],
}

frames = 0
fpsTime = time.time()
fpsCurrent = 0

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
black_line_thresh_slider.set(config_values["black_line_threshold"])  # Set initial value
black_line_thresh_slider.pack()

black_line_thresh_value = tk.StringVar()
black_line_thresh_label = ttk.Label(frame, textvariable=black_line_thresh_value)
black_line_thresh_label.pack()

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

# Create sliders for rescue_circle_conf
rescue_circle_minDist_label = ttk.Label(frame, text="rescue_circle_minDist")
rescue_circle_minDist_label.pack()

rescue_circle_minDist_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
rescue_circle_minDist_slider.set(config_values["rescue_circle_conf"]["minDist"])  # Set initial value
rescue_circle_minDist_slider.pack()

rescue_circle_minDist_value = tk.StringVar()
rescue_circle_minDist_label = ttk.Label(frame, textvariable=rescue_circle_minDist_value)
rescue_circle_minDist_label.pack()

rescue_circle_param1_label = ttk.Label(frame, text="rescue_circle_param1")
rescue_circle_param1_label.pack()

rescue_circle_param1_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
rescue_circle_param1_slider.set(config_values["rescue_circle_conf"]["param1"])  # Set initial value
rescue_circle_param1_slider.pack()

rescue_circle_param1_value = tk.StringVar()
rescue_circle_param1_label = ttk.Label(frame, textvariable=rescue_circle_param1_value)
rescue_circle_param1_label.pack()

rescue_circle_param2_label = ttk.Label(frame, text="rescue_circle_param2")
rescue_circle_param2_label.pack()

rescue_circle_param2_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
rescue_circle_param2_slider.set(config_values["rescue_circle_conf"]["param2"])  # Set initial value
rescue_circle_param2_slider.pack()

rescue_circle_param2_value = tk.StringVar()
rescue_circle_param2_label = ttk.Label(frame, textvariable=rescue_circle_param2_value)
rescue_circle_param2_label.pack()

rescue_circle_minRadius_label = ttk.Label(frame, text="rescue_circle_minRadius")
rescue_circle_minRadius_label.pack()

rescue_circle_minRadius_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
rescue_circle_minRadius_slider.set(config_values["rescue_circle_conf"]["minRadius"])  # Set initial value
rescue_circle_minRadius_slider.pack()

rescue_circle_minRadius_value = tk.StringVar()
rescue_circle_minRadius_label = ttk.Label(frame, textvariable=rescue_circle_minRadius_value)
rescue_circle_minRadius_label.pack()

rescue_circle_maxRadius_label = ttk.Label(frame, text="rescue_circle_maxRadius")
rescue_circle_maxRadius_label.pack()

rescue_circle_maxRadius_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
rescue_circle_maxRadius_slider.set(config_values["rescue_circle_conf"]["maxRadius"])  # Set initial value
rescue_circle_maxRadius_slider.pack()

rescue_circle_maxRadius_value = tk.StringVar()
rescue_circle_maxRadius_label = ttk.Label(frame, textvariable=rescue_circle_maxRadius_value)
rescue_circle_maxRadius_label.pack()

# Function to handle slider value changes
def on_slider_change(event):
    global config_values
    config_values["black_line_threshold"] = int(black_line_thresh_slider.get())

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

    config_values["rescue_circle_conf"]["minDist"] = int(rescue_circle_minDist_slider.get())
    config_values["rescue_circle_conf"]["param1"] = int(rescue_circle_param1_slider.get())
    config_values["rescue_circle_conf"]["param2"] = int(rescue_circle_param2_slider.get())
    config_values["rescue_circle_conf"]["minRadius"] = int(rescue_circle_minRadius_slider.get())
    config_values["rescue_circle_conf"]["maxRadius"] = int(rescue_circle_maxRadius_slider.get())

    # Update the value labels
    black_line_thresh_value.set(str(int(black_line_thresh_slider.get())))
    green_h_min_value.set(str(int(green_h_min_slider.get())))
    green_s_min_value.set(str(int(green_s_min_slider.get())))
    green_v_min_value.set(str(int(green_v_min_slider.get())))
    green_h_max_value.set(str(int(green_h_max_slider.get())))
    green_s_max_value.set(str(int(green_s_max_slider.get())))
    green_v_max_value.set(str(int(green_v_max_slider.get())))

    rescue_circle_minDist_value.set(str(int(rescue_circle_minDist_slider.get())))
    rescue_circle_param1_value.set(str(int(rescue_circle_param1_slider.get())))
    rescue_circle_param2_value.set(str(int(rescue_circle_param2_slider.get())))
    rescue_circle_minRadius_value.set(str(int(rescue_circle_minRadius_slider.get())))
    rescue_circle_maxRadius_value.set(str(int(rescue_circle_maxRadius_slider.get())))
    

# Bind the slider event to the on_slider_change function
black_line_thresh_slider.bind("<B1-Motion>", on_slider_change)
green_h_min_slider.bind("<B1-Motion>", on_slider_change)
green_s_min_slider.bind("<B1-Motion>", on_slider_change)
green_v_min_slider.bind("<B1-Motion>", on_slider_change)
green_h_max_slider.bind("<B1-Motion>", on_slider_change)
green_s_max_slider.bind("<B1-Motion>", on_slider_change)
green_v_max_slider.bind("<B1-Motion>", on_slider_change)

rescue_circle_minDist_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_param1_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_param2_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_minRadius_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_maxRadius_slider.bind("<B1-Motion>", on_slider_change)

# MAIN LOOP
def main_program():
    while True:
        if frames % 20 == 0 and frames != 0:
            fpsCurrent = int(20/(time.time()-fpsTime))
            fpsTime = time.time()
            print(f"Processing FPS: {fpsCurrent} | Camera FPS: {cams.get_fps(0)}")

        changed_black_contour = False
        img0 = cams.read_stream(0)
        # cv2.imwrite("testImg.jpg", img0)
        if (img0 is None):
            continue
        img0 = img0.copy()

        img0 = img0[0:img0.shape[0]-38, 0:img0.shape[1]]
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        img0_clean = img0.copy() # Used for displaying the image without any overlays

        # #Find the black in the image
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        # img0_gray = cv2.equalizeHist(img0_gray)
        img0_gray = cv2.GaussianBlur(img0_gray, (5, 5), 0)

        print(config_values["black_line_threshold"])
        img0_gray_scaled = calibration_map * img0_gray

        img0_binary = (img0_gray_scaled > config_values["black_line_threshold"]).astype(np.uint8)
        img0_binary = cv2.morphologyEx(img0_binary, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

        img0_hsv = cv2.cvtColor(img0, cv2.COLOR_RGB2HSV)

        #Find the green in the image
        img0_green = cv2.bitwise_not(cv2.inRange(img0_hsv, config_values["green_turn_hsv_threshold"][0], config_values["green_turn_hsv_threshold"][1]))
        img0_green = cv2.erode(img0_green, np.ones((5,5),np.uint8), iterations=1)

        # #Remove the green from the black (since green looks like black when grayscaled)
        img0_line = cv2.dilate(img0_binary, np.ones((5,5),np.uint8), iterations=2)
        img0_line = cv2.bitwise_or(img0_binary, cv2.bitwise_not(img0_green))

        # -----------
        circles = cv2.HoughCircles(img0_gray, cv2.HOUGH_GRADIENT, **config_values["rescue_circle_conf"])

        img0_circles = img0_clean.copy()
        if circles is not None:

            # Round the circle parameters to integers
            detected_circles = np.round(circles[0, :]).astype(int)
            detected_circles = sorted(detected_circles , key = lambda v: [v[1], v[1]],reverse=True)
            
            # Draw the detected circles on the original image
            for (x, y, r) in detected_circles:
                cv2.circle(img0_circles, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img0_circles, (x, y), 2, (0, 0, 255), 3)
        else:
            cv2.putText(img0_circles, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        preview_image_img0 = cv2.resize(img0, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0", preview_image_img0)

        # preview_image_img0_binary = cv2.resize(img0_binary, (0,0), fx=0.8, fy=0.7)
        # cv2.imshow("img0_binary", preview_image_img0_binary)

        # preview_image_img0_line = cv2.resize(img0_line, (0,0), fx=0.8, fy=0.7)
        # cv2.imshow("img0_line", preview_image_img0_line)

        # preview_image_img0_green = cv2.resize(img0_green, (0,0), fx=0.8, fy=0.7)
        # cv2.imshow("img0_green", preview_image_img0_green)

        preview_image_img0_circles = cv2.resize(img0_circles, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_circles", preview_image_img0_circles)

        # def mouseCallbackHSV(event, x, y, flags, param):
        #     if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        #         # Print HSV value only when the left mouse button is pressed and mouse is moving
        #         hsv_value = img0_hsv[y, x]
        #         print(f"HSV: {hsv_value}")
        # # Show HSV preview with text on hover to show HSV values
        # preview_image_img0_hsv = cv2.resize(img0_hsv, (0,0), fx=0.8, fy=0.7)
        # cv2.imshow("img0_hsv", preview_image_img0_hsv)
        # cv2.setMouseCallback("img0_hsv", mouseCallbackHSV)

        # preview_image_img0_gray_scaled = cv2.resize(img0_gray_scaled, (0,0), fx=0.8, fy=0.7)
        # cv2.imshow("img0_gray_scaled", preview_image_img0_gray_scaled)

        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            break

    cams.stop()

# Start the main program logic in a new thread
program_thread = threading.Thread(target=main_program)
program_thread.daemon = True
program_thread.start()

# Start the Tkinter UI main loop
root.mainloop()