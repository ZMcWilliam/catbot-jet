import time
import cv2
import json
import helper_camera
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk


# Load the calibration map from the JSON file
with open("calibration.json", "r") as json_file:
    calibration_data = json.load(json_file)
calibration_map = 255 / np.array(calibration_data["calibration_map_w"])
calibration_map_rescue = 255 / np.array(calibration_data["calibration_map_rescue_w"])

with open("config.json", "r") as json_file:
    config_data = json.load(json_file)

black_contour_threshold = 5000
config_values = {
    "black_line_threshold": config_data["black_line_threshold"],
    "black_rescue_threshold": config_data["black_rescue_threshold"],
    "green_turn_hsv_threshold": [np.array(bound) for bound in config_data["green_turn_hsv_threshold"]],
    "red_hsv_threshold": [np.array(bound) for bound in config_data["red_hsv_threshold"]],
    "rescue_circle_conf": config_data["rescue_circle_conf"],
}

cam = helper_camera.CameraStream(
    camera_num = 0, 
    processing_conf = {
        "calibration_map": calibration_map,
        "black_line_threshold": config_values["black_line_threshold"],
        "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
        "red_hsv_threshold": config_values["red_hsv_threshold"],
    }
)
cam.start_stream()

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

black_rescue_label = ttk.Label(frame, text="black_rescue_threshold")
black_rescue_label.pack()

black_rescue_thresh_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
black_rescue_thresh_slider.set(config_values["black_rescue_threshold"])  # Set initial value
black_rescue_thresh_slider.pack()

black_rescue_thresh_value = tk.StringVar()
black_rescue_thresh_label = ttk.Label(frame, textvariable=black_rescue_thresh_value)
black_rescue_thresh_label.pack()

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


# Create sliders for red_hsv_threshold
red_label = ttk.Label(frame, text="red_hsv_threshold")
red_label.pack()

red_h_min_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
red_h_min_slider.set(config_values["red_hsv_threshold"][0][0])  # Set initial value
red_h_min_slider.pack()

red_h_min_value = tk.StringVar()
red_h_min_label = ttk.Label(frame, textvariable=red_h_min_value)
red_h_min_label.pack()

red_s_min_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
red_s_min_slider.set(config_values["red_hsv_threshold"][0][1])  # Set initial value
red_s_min_slider.pack()

red_s_min_value = tk.StringVar()
red_s_min_label = ttk.Label(frame, textvariable=red_s_min_value)
red_s_min_label.pack()

red_v_min_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
red_v_min_slider.set(config_values["red_hsv_threshold"][0][2])  # Set initial value
red_v_min_slider.pack()

red_v_min_value = tk.StringVar()
red_v_min_label = ttk.Label(frame, textvariable=red_v_min_value)
red_v_min_label.pack()

red_h_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
red_h_max_slider.set(config_values["red_hsv_threshold"][1][0])  # Set initial value
red_h_max_slider.pack()

red_h_max_value = tk.StringVar()
red_h_max_label = ttk.Label(frame, textvariable=red_h_max_value)
red_h_max_label.pack()

red_s_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
red_s_max_slider.set(config_values["red_hsv_threshold"][1][1])  # Set initial value
red_s_max_slider.pack()

red_s_max_value = tk.StringVar()
red_s_max_label = ttk.Label(frame, textvariable=red_s_max_value)
red_s_max_label.pack()

red_v_max_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
red_v_max_slider.set(config_values["red_hsv_threshold"][1][2])  # Set initial value
red_v_max_slider.pack()

red_v_max_value = tk.StringVar()
red_v_max_label = ttk.Label(frame, textvariable=red_v_max_value)
red_v_max_label.pack()

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

rescue_circle_heightBuffer = ttk.Label(frame, text="rescue_circle_heightBuffer")
rescue_circle_heightBuffer.pack()

rescue_circle_heightBuffer_slider = ttk.Scale(frame, from_=0, to=400, orient="horizontal", length=400)
rescue_circle_heightBuffer_slider.set(config_values["rescue_circle_conf"]["heightBuffer"])  # Set initial value
rescue_circle_heightBuffer_slider.pack()

rescue_circle_heightBuffer_value = tk.StringVar()
rescue_circle_heightBuffer_label = ttk.Label(frame, textvariable=rescue_circle_heightBuffer_value)
rescue_circle_heightBuffer_label.pack()

rescue_circle_lowHeightMinRadius = ttk.Label(frame, text="rescue_circle_lowHeightMinRadius")
rescue_circle_lowHeightMinRadius.pack()

rescue_circle_lowHeightMinRadius_slider = ttk.Scale(frame, from_=0, to=255, orient="horizontal", length=400)
rescue_circle_lowHeightMinRadius_slider.set(config_values["rescue_circle_conf"]["lowHeightMinRadius"])  # Set initial value
rescue_circle_lowHeightMinRadius_slider.pack()

rescue_circle_lowHeightMinRadius_value = tk.StringVar()
rescue_circle_lowHeightMinRadius_label = ttk.Label(frame, textvariable=rescue_circle_lowHeightMinRadius_value)
rescue_circle_lowHeightMinRadius_label.pack()

# Function to handle slider value changes
def on_slider_change(event):
    global config_values
    config_values["black_line_threshold"] = int(black_line_thresh_slider.get())
    config_values["black_rescue_threshold"] = int(black_rescue_thresh_slider.get())

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

    red_h_min = int(red_h_min_slider.get())
    red_s_min = int(red_s_min_slider.get())
    red_v_min = int(red_v_min_slider.get())
    red_h_max = int(red_h_max_slider.get())
    red_s_max = int(red_s_max_slider.get())
    red_v_max = int(red_v_max_slider.get())

    # Convert the red threshold values to numpy arrays
    red_hsv_threshold = [
        np.array([red_h_min, red_s_min, red_v_min]),
        np.array([red_h_max, red_s_max, red_v_max])
    ]

    config_values["red_hsv_threshold"] = red_hsv_threshold

    config_values["rescue_circle_conf"]["minDist"] = int(rescue_circle_minDist_slider.get())
    config_values["rescue_circle_conf"]["param1"] = int(rescue_circle_param1_slider.get())
    config_values["rescue_circle_conf"]["param2"] = int(rescue_circle_param2_slider.get())
    config_values["rescue_circle_conf"]["minRadius"] = int(rescue_circle_minRadius_slider.get())
    config_values["rescue_circle_conf"]["maxRadius"] = int(rescue_circle_maxRadius_slider.get())
    config_values["rescue_circle_conf"]["heightBuffer"] = int(rescue_circle_heightBuffer_slider.get())
    config_values["rescue_circle_conf"]["lowHeightMinRadius"] = int(rescue_circle_lowHeightMinRadius_slider.get())

    # Update the value labels
    black_line_thresh_value.set(str(int(black_line_thresh_slider.get())))
    black_rescue_thresh_value.set(str(int(black_rescue_thresh_slider.get())))
    green_h_min_value.set(str(int(green_h_min_slider.get())))
    green_s_min_value.set(str(int(green_s_min_slider.get())))
    green_v_min_value.set(str(int(green_v_min_slider.get())))
    green_h_max_value.set(str(int(green_h_max_slider.get())))
    green_s_max_value.set(str(int(green_s_max_slider.get())))
    green_v_max_value.set(str(int(green_v_max_slider.get())))

    red_h_min_value.set(str(int(red_h_min_slider.get())))
    red_s_min_value.set(str(int(red_s_min_slider.get())))
    red_v_min_value.set(str(int(red_v_min_slider.get())))
    red_h_max_value.set(str(int(red_h_max_slider.get())))
    red_s_max_value.set(str(int(red_s_max_slider.get())))
    red_v_max_value.set(str(int(red_v_max_slider.get())))

    rescue_circle_minDist_value.set(str(int(rescue_circle_minDist_slider.get())))
    rescue_circle_param1_value.set(str(int(rescue_circle_param1_slider.get())))
    rescue_circle_param2_value.set(str(int(rescue_circle_param2_slider.get())))
    rescue_circle_minRadius_value.set(str(int(rescue_circle_minRadius_slider.get())))
    rescue_circle_maxRadius_value.set(str(int(rescue_circle_maxRadius_slider.get())))
    rescue_circle_heightBuffer_value.set(str(int(rescue_circle_heightBuffer_slider.get())))
    rescue_circle_lowHeightMinRadius_value.set(str(int(rescue_circle_lowHeightMinRadius_slider.get())))
    

# Bind the slider event to the on_slider_change function
black_line_thresh_slider.bind("<B1-Motion>", on_slider_change)
black_rescue_thresh_slider.bind("<B1-Motion>", on_slider_change)
green_h_min_slider.bind("<B1-Motion>", on_slider_change)
green_s_min_slider.bind("<B1-Motion>", on_slider_change)
green_v_min_slider.bind("<B1-Motion>", on_slider_change)
green_h_max_slider.bind("<B1-Motion>", on_slider_change)
green_s_max_slider.bind("<B1-Motion>", on_slider_change)
green_v_max_slider.bind("<B1-Motion>", on_slider_change)

red_h_min_slider.bind("<B1-Motion>", on_slider_change)
red_s_min_slider.bind("<B1-Motion>", on_slider_change)
red_v_min_slider.bind("<B1-Motion>", on_slider_change)
red_h_max_slider.bind("<B1-Motion>", on_slider_change)
red_s_max_slider.bind("<B1-Motion>", on_slider_change)
red_v_max_slider.bind("<B1-Motion>", on_slider_change)

rescue_circle_minDist_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_param1_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_param2_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_minRadius_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_maxRadius_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_heightBuffer_slider.bind("<B1-Motion>", on_slider_change)
rescue_circle_lowHeightMinRadius_slider.bind("<B1-Motion>", on_slider_change)

# MAIN LOOP
def main_program():
    while True:
        if frames % 20 == 0 and frames != 0:
            fpsCurrent = int(20/(time.time()-fpsTime))
            fpsTime = time.time()
            print(f"Processing FPS: {fpsCurrent} | Camera FPS: {cam.get_fps()}")
        
        cam.set_processing_conf({
            "calibration_map": calibration_map,
            "black_line_threshold": config_values["black_line_threshold"],
            "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
            "red_hsv_threshold": config_values["red_hsv_threshold"],
        })

        frame_processed = cam.read_stream_processed()
        if (frame_processed is None or frame_processed["resized"] is None):
            print("Waiting for image...")
            continue

        img0 = frame_processed["resized"].copy()
        img0_clean = img0.copy() # Used for displaying the image without any overlays

        img0_gray = frame_processed["gray"].copy()
        # img0_gray_scaled = frame_processed["gray_scaled"].copy()
        img0_binary = frame_processed["binary"].copy()
        img0_hsv = frame_processed["hsv"].copy()
        img0_green = frame_processed["green"].copy()
        img0_line = frame_processed["line"].copy()

        img0_red = cv2.bitwise_not(cv2.inRange(img0_hsv, config_values["red_hsv_threshold"][0], config_values["red_hsv_threshold"][1]))
        img0_red = cv2.dilate(img0_red, np.ones((5,5),np.uint8), iterations=2)

        img0_binary_rescue = ((calibration_map_rescue * img0_gray > config_values["black_rescue_threshold"]) * 255).astype(np.uint8)
        img0_binary_rescue = cv2.morphologyEx(img0_binary_rescue, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))

        # Only areas of img0_binary_rescue that are also in img0_red are kept (temp using red as mask for now, need to reduce the mess of this script)
        img0_binary_rescue_block = cv2.bitwise_and(cv2.bitwise_not(img0_binary_rescue), cv2.bitwise_not(img0_red))

        img0_gray_rescue_scaled = img0_gray * config_values["rescue_circle_conf"]["grayScaleMultiplier"]
        img0_gray_rescue_scaled = np.clip(img0_gray_rescue_scaled, 0, 255).astype(np.uint8)

        img0_blurred = cv2.medianBlur(img0_gray_rescue_scaled, 9)
        circles = cv2.HoughCircles(img0_blurred, cv2.HOUGH_GRADIENT, **{
            "dp": config_values["rescue_circle_conf"]["dp"],
            "minDist": config_values["rescue_circle_conf"]["minDist"],
            "param1": config_values["rescue_circle_conf"]["param1"],
            "param2": config_values["rescue_circle_conf"]["param2"],
            "minRadius": config_values["rescue_circle_conf"]["minRadius"],
            "maxRadius": config_values["rescue_circle_conf"]["maxRadius"],
        })

        img0_circles = img0_clean.copy()
        if circles is not None:

            # Round the circle parameters to integers
            detected_circles = np.round(circles[0, :]).astype(int)
            detected_circles = sorted(detected_circles , key = lambda v: [v[1], v[1]],reverse=True)

            # If the circle is in the bottom half of the image, check that the radius is greater than 40
            valid_circles = []
            for (x, y, r) in detected_circles:
                if y > config_values["rescue_circle_conf"]["heightBuffer"] and r > config_values["rescue_circle_conf"]["lowHeightMinRadius"]:
                    valid_circles.append([x, y, r])
                elif y <= config_values["rescue_circle_conf"]["heightBuffer"]:
                    valid_circles.append([x, y, r])

            # Draw the height buffer
            cv2.line(img0_circles, (0, config_values["rescue_circle_conf"]["heightBuffer"]), (img0_circles.shape[1], config_values["rescue_circle_conf"]["heightBuffer"]), (255, 0, 0), 2)

            # Draw the detected circles on the original image
            for (x, y, r) in detected_circles:
                cv2.circle(img0_circles, (x, y), r, (0, 0, 255), 2)
                cv2.circle(img0_circles, (x, y), 2, (0, 0, 255), 3)

            for i, (x, y, r) in enumerate(valid_circles):
                cv2.circle(img0_circles, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img0_circles, (x, y), 2, (0, 255, 0), 3)
                cv2.putText(img0, "#{}".format(i), (x , y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 255, 255), 2)
                
        else:
            cv2.putText(img0_circles, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        preview_image_img0 = cv2.resize(img0, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0", preview_image_img0)

        # preview_image_img0_blurred = cv2.resize(img0_blurred, (0,0), fx=0.8, fy=0.7)
        # cv2.imshow("img0_blurred", preview_image_img0_blurred)

        preview_image_img0_binary = cv2.resize(img0_binary, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_binary", preview_image_img0_binary)

        preview_image_img0_binary_rescue = cv2.resize(img0_binary_rescue, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_binary_rescue", preview_image_img0_binary_rescue)

        preview_image_img0_binary_rescue_block = cv2.resize(img0_binary_rescue_block, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_binary_rescue_block", preview_image_img0_binary_rescue_block)

        preview_image_img0_line = cv2.resize(img0_line, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_line", preview_image_img0_line)

        preview_image_img0_green = cv2.resize(img0_green, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_green", preview_image_img0_green)

        preview_image_img0_red = cv2.resize(img0_red, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_red", preview_image_img0_red)

        preview_image_img0_circles = cv2.resize(img0_circles, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_circles", preview_image_img0_circles)

        preview_image_img0_gray_rescue_scaled = cv2.resize(img0_gray_rescue_scaled, (0,0), fx=0.8, fy=0.7)
        cv2.imshow("img0_gray_rescue_scaled", preview_image_img0_gray_rescue_scaled)

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

    cam.stop()

# Start the main program logic in a new thread
program_thread = threading.Thread(target=main_program)
program_thread.daemon = True
program_thread.start()

# Start the Tkinter UI main loop
root.mainloop()