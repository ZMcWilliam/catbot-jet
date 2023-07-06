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

with open("config2.json", "r") as json_file:
    config_data = json.load(json_file)

gui_width = 400
btn_height = 30

btn_img = None
selected_tab = "main"

btn_width = gui_width // len(config_data)
btn_pad = 2
btn_txt_size = 0.4

btn_locations = {}


cam = None

def draw_btns():
    global selected_tab
    
    cv2.rectangle(btn_img, (0, 0), (gui_width, btn_height), (239, 239, 239), -1)
    for i, tab_id in enumerate(config_data):
        tab = config_data[tab_id]

        button_x = i * btn_width
        button_text = tab["title"]
        
        btn_locations[tab_id] = [[button_x + btn_pad, 0], [button_x + btn_width - (btn_pad * 2), btn_height]]
        cv2.rectangle(
            btn_img, (button_x + btn_pad, 0), 
            (button_x + btn_width - (btn_pad * 2), btn_height), 
            (150, 150, 150) if tab_id == selected_tab else (200, 200, 200), 
            -1
        )
        
        text_size, _ = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, btn_txt_size, 1)
        text_x = button_x + (btn_width - text_size[0] - (btn_pad * 2)) // 2
        text_y = (btn_height + text_size[1]) // 2
        cv2.putText(btn_img, button_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, btn_txt_size, (50, 50, 50), 1)

def btn_callback(event, x, y, flags, param):
    global selected_tab

    for tab_id in btn_locations:
        loc = btn_locations[tab_id]
        # Check if the x and y is within the button
        if loc[0][0] <= x and x <= loc[1][0] and loc[0][1] <= y and y <= loc[1][1]:
            if event == 1: # Mouse down
                print("Selected tab is now: " + selected_tab)
                show_selected_tab(tab_id)

def show_selected_tab(tab_id):
    global selected_tab
    global btn_img
    global cam

    selected_tab = tab_id

    cv2.destroyAllWindows()
    btn_img = np.zeros((btn_height, gui_width, 3), dtype=np.uint8)

    cv2.namedWindow("Config")
    
    draw_btns()
    cv2.imshow("Config", btn_img)
    cv2.setMouseCallback("Config", btn_callback)

    def create_callback(i_selected_tab, i_conf_id, i_data_id):
        def trackbar_callback(x):
            if config_data[i_selected_tab]["configs"][i_conf_id]["type"] == "float":
                x = x / 10
            config_data[i_selected_tab]["configs"][i_conf_id]["data"][i_data_id] = x

        return trackbar_callback

    for conf_id in config_data[selected_tab]["configs"]:
        conf = config_data[selected_tab]["configs"][conf_id]
        print(conf_id, conf)
        for data_id in conf["data"]:
            val = conf["data"][data_id]
            ran = conf["range"]

            trackbar_title = conf["title"] + (" - " + data_id if data_id != "val" else "")
            
            # The trackbar only supports integers, so make floats 10x larger
            if conf["type"] == "float":
                val = int(val * 10)
                ran = [int(r * 10) for r in ran]
                trackbar_title += " (*10)"

            cv2.createTrackbar(
                trackbar_title, 
                "Config", 
                val,
                ran[1],
                create_callback(selected_tab, conf_id, data_id)
            )

            print("Created Trackbar: " + trackbar_title)
    
    while True:
        config_values = {
            "calibration_map": calibration_map,
            "black_line_threshold": config_data["main"]["configs"]["black_line_threshold"]["data"]["val"],
            "black_rescue_threshold": config_data["main"]["configs"]["black_rescue_threshold"]["data"]["val"],
            "green_turn_hsv_threshold": [np.array(bound) for bound in [
                [
                    config_data["green"]["configs"]["green_turn_hsv_threshold"]["data"]["L-H"],
                    config_data["green"]["configs"]["green_turn_hsv_threshold"]["data"]["L-S"],
                    config_data["green"]["configs"]["green_turn_hsv_threshold"]["data"]["L-V"]
                ],
                [
                    config_data["green"]["configs"]["green_turn_hsv_threshold"]["data"]["H-H"],
                    config_data["green"]["configs"]["green_turn_hsv_threshold"]["data"]["H-S"],
                    config_data["green"]["configs"]["green_turn_hsv_threshold"]["data"]["H-V"]
                ],
            ]],
            "red_hsv_threshold": [np.array(bound) for bound in [
                [
                    config_data["red"]["configs"]["red_hsv_threshold"]["data"]["L-H"],
                    config_data["red"]["configs"]["red_hsv_threshold"]["data"]["L-S"],
                    config_data["red"]["configs"]["red_hsv_threshold"]["data"]["L-V"]
                ],
                [
                    config_data["red"]["configs"]["red_hsv_threshold"]["data"]["H-H"],
                    config_data["red"]["configs"]["red_hsv_threshold"]["data"]["H-S"],
                    config_data["red"]["configs"]["red_hsv_threshold"]["data"]["H-V"]
                ],
            ]],
            "rescue_block_hsv_threshold": [np.array(bound) for bound in [
                [
                    config_data["block"]["configs"]["rescue_block_hsv_threshold"]["data"]["L-H"],
                    config_data["block"]["configs"]["rescue_block_hsv_threshold"]["data"]["L-S"],
                    config_data["block"]["configs"]["rescue_block_hsv_threshold"]["data"]["L-V"]
                ],
                [
                    config_data["block"]["configs"]["rescue_block_hsv_threshold"]["data"]["H-H"],
                    config_data["block"]["configs"]["rescue_block_hsv_threshold"]["data"]["H-S"],
                    config_data["block"]["configs"]["rescue_block_hsv_threshold"]["data"]["H-V"]
                ],
            ]],
            "rescue_circle_conf": {
                "dp": config_data["circle"]["configs"]["rescue_circle_dp"]["data"]["val"],
                "minDist": config_data["circle"]["configs"]["rescue_circle_minDist"]["data"]["val"],
                "param1": config_data["circle"]["configs"]["rescue_circle_param1"]["data"]["val"],
                "param2": config_data["circle"]["configs"]["rescue_circle_param2"]["data"]["val"],
                "minRadius": config_data["circle"]["configs"]["rescue_circle_minRadius"]["data"]["val"],
                "maxRadius": config_data["circle"]["configs"]["rescue_circle_maxRadius"]["data"]["val"]
            },
            "rescue_circle_minradius_offset": config_data["circle"]["configs"]["rescue_circle_minradius_offset"]["data"]["val"],
            "rescue_binary_gray_scale_multiplier": config_data["block"]["configs"]["rescue_binary_gray_scale_multiplier"]["data"]["val"]
        }

        if cam is None:
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

        cam.set_processing_conf({
            "calibration_map": config_values["calibration_map"],
            "black_line_threshold": config_values["black_line_threshold"],
            "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
            "red_hsv_threshold": config_values["red_hsv_threshold"],
        })

        frame_processed = cam.read_stream_processed()
        if (frame_processed is None or frame_processed["resized"] is None):
            print("Waiting for image...")
            time.sleep(0.5)
            continue

        img0 = frame_processed["resized"].copy()
        img0_clean = img0.copy() # Used for displaying the image without any overlays

        img0_gray = frame_processed["gray"].copy()
        img0_gray_scaled = frame_processed["gray_scaled"].copy()
        img0_binary = frame_processed["binary"].copy()
        img0_hsv = frame_processed["hsv"].copy()
        img0_green = frame_processed["green"].copy()
        img0_line = frame_processed["line"].copy()

        print(config_values["red_hsv_threshold"])
        img0_red = cv2.bitwise_not(cv2.inRange(img0_hsv, config_values["red_hsv_threshold"][0], config_values["red_hsv_threshold"][1]))
        img0_red = cv2.dilate(img0_red, np.ones((5,5),np.uint8), iterations=2)

        img0_gray_rescue_calibrated = calibration_map_rescue * img0_gray
        img0_binary_rescue = ((img0_gray_rescue_calibrated > config_values["black_rescue_threshold"]) * 255).astype(np.uint8)
        img0_binary_rescue = cv2.morphologyEx(img0_binary_rescue, cv2.MORPH_OPEN, np.ones((13,13),np.uint8))

        img0_gray_rescue_scaled = img0_gray_rescue_calibrated * (config_values["rescue_binary_gray_scale_multiplier"] - 2.5)
        img0_gray_rescue_scaled = np.clip(img0_gray_rescue_calibrated, 0, 255).astype(np.uint8)
        
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
        sorted_circles = []
        if circles is not None:
            detected_circles = np.round(circles[0, :]).astype(int)
            detected_circles = sorted(detected_circles , key = lambda v: [v[1], v[1]],reverse=True)
            
            # If the circle is in the bottom half of the image, check that the radius is greater than 40
            # valid_circles = []
            # for (x, y, r) in detected_circles:
            #     if y > config_values["rescue_circle_conf"]["heightBuffer"] and r > config_values["rescue_circle_conf"]["lowHeightMinRadius"]:
            #         valid_circles.append([x, y, r])
            #     elif y <= config_values["rescue_circle_conf"]["heightBuffer"]:
            #         valid_circles.append([x, y, r])

            # Draw the detected circles on the original image
            for (x, y, r) in detected_circles:
                cv2.circle(img0_circles, (x, y), r, (0, 0, 255), 2)
                cv2.circle(img0_circles, (x, y), 2, (0, 0, 255), 3)

            height_bar_qty = 13
            height_bar_minRadius = [a - 7 for a in [90, 85, 80, 72, 66, 58, 52, 44, 38, 31, 23, 16, 15]] # This could become a linear function... but it works for now
            height_bar_maxRadius = [b + config_values["rescue_circle_minradius_offset"] for b in height_bar_minRadius]

            height_bars = [(img0.shape[0] / height_bar_qty) * i for i in range(height_bar_qty - 1, -1, -1)]
            height_bar_circles = [[] for i in range(height_bar_qty)]
            for (x, y, r) in detected_circles:
                for i in range(height_bar_qty):
                    if y >= height_bars[i]:
                        if r >= height_bar_minRadius[i] and r <= height_bar_maxRadius[i]: 
                            height_bar_circles[i].append([x, y, r, i])
                        break
                
                # Sort the circles in each bar by x position
                height_bar_circles[i] = sorted(height_bar_circles[i], key = lambda v: [v[0], v[0]])

            # Compile circles into a single list, horizontally in height bar (closest to furthest)
            for i in range(height_bar_qty):
                sorted_circles += height_bar_circles[i]
            
            # Draw horizontal lines for height bars
            for i in range(height_bar_qty):
                cv2.line(img0_circles, (0, int(height_bars[i])), (img0.shape[1], int(height_bars[i])), (255, 255, 255), 1)
            
            # Draw the sorted circles on the original image
            for i, (x, y, r, bar) in enumerate(sorted_circles):
                cv2.circle(img0_circles, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img0_circles, (x, y), 2, (0, 255, 0), 3)
                print(bar, i, r)
                cv2.putText(img0_circles, f"{bar}-{i}-{r}", (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 125, 255), 2)
        else:
            cv2.putText(img0_circles, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        img0_block_mask = cv2.inRange(img0_hsv, config_values["rescue_block_hsv_threshold"][0], config_values["rescue_block_hsv_threshold"][1])
        img0_binary_rescue_block = cv2.bitwise_and(cv2.bitwise_not(img0_binary_rescue), img0_block_mask)
        img0_binary_rescue_block = cv2.morphologyEx(img0_binary_rescue_block, cv2.MORPH_OPEN, np.ones((13,13),np.uint8))

        contours_block = cv2.findContours(img0_binary_rescue_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contours_block = [{
            "contour": c, 
            "contourArea": cv2.contourArea(c),
            "boundingRect": cv2.boundingRect(c),
            "touching_sides": [],
            "near_sides": []
        } for c in contours_block]

        # Figure out which sides of the image the contours are touching
        side_threshold = [10, 50]
        for i, contour in enumerate(contours_block):
            x, y, w, h = contour["boundingRect"]

            if x < side_threshold[0]: contours_block[i]["touching_sides"].append("left")
            if y < side_threshold[0]: contours_block[i]["touching_sides"].append("top")
            if x + w > img0.shape[1] - side_threshold[0]: contours_block[i]["touching_sides"].append("right")
            if y + h > img0.shape[0] - side_threshold[0]: contours_block[i]["touching_sides"].append("bottom")

            if x < side_threshold[1]: contours_block[i]["near_sides"].append("left")
            if y < side_threshold[1]: contours_block[i]["near_sides"].append("top")
            if x + w > img0.shape[1] - side_threshold[1]: contours_block[i]["near_sides"].append("right")
            if y + h > img0.shape[0] - side_threshold[1]: contours_block[i]["near_sides"].append("bottom")
        
        # Filter by area and ensure it doesn't touch the top
        contours_block = [c for c in contours_block if c["contourArea"] > 10000 and "top" not in c["touching_sides"]]
        
        # Sort by area
        contours_block = sorted(contours_block, key=lambda c: c["contourArea"], reverse=True)

        cv2.drawContours(img0, [c["contour"] for c in contours_block], -1, (0, 0, 255), 2)
        if len(contours_block) > 0:
            contour_block = contours_block[0]

            cv2.rectangle(img0, contour_block["boundingRect"], (0, 255, 0), 2)

            cx = contour_block["boundingRect"][0] + contour_block["boundingRect"][2] / 2
            cy = contour_block["boundingRect"][1] + contour_block["boundingRect"][3] / 2

            cv2.circle(img0, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        for req_img in config_data[selected_tab]["images"]:
            img0_preview = None

            if req_img == "img0": img0_preview = img0
            elif req_img == "img0_gray_scaled": img0_preview = img0_gray_scaled
            elif req_img == "img0_binary": img0_preview = img0_binary
            elif req_img == "img0_line": img0_preview = img0_line
            elif req_img == "img0_hsv": img0_preview = img0_hsv
            elif req_img == "img0_green": img0_preview = img0_green
            elif req_img == "img0_red": img0_preview = img0_red
            elif req_img == "img0_line": img0_preview = img0_line
            elif req_img == "img0_binary_rescue": img0_preview = img0_binary_rescue
            elif req_img == "img0_gray_rescue_scaled": img0_preview = img0_gray_rescue_scaled
            elif req_img == "img0_binary_rescue_block": img0_preview = img0_binary_rescue_block
            elif req_img == "img0_circles": img0_preview = img0_circles
            elif req_img == "img0_block_mask": img0_preview = img0_block_mask

            img0_preview = cv2.resize(img0_preview, (0, 0), fx=0.8, fy=0.7)

            cv2.imshow(req_img, img0_preview)

        k = cv2.waitKey(1)
        if (k & 0xFF == ord('q')):
            break

show_selected_tab(selected_tab)
cv2.destroyAllWindows()
