import time
import json
import numpy as np
import cv2
from helpers import camera as c

# Load the calibration map from the JSON file
with open("calibration.json", "r", encoding="utf-8") as json_file:
    calibration_data = json.load(json_file)
calibration_map = 255 / np.array(calibration_data["calibration_map_w"])
calibration_map_obst = 255 / np.array(calibration_data["calibration_map_w_obst"])
calibration_map_silver = 255 / np.array(calibration_data["calibration_map_w_silver"])

with open("config.json", "r", encoding="utf-8") as json_file:
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

frames = 0

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

    for tab_id, loc in btn_locations.items():
        # Check if the x and y is within the button
        if loc[0][0] <= x and x <= loc[1][0] and loc[0][1] <= y and y <= loc[1][1]:
            if event == 1: # Mouse down
                print("Selected tab is now: " + selected_tab)
                show_selected_tab(tab_id)

has_moved_windows = False

line_iterations = 7
def show_selected_tab(tab_id):
    global has_moved_windows
    global selected_tab
    global btn_img
    global cam
    global frames
    global line_iterations

    selected_tab = tab_id

    # cv2.destroyAllWindows()
    cv2.destroyWindow("Config")
    cv2.namedWindow("Config")
    cv2.setMouseCallback("Config", btn_callback)
    cv2.moveWindow("Config", 100, 50)
    btn_img = np.zeros((btn_height, gui_width, 3), dtype=np.uint8)

    draw_btns()
    cv2.imshow("Config", btn_img)

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
            "calibration_map_obst": calibration_map_obst,
            "calibration_map_silver": calibration_map_silver,
            "black_line_threshold": config_data["main"]["configs"]["black_line_threshold"]["data"]["val"],
            "black_silver_threshold": config_data["main"]["configs"]["black_silver_threshold"]["data"]["val"],
            "obstacle_line_threshold": config_data["main"]["configs"]["obstacle_line_threshold"]["data"]["val"],
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
            ]]
        }

        if cam is None:
            cam = c.CameraStream(
                camera_num=0,

            )
            cam.wait_for_image()

        cam.set_processing_conf({
            "calibration_map": config_values["calibration_map"],
            "calibration_map_silver": config_values["calibration_map_silver"],
            "black_line_threshold": config_values["black_line_threshold"],
            "black_silver_threshold": config_values["black_silver_threshold"],
            "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
            "red_hsv_threshold": config_values["red_hsv_threshold"],
        })

        frames += 1

        frame_processed = cam.read_stream_processed()
        if (frame_processed is None or frame_processed["resized"] is None):
            print("Waiting for image...")
            time.sleep(0.5)
            continue
        
        img0 = frame_processed["resized"].copy()
        img0_raw = frame_processed["raw"].copy()
        img0_clean = img0.copy() # Used for displaying the image without any overlays

        img0_gray = frame_processed["gray"].copy()
        img0_gray_scaled = frame_processed["gray_scaled"].copy()
        img0_binary = frame_processed["binary"].copy()
        img0_hsv = frame_processed["hsv"].copy()
        img0_green = frame_processed["green"].copy()
        img0_line = frame_processed["line"].copy()
        img0_silver_binary = frame_processed["silver_binary"].copy()

        if frames % 120 < 60:
            img0_line = cv2.erode(img0_line, np.ones((5, 5), np.uint8), iterations=line_iterations)
            img0_line = cv2.dilate(img0_line, np.ones((5, 5), np.uint8), iterations=line_iterations)

        img0_resized_obst = cam.resize_image_obstacle(img0_raw)
        img0_gray_obst = cv2.cvtColor(img0_resized_obst, cv2.COLOR_BGR2GRAY)
        img0_gray_obst = cv2.GaussianBlur(img0_gray_obst, (5, 5), 0)
        img0_gray_obst_scaled = img0_gray_obst * calibration_map_obst

        img0_binary_obstacle = ((img0_gray_obst_scaled > config_values["obstacle_line_threshold"]) * 255).astype(np.uint8)
        img0_binary_obstacle = cv2.morphologyEx(img0_binary_obstacle, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

        img0_red = cv2.bitwise_not(cv2.inRange(img0_hsv, config_values["red_hsv_threshold"][0], config_values["red_hsv_threshold"][1]))
        img0_red = cv2.dilate(img0_red, np.ones((5, 5), np.uint8), iterations=2)

        images = ["img0", "img0_gray_scaled", "img0_binary", "img0_line", "img0_hsv", "img0_green", "img0_red", "img0_gray_obst_scaled", "img0_binary_obstacle", "img0_silver_binary"]
        # for req_img in config_data[selected_tab]["images"]:
        for req_img in images:
            img0_preview = None

            if req_img == "img0": img0_preview = img0
            elif req_img == "img0_gray_scaled": img0_preview = img0_gray_scaled
            elif req_img == "img0_binary": img0_preview = img0_binary
            elif req_img == "img0_line": img0_preview = img0_line
            elif req_img == "img0_hsv": img0_preview = img0_hsv
            elif req_img == "img0_green": img0_preview = img0_green
            elif req_img == "img0_red": img0_preview = img0_red
            elif req_img == "img0_silver_binary": img0_preview = img0_silver_binary
            elif req_img == "img0_gray_obst_scaled": img0_preview = img0_gray_obst_scaled
            elif req_img == "img0_binary_obstacle": img0_preview = img0_binary_obstacle

            img0_preview = cv2.resize(img0_preview, (0, 0), fx=0.8, fy=0.7)

            cv2.imshow(req_img, img0_preview)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break

        if k & 0xFF == ord('l'):
            line_iterations = int(input("Line iterations: "))

        if not has_moved_windows:
            has_moved_windows = True
            # for req_img in config_data[selected_tab]["images"]:
            for req_img in images:
                loc_target = None

                base_left = 510
                base_top = 50
                x_split = 250
                y_split = 250
                if req_img == "img0": loc_target = [base_left + (x_split * 0), base_top + (y_split * 0)]
                elif req_img == "img0_gray_scaled": loc_target = [base_left + (x_split * 1), base_top + (y_split * 0)]
                elif req_img == "img0_binary": loc_target = [base_left + (x_split * 2), base_top + (y_split * 0)]
                elif req_img == "img0_line": loc_target = [base_left + (x_split * 3), base_top + (y_split * 0)]
                elif req_img == "img0_hsv": loc_target = [base_left + (x_split * 0), base_top + (y_split * 1)]
                elif req_img == "img0_green": loc_target = [base_left + (x_split * 1), base_top + (y_split * 1)]
                elif req_img == "img0_red": loc_target = [base_left + (x_split * 2), base_top + (y_split * 1)]
                elif req_img == "img0_silver_binary": loc_target = [base_left + (x_split * 3), base_top + (y_split * 1)]
                elif req_img == "img0_gray_obst_scaled": loc_target = [base_left + (x_split * 0), base_top + (y_split * 2)]
                elif req_img == "img0_binary_obstacle": loc_target = [base_left + (x_split * 2), base_top + (y_split * 2)]

                cv2.moveWindow(req_img, loc_target[0], loc_target[1])

show_selected_tab(selected_tab)
cv2.destroyAllWindows()
