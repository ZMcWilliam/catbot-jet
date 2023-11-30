import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import motorkit as m
from helpers import camera as c
from helpers import config
from helpers.servokit import ServoManager

servo = ServoManager()

servo.gate.toMin()
servo.lift.toMax()
servo.claw.toMax()
servo.cam.to(80)

cam = c.CameraStream(
    camera_num=0,
    processing_conf=config.processing_conf
)
cam.wait_for_image()

model = YOLO("ml/model/victims.pt")

frames = 0
start_time = time.time()
try:
    circle_check_counter = 0
    while True:
        frames += 1

        frame_processed = cam.read_stream_processed()
        img0_raw = frame_processed["raw"].copy()
        img0_resized_evac = cv2.resize(img0_raw, (160, 120))

        start_inf = time.time()
        results = model(img0_resized_evac)
        inference_time_ms = (time.time() - start_inf) * 1000
        current_fps = frames / (time.time() - start_time)

        found_victims = []
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1 * img0_raw.shape[1] / img0_resized_evac.shape[1]),
                    int(y1 * img0_raw.shape[0] / img0_resized_evac.shape[0]),
                    int(x2 * img0_raw.shape[1] / img0_resized_evac.shape[1]),
                    int(y2 * img0_raw.shape[0] / img0_resized_evac.shape[0]),
                )

                conf = int(box.conf * 100)

                obj_type = int(box.cls[0])

                vert_weight = 2
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                horizontal_distance = midpoint_x - (img0_raw.shape[1] / 2)
                vertical_distance = midpoint_y - img0_raw.shape[0]
                dist_amt = int(math.sqrt(horizontal_distance ** 2 + (vertical_distance * vert_weight) ** 2))
                    
                found_victims.append([obj_type, conf, dist_amt, [midpoint_x, midpoint_y], [horizontal_distance, vertical_distance], [x1, y1, x2, y2]])

                cv2.rectangle(img0_raw, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img0_raw, f"{obj_type} ({conf}) {dist_amt}", [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        v_can_approach = False
        v_target = None
        if len(found_victims) > 0:
            found_victims = sorted(found_victims, key=lambda x: x[2])
            v_target = found_victims[0]
            
            # If a victim is within these limits (horz/vert distance offsets), we can approach it
            approach_range = [[-85, 85], [-100, 0]]

            v_can_approach = approach_range[0][0] < v_target[4][0] < approach_range[0][1] and approach_range[1][0] < v_target[4][1] < approach_range[1][1]
                
            if v_can_approach:
                circle_check_counter += 1
                cv2.putText(img0_raw, f"APPROACH {circle_check_counter}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                circle_check_counter = 0

            # Highlight the target
            cv2.rectangle(img0_raw, (v_target[5][0], v_target[5][1]), (v_target[5][2], v_target[5][3]), (0, 255, 0), 3)

            # Draw a box around the appraoch range
            cv2.rectangle(
                img0_raw, 
                (int(img0_raw.shape[1] / 2) + approach_range[0][0], int(img0_raw.shape[0]) + approach_range[1][0]), 
                (int(img0_raw.shape[1] / 2) + approach_range[0][1], int(img0_raw.shape[0]) + approach_range[1][1]), 
                (0, 255, 255), 
                2
            )

            # Draw midpoint with coord
            cv2.circle(img0_raw, (int(v_target[3][0]), int(v_target[3][1])), 5, (0, 0, 255), -1)
            cv2.putText(img0_raw, f"{v_target[3][0]}, {v_target[3][1]}", (int(v_target[3][0]) + 10, int(v_target[3][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw the horizontal and vertical distance just below
            cv2.putText(img0_raw, f"X: {v_target[4][0]}", (int(v_target[3][0]) + 10, int(v_target[3][1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 165, 255), 2)
            cv2.putText(img0_raw, f"Y: {v_target[4][1]}", (int(v_target[3][0]) + 10, int(v_target[3][1]) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 165, 255), 2)

            cv2.putText(img0_raw, f"{len(found_victims)} FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

            left_speed = 0
            right_speed = 0
        else:
            circle_check_counter = 0
            cv2.putText(img0_raw, "NONE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

        cv2.putText(img0_raw, f"{inference_time_ms:.2f} ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)
        cv2.putText(img0_raw, f"{current_fps:.1f} fps", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)

        # Define the motor speeds
        left_speed = None
        right_speed = None

        if circle_check_counter > 0:
            # Stop, and make sure we see the circle a bit more
            left_speed = 0
            right_speed = 0

            if circle_check_counter >= 3:
                # 3 valid approach signals in a row, stop and grab
                m.run_tank(30, 30)
                time.sleep(0.2)
                servo.claw.toMin()
                time.sleep(0.6)
                m.run_tank_for_time(-30, -30, 500)
                servo.cam.toMin() # Get the camera out of the way
                servo.lift.toMin()
                time.sleep(0.8)
                servo.claw.toMax() # Release the victim
                time.sleep(1)

                # Return to search
                circle_check_counter = 0
                servo.lift.toMax()
                servo.cam.to(80)
                time.sleep(1)
        elif v_target is not None:
            # Calculate the horizontal and vertical distances
            horizontal_distance = v_target[4][0]
            vertical_distance = v_target[4][1]

            if vertical_distance >= -200:
                # Victim is closer in height, steer on the spot
                if -60 <= horizontal_distance <= 60:
                    # Within acceptable horizontal range, slowly go forward
                    left_speed = 30
                    right_speed = 30
                elif horizontal_distance < -60:
                    # Victim is to the left, turn left
                    left_speed = -30
                    right_speed = 30
                elif horizontal_distance > 60:
                    # Victim is to the right, turn right
                    left_speed = 30
                    right_speed = -30
            else:
                # Victim is further away
                if -80 <= horizontal_distance <= 80:
                    # Within acceptable horizontal range, move forward
                    left_speed = 40
                    right_speed = 40
                else:
                    steering_factor = horizontal_distance / (img0_raw.shape[1] / 2)
                    left_speed = 20 + (40 * steering_factor)
                    right_speed = 20 - (40 * steering_factor)

        if left_speed is None or right_speed is None:
            left_speed = -30
            right_speed = 30

        m.run_tank(left_speed, right_speed)

        # Draw speeds
        cv2.putText(img0_raw, f"L: {left_speed}", (img0_raw.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)
        cv2.putText(img0_raw, f"R: {right_speed}", (img0_raw.shape[1] - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 172, 134), 2)

        cv2.imshow("Victim Detection", img0_raw)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        
        print(f"Amt: {len(found_victims)} | FPS: {current_fps:.2f} | Inference: {inference_time_ms:.2f} ms")
except KeyboardInterrupt:
    print("Main loop interrupted. Exiting...")

cam.stop()
cv2.destroyAllWindows()
