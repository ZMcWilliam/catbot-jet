import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import camera as c
from helpers import config

cam = c.CameraStream(
    camera_num=0,
    processing_conf=config.processing_conf
)
cam.wait_for_image()

model = YOLO("ml/model/corners.pt")

frames = 0
start_time = time.time()
try:
    while True:
        frames += 1

        frame_processed = cam.read_stream_processed()
        img0 = frame_processed["raw"].copy()
        img0_resized_evac = cv2.resize(img0, (145, 132))

        start_inf = time.time()
        
        results = model(img0_resized_evac)

        inference_time_ms = (time.time() - start_inf) * 1000
        current_fps = frames / (time.time() - start_time)

        found_corners = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1 * img0.shape[1] / 160),
                    int(y1 * img0.shape[0] / 120),
                    int(x2 * img0.shape[1] / 160),
                    int(y2 * img0.shape[0] / 120),
                )
                
                conf = int(box.conf * 100)

                obj_type = "red" if int(box.cls[0]) else "green"
                obj_col = (0, 0, 255) if int(box.cls[0]) else (0, 255, 0)

                cv2.rectangle(img0, (x1, y1), (x2, y2), obj_col, 3)
                cv2.putText(img0, f"{obj_type} ({conf})", [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, obj_col, 2)
                found_corners += 1

        if found_corners > 0:
            cv2.putText(img0, f"{found_corners} FOUND", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)
        else:
            cv2.putText(img0, "NONE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

        cv2.putText(img0, f"{inference_time_ms:.2f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
        cv2.putText(img0, f"{current_fps:.1f} fps", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)

        cv2.imshow("Corner Detection", img0)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        
        print(f"Amt: {found_corners} | FPS: {current_fps:.2f} | Inference: {inference_time_ms:.2f} ms")
except KeyboardInterrupt:
    print("Main loop interrupted. Exiting...")

cam.stop()
cv2.destroyAllWindows()
