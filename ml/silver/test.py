import os
import time
import cv2
import numpy as np
from helpers import camera as c
from helpers import config

import tensorflow as tf

cam = c.CameraStream(
    camera_num=0,
    processing_conf=config.processing_conf
)
cam.wait_for_image()

loaded = tf.saved_model.load("ml/model/silver-trt")
infer = loaded.signatures["serving_default"]
print("Loaded model. Waiting for first inference...")

def prepare_frame(frame):
    resized_frame = cv2.resize(frame, (72, 66))
    input_data = np.expand_dims(resized_frame, axis=-1) # Add a channel dimension

    # Convert input_data to a TensorFlow Tensor with a batch dimension
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    input_data = tf.expand_dims(input_data, axis=0)
    return input_data

start_inf = time.time()
frame_processed = cam.read_stream_processed()
input_data = prepare_frame(frame_processed["gray"])
labelling = infer(tf.constant(input_data, dtype=float))
inference_time_ms = (time.time() - start_inf) * 1000
print(f"Initial Inference: {inference_time_ms:.2f} ms")

frames = 0
start_time = time.time()
try:
    while True:
        frames += 1

        frame_processed = cam.read_stream_processed()
        img0 = frame_processed["resized"]
        img0_binary = frame_processed["binary"].copy()

        img0_green = frame_processed["green"].copy()
        img0_line = frame_processed["line"].copy()

        img0_line_not = cv2.bitwise_not(img0_line)

        raw_white_contours, _ = cv2.findContours(img0_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black_contours, _ = cv2.findContours(img0_line_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        input_data = prepare_frame(frame_processed["gray"])

        start_inf = time.time()

        # Run inference
        labelling = infer(tf.constant(input_data, dtype=float))
        label_key = list(labelling.keys())[0] # Currently dense_2
        labels = labelling[label_key]

        inference_time_ms = (time.time() - start_inf) * 1000
        current_fps = frames / (time.time() - start_time)

        # Get the predicted class (0 for "without" and 1 for "with")
        predicted_class = int(np.argmax(labels, axis=1))

        if predicted_class == 1:
            top_check_threshold = 20
            sides_touching = []
            black_filtered = [c for c in black_contours if cv2.contourArea(c) > 500]
            for black_rect in [cv2.boundingRect(c) for c in black_filtered]:
                if "bottom" not in sides_touching and black_rect[1] + black_rect[3] > img0_binary.shape[0] - 3:
                    sides_touching.append("bottom")
                if "left" not in sides_touching and black_rect[0] < 20:
                    sides_touching.append("left")
                if "right" not in sides_touching and black_rect[0] + black_rect[2] > img0_binary.shape[1] - 20:
                    sides_touching.append("right")
                if "top" not in sides_touching and black_rect[1] < 20:
                    sides_touching.append("top")
            total_green_area = np.count_nonzero(img0_green == 0)

            check_a = sum([sum(a) for a in img0_binary[0:top_check_threshold]]) == img0_binary.shape[1] * 255 * top_check_threshold
            check_b = "bottom" in sides_touching
            check_c = total_green_area < 500
            if check_a and check_b and check_c:
                cv2.putText(img0, "CONFIRM", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

            cv2.putText(img0, "SILVER", (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

            cv2.putText(img0, "A: " + str(check_a), (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0, "B: " + str(check_b), (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0, "C: " + str(check_c), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0, "Green Area: " + str(total_green_area), (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0, "Sides: " + str(sides_touching), (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
            cv2.putText(img0, f"Blk Num: {len(black_contours)}/{len(black_filtered)}", (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)

        cv2.putText(img0, f"{inference_time_ms:.2f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)
        cv2.putText(img0, f"{current_fps:.1f} fps", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 40), 2)

        cv2.imshow("Silver Detection", img0)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        
        print(f"Pred: {predicted_class} | FPS: {current_fps:.2f} | Inference: {inference_time_ms:.2f} ms")
except KeyboardInterrupt:
    print("Main loop interrupted. Exiting...")

cam.stop()
cv2.destroyAllWindows()
