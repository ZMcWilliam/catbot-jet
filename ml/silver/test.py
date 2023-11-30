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

loaded = tf.saved_model.load("ml/model/silver/trt")
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
            cv2.putText(img0, "SILVER", (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 0, 160), 2)

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
