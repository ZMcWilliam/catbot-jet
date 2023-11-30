import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

TARGET_WIDTH = 72
TARGET_HEIGHT = 66
MODEL = "silver"

TRAIN_PATH = f"ml/data_out/{MODEL}/train"
TEST_PATH = f"ml/data_out/{MODEL}/test"
MODEL_OUTPUT_PATH = f"ml/model/_tmp/{MODEL}/saved"
MODEL_OUTPUT_PATH_TRT = f"ml/model/{MODEL}-trt"

# Ensure that the directories exist or create them
for directory in [MODEL_OUTPUT_PATH, MODEL_OUTPUT_PATH_TRT]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load images for training
train_images = []
train_targets = []

for category in ["with", "without"]:
    for filename in os.listdir(os.path.join(TRAIN_PATH, category)):
        image_path = os.path.join(TRAIN_PATH, category, filename)
        img0_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img0_gray = cv2.resize(img0_gray, (0,0), fx=0.5, fy=0.5)
        train_images.append(img0_gray)
        train_targets.append(0 if category == "without" else 1)

train_images = np.array(train_images)
train_targets = np.array(train_targets)

print(f"Loaded {len(train_images)} training images and {len(train_targets)} targets.")

# Load images for testing
test_images = []
test_targets = []

for category in ["with", "without"]:
    for filename in os.listdir(os.path.join(TEST_PATH, category)):
        image_path = os.path.join(TEST_PATH, category, filename)
        img0_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img0_gray = cv2.resize(img0_gray, (0,0), fx=0.5, fy=0.5)

        test_images.append(img0_gray)
        test_targets.append(0 if category == "without" else 1)

print(f"Loaded {len(test_images)} testing images and {len(test_targets)} targets.")

# Define the model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Rescaling(1./255, input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1)))
model.add(tf.keras.layers.Conv2D(4, 5, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(8))
model.add(tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(2))

model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)

model.summary()

# Train the model
model.fit(train_images, train_targets, batch_size=10, epochs=40, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(np.array(test_images), np.array(test_targets), verbose=2)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save(MODEL_OUTPUT_PATH)

# Create a TensorRT converter
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=MODEL_OUTPUT_PATH,
    precision_mode=trt.TrtPrecisionMode.FP16
)

trt_model = converter.convert()
converter.summary()

def input_fn():
    input_shape = (TARGET_HEIGHT, TARGET_WIDTH, 1)
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    yield [dummy_input]

# Build TensorRT engines before deployment to save time at runtime
converter.build(input_fn=input_fn)
converter.save(MODEL_OUTPUT_PATH_TRT)

# Evaluate the TRT model
loaded = tf.saved_model.load(f"ml/model/{MODEL}/trt")
infer = loaded.signatures["serving_default"]

test_input_data = []
for image in test_images:
    input_data = np.expand_dims(image, axis=-1)  # Add a channel dimension
    test_input_data.append(input_data)

test_input_data = np.array(test_input_data)
test_input_data = test_input_data.reshape(-1, 66, 72, 1)

labelling = infer(tf.constant(test_input_data, dtype=float))
label_key = list(labelling.keys())[0]
predicted_labels = labelling[label_key].numpy().argmax(axis=1)

correct_predictions = np.sum(predicted_labels == np.array(test_targets))
trt_accuracy = correct_predictions / len(test_targets)

print(f"Accuracy: {trt_accuracy:.4f} ({correct_predictions}/{len(test_targets)}). Before: {test_accuracy:.4f}")
