import cv2
import os
import numpy as np

TRAIN_PERCENTAGE = 0.8
SKIP_AMT = 4 # Skip every n images
TARGET_INPUT_PATH = "ml/data_in/silver"
UPSAMPLED_IMAGES_PATH = "ml/data_out/silver"

# Create out directories if they don't exist
for folder_type in ["with", "without"]:
    for dataset_type in ["train", "test"]:
        path = os.path.join(UPSAMPLED_IMAGES_PATH, dataset_type, folder_type)
        os.makedirs(path, exist_ok=True)

# Collate all images with labels ("with" or "without")
image_data = []
for folder_type in ["with", "without"]:
    folder_path = os.path.join(TARGET_INPUT_PATH, folder_type)
    for i, filename in enumerate(os.listdir(folder_path)):
        if i % SKIP_AMT != 0:
            continue
        image_path = os.path.join(folder_path, filename)
        image_data.append((image_path, folder_type))

# Iterate over all images and split them into train and test sets
image_processed = []
for image_path, folder_type in image_data:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_flipped = np.flip(image, axis=1)

    image_processed.append((image, image_path, folder_type))
    image_processed.append((image_flipped, image_path[:-4] + "_flip.png", folder_type))

# Shuffle the processed images
np.random.shuffle(image_processed)

# Split the images into train and test sets
split_index = int(len(image_processed) * TRAIN_PERCENTAGE)

for i, (image, image_path, folder_type) in enumerate(image_processed):
    dataset_type = "train" if i < split_index else "test"

    out_folder = os.path.join(UPSAMPLED_IMAGES_PATH, dataset_type, folder_type)
    out_filename = os.path.basename(image_path)

    cv2.imwrite(os.path.join(out_folder, out_filename), image)

    print(f"Processed {i + 1}/{len(image_processed)} images.", end="\r")

print(f"Processed {i}/{len(image_processed)} images. Train: {split_index}, Test: {len(image_processed) - split_index}")
