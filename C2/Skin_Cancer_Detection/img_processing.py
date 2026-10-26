import os
import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "images"))
IMAGE_SIZE = (64, 64)

print(f"Target Images Directory: {IMAGES_DIR}")


df = pd.read_csv("metadata_cleaned.csv", index_col="img_id")

img_processed = []
valid_ids = []


for img_id in df.index:
    img_path = os.path.join(IMAGES_DIR, img_id)

    if os.path.exists(img_path):
        img = Image.open(img_path).convert('L')
        img = img.resize(IMAGE_SIZE)

        to_array_img = np.array(img, dtype=np.float32) / 255.0

        img_processed.append(to_array_img)
        valid_ids.append(img_id)
    else:
        print(f"Missing PNG for img_id: {img_id}")

if len(img_processed) == 0:
    raise ValueError(f"No PNG images were found in: {IMAGES_DIR}\nCheck if your path needs fewer or more '..' steps.")


X_images = np.array(img_processed)
X_ids = np.array(valid_ids)
X_images_flat = X_images.reshape(X_images.shape[0], -1)

np.save("images_greyscale.npy", X_images_flat)
np.save("images_ids.npy", X_ids)

print(f"Successfully processed and saved {X_images_flat.shape[0]} images to 'images_greyscale.npy'!")