import os
import cv2
import numpy as np

def load_brain_tumor_dataset(root_dir):
    """ Loads brain tumor dataset from the specified directory structure."""

    classes = ["glioma", "meningioma", "pituitary"]
    X, y = [], []

    # Iterate through each class folder and load images
    for label, cls in enumerate(classes):
        folder = os.path.join(root_dir, cls)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), classes