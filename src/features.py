from skimage.feature import hog
import numpy as np

def extract_hog_features(images):
    """ Extract hog features from images """
    hog_features = []
    for img in images:
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        hog_features.append(feat)
    return np.array(hog_features)