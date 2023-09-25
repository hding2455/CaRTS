import cv2
import numpy as np

def mask_denoise(image):
    _, CCs = cv2.connectedComponents(image, connectivity=4)
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        if (CCs == i).sum() < 3000:
            image[CCs == i] = 0
    _, CCs = cv2.connectedComponents(255 - image, connectivity=4)
    labels = np.unique(CCs)
    for i in labels:
        if i == 0:
            continue
        if (CCs == i).sum() < 3000:
            image[CCs == i] = 255
    return image