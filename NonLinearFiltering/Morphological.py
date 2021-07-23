import numpy as np
import cv2


def erosion(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.erode(image, kernel)


def dilatation(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.dilate(image, kernel)


def morphological_open(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morphological_close(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def median_open_close(image, kernel_dim):
    op = morphological_open(image, kernel_dim)
    cl = morphological_close(image, kernel_dim)
    med = (op + cl) / 2
    return med
