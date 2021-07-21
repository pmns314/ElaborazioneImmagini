""" Civale Dario mat. 0622701620"""

import numpy as np
import cv2
import copy


def erosion(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.erode(image, kernel)


def dilatation(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.dilate(image, kernel)


def open(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def close(image, kernel):
    kernel = np.array(kernel, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def median_open_close(image, kernel_dim):
    op = open(image, kernel_dim)
    cl = close(image, kernel_dim)
    med = (op + cl) / 2
    return med


def add_gaussian(image, mean, sigma):
    # gaussian = mean + sigma * np.random.randn(*image.shape)
    gaussian = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
    if len(image.shape) > 2:
        red = image[:, :, 0] + gaussian
        green = image[:, :, 1] + gaussian
        blue = image[:, :, 2] + gaussian
        return np.dstack((red, green, blue))
    else:

        im_noise = image + gaussian
        a = np.where(im_noise > 1, 1, im_noise)
        b = np.where(a < 0, 0, a)
    return b


def add_salt_pepper(image, p):
    sp = copy.deepcopy(image)
    sp[np.random.rand(*image.shape) < p / 2] = 0
    aux = np.random.rand(*sp.shape)
    sp[(aux > p / 2) & (aux < p)] = 1
    return sp