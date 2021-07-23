import numpy as np
import copy


def add_gaussian(image, mean, sigma):
    im = copy.deepcopy(image)
    noise = np.random.normal(mean, sigma, im.shape)
    return np.clip(noise + image, 0.0, 1.0)


def add_salt_pepper(image, p):
    sp = copy.deepcopy(image)
    sp[np.random.rand(*image.shape) < p / 2] = 0
    aux = np.random.rand(*sp.shape)
    sp[(aux > p / 2) & (aux < p)] = 1
    return sp
