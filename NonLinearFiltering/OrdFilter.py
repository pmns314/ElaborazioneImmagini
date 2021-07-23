import numpy as np

from scipy import signal
import cv2
import copy


#
# def order_filter(image, order, kernel_size):
#     center_kernel = int((kernel_size - 1) / 2)
#     filter_image = np.zeros([image.shape[0], image.shape[1]])
#     border_filling = np.zeros([image.shape[0] + 2 * center_kernel, image.shape[1] + 2 * center_kernel])
#     border_filling[center_kernel:-center_kernel, center_kernel:-center_kernel] = image[:, :]
#
#     for i in range(center_kernel, border_filling.shape[0] - center_kernel):
#         for j in range(center_kernel, border_filling.shape[1] - center_kernel):
#
#             list = []
#             window = border_filling[i - center_kernel: i + center_kernel + 1,
#                      j - center_kernel: j + center_kernel + 1]
#
#             for x in range(window.shape[0]):
#                 for y in range(window.shape[1]):
#                     list.append(window[x][y])
#             list.sort()
#             filter_image[i - center_kernel][j - center_kernel] = list[order - 1]
#
#     return filter_image


def bitonic_filter(image, nh, o, strle):
    sigma = 0.5 * nh / np.sqrt(2)
    mask = strle
    on = int(np.round(o * np.sum(mask)))
    on_c = np.sum(mask) - on - 1
    i_op = order_filter(order_filter(image, mask, on), mask, on_c)
    i_cl = order_filter(order_filter(image, mask, on_c), mask, on)
    ksize = int(6 * np.floor(sigma) + 1)
    # gaussian blur effettua la convoluzione con il kernel gaussiano
    e_op = np.abs(cv2.GaussianBlur(image - i_op, (ksize, ksize), sigma))
    e_cl = np.abs(cv2.GaussianBlur(i_cl - image, (ksize, ksize), sigma))
    return np.divide(np.multiply(e_op, i_cl) + np.multiply(e_cl, i_op), (e_op + e_cl + 10 ** -10))


def order_filter(image, kernel, order):
    if len(image.shape) > 2:

        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
        red_med = signal.order_filter(red, kernel, order)
        green_med = signal.order_filter(green, kernel, order)
        blue_med = signal.order_filter(blue, kernel, order)

        return np.dstack((red_med, green_med, blue_med))

    else:

        return signal.order_filter(image, kernel, order)


def anysotropic_filter(image):
    im = copy.deepcopy(image)
    T = 5
    k = 0.5
    alpha = 1 / 6

    for n in range(T):
        Dx = np.zeros([*im.shape])
        Dy = np.zeros([*im.shape])
        Dx[:-1, :] = np.diff(im, axis=0)
        Dy[:, :-1] = np.diff(im, axis=1)
        d0 = Dx
        d1 = Dy
        d2 = np.zeros([*im.shape])
        d2[1:, :] = -Dx[:-1, :]
        d3 = np.zeros([*im.shape])
        d3[:, 1:] = -Dy[:, :-1]
        term0 = np.multiply(np.exp(-np.square(np.abs(np.divide(d0, k)))), d0)
        term1 = np.multiply(np.exp(-np.square(np.abs(np.divide(d1, k)))), d1)
        term2 = np.multiply(np.exp(-np.square(np.abs(np.divide(d1, k)))), d2)
        term3 = np.multiply(np.exp(-np.square(np.abs(np.divide(d1, k)))), d3)
        im_it = im + np.multiply(alpha, (term0 + term1 + term2 + term3))
        im = im_it
    return im
