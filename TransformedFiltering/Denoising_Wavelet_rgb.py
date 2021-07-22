""" Mansi Paolo mat. 0622701542"""

import math
import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.metrics


def show_image_from_coefficients(coeffs, title=None):
    sh = coeffs[len(coeffs) - 1][0].shape

    C = np.ones([sh[0] * 2, sh[1] * 2])
    dim_rows, dim_cols = coeffs[0].shape
    for i in range(len(coeffs)):
        if i == 0:
            C[:dim_rows, :dim_cols] = coeffs[i]
        else:
            dim_rows, dim_cols = coeffs[i][0].shape
            C[:dim_rows, dim_cols:dim_cols * 2] = coeffs[i][0]
            C[dim_rows:dim_rows * 2, :dim_cols] = coeffs[i][1]
            C[dim_rows:dim_rows * 2, dim_cols:dim_cols * 2] = coeffs[i][2]
    plt.imshow(C, vmin=-0.5, vmax=2, cmap='jet')
    if title is not None:
        plt.title(title)
    return


def get_threshold(image, wname):
    coeff = pywt.wavedec2(image, wname, 1)
    A = np.array([np.concatenate(([*coeff]), axis=None)])
    n = ((coeff[0][1]).shape[0])
    X = A[:, n ** 2 + 1:]
    sigma = np.median(np.abs(X)) / 0.6745
    thr = sigma * math.sqrt(2 * math.log((n * 2) ** 2))
    return thr


def _plot_wavelet_transform(coeff, title, ch, pre):
    if pre is True:
        if title is None:
            title = 'Pre Denoising'
        if ch == 1:
            # fig, axs = plt.subplots(2, 3, constrained_layout=True)
            plt.subplot(231)
            title = title + '- B'
        elif ch == 2:
            plt.subplot(232)
            title = title + '- G'
        elif ch == 3:
            plt.subplot(233)
            title = title + '- R'
        else:
            plt.figure()
            plt.subplot(121)
    else:
        if title is None:
            title = 'Post Denoising'
        if ch == 1:
            plt.subplot(234)
            title = title + '- B'
        elif ch == 2:
            plt.subplot(235)
            title = title + '- G'
        elif ch == 3:
            plt.subplot(236)
            title = title + '- R'
        else:
            plt.subplot(122)
    show_image_from_coefficients(coeff, title)


def _denoise(image, wname, levels, flag_show=None, title=None, ch=-1):
    # Wavelet Transform
    coeff = pywt.wavedec2(image, wname, level=levels)

    # Plot Wavelet Transform
    if flag_show is not None and flag_show[0] is True:
        _plot_wavelet_transform(coeff, None if title is None else title[0], ch, True)

    # Get Threshold
    thr = get_threshold(image, wname)

    # Denoising Wavelet Transform
    for i in range(1, len(coeff)):
        for imm in coeff[i]:
            imm[abs(imm) < thr] = 0

    # Plot Wavelet Tranform Denoised
    if flag_show is not None and flag_show[1] is True:
        _plot_wavelet_transform(coeff, None if title is None else title[1], ch, False)

    # Reconstruct Image
    return pywt.waverec2(coeff, wname)


def denoise(image, wname, levels, channels, flag_show=None, title=None):
    if channels == 1:
        im = _denoise(image, wname, levels, flag_show, title, -1)
    else:
        im = np.zeros([*image.shape])
        for c in range(channels):
            im[:, :, c] = _denoise(image[:, :, c], wname, levels, flag_show, title, c + 1)
    return im


def get_channels_number(image):
    return 1 if len(image.shape) == 2 else image.shape[2]


# Display image
def display(image, title=''):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, image)
    h, w = image.shape[0:2]
    neww = 400
    newh = int(neww * (h / w))
    cv2.resizeWindow(title, neww, newh)


if __name__ == '__main__':
    I = cv2.imread('../images/rgb/peppers.png', -1)
    I_noise = skimage.util.random_noise(I, 'gaussian')
    I = np.uint16(I)

    display(I_noise, 'Original with Noise')
    channels = get_channels_number(I_noise)

    # Definizione dei parametri della trasformata wavelet
    wname = 'sym5'
    levels = pywt.dwtn_max_level(I_noise.shape[:-1], wname)

    print('Level : ', levels, '\n', end='\t\t')
    # Denoise
    de_image = denoise(I_noise, wname, levels, channels, (True, True),
                       title=['Pre_denoise ' + str(levels), 'Post_denoise ' + str(levels)])
    display(de_image, 'Denoised image with level= ' + str(levels))
    # Correzione dimensione immagine
    if I.shape != de_image.shape:
        de_image = de_image[:-1, :]
    de_image = skimage.img_as_uint(de_image / 255)
    print('PSNR: ' + '%.4f' % skimage.metrics.peak_signal_noise_ratio(I, de_image), end='\t\t')
    print('SSIM: ' + '%.4f' % skimage.metrics.structural_similarity(I, de_image, multichannel=(channels != 1)))

    plt.show()
    cv2.waitKey(0)
