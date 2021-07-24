""" Mansi Paolo mat. 0622701542"""

import math
import pywt
import skimage.metrics
import skimage.util
import scipy.signal as ss
from Utils import *


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


def _plot_wavelet_transform(coeff, ch, pre):
    if pre is True:
        title = 'Pre Denoising'
        if ch == 1:
            # fig, axs = plt.subplots(2, 3, constrained_layout=True)
            plt.figure()
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


def get_universal_threshold(coeff):
    A = np.array([np.concatenate(([*coeff]), axis=None)])
    n = ((coeff[0][1]).shape[0])
    X = A[:, n ** 2 + 1:]
    sigma = np.median(np.abs(X)) / 0.6745
    thr = sigma * math.sqrt(2 * math.log((n * 2) ** 2))
    return thr


def denoising_coefficients(coeff, mode, dim_neigh):
    # Get Threshold
    univ_thr = get_universal_threshold(coeff)

    if mode == 'univ':
        for i in range(1, len(coeff)):
            for imm in coeff[i]:
                imm[abs(imm) < univ_thr] = 0

    elif mode == 'neigh':
        mask = np.ones([dim_neigh, dim_neigh])
        for c in range(1, len(coeff)):
            denoised_coeff = [0, 0, 0]
            for i in range(0, 3):
                imm = coeff[c][i]
                S = ss.convolve2d(imm * imm, mask, 'same')
                B = 1 - (univ_thr ** 2 / S)
                B[B < 0] = 0
                denoised_coeff[i] = imm * B
            coeff[c] = tuple(denoised_coeff)
    else:
        raise Exception('Mode not supported')
    return coeff


def _denoising(image, wname, levels, ch, mode, dim_neigh, show):
    # Wavelet Transform
    coeff = pywt.wavedec2(image, wname, level=levels)

    # Plot Wavelet Transform
    if show is True:
        _plot_wavelet_transform(coeff, ch, True)

    # Denoising Wavelet Transform
    denoised_coeff = denoising_coefficients(coeff, mode, dim_neigh)

    # Plot Wavelet Tranform Denoised
    if show is True:
        _plot_wavelet_transform(denoised_coeff, ch, False)

    # Reconstruct Image
    return pywt.waverec2(denoised_coeff, wname)


def denoise_image(image, wname='db3', levels=None, mode='neigh', dim_neigh=3, show=False):
    channels = get_channels_number(image)
    if levels is None:
        levels = pywt.dwtn_max_level(image.shape[:-1], wname)

    if channels == 1:
        im = _denoising(image, wname, levels, -1, mode, dim_neigh, show)
    else:
        im = np.zeros([*image.shape])
        for c in range(channels):
            res = _denoising(image[:, :, c], wname, levels, c + 1, mode, dim_neigh, show)
            if res.shape[0] != image.shape[0]:
                res = res[:-1, :]
            if res.shape[1] != image.shape[1]:
                res = res[:, :-1]
            im[:, :, c] = res
    return im


if __name__ == '__main__':
    I = cv2.imread('../images/b&w/cameraman.tif', -1)
    display(I, 'Original')
    I_noise = skimage.util.random_noise(I, 'speckle')
    I = np.uint16(I)
    display(I_noise, 'Original with Noise')
    channels = get_channels_number(I_noise)

    # Definizione dei parametri della trasformata wavelet
    wname = 'db3'
    levels = pywt.dwtn_max_level(I_noise.shape[:-1], wname)

    print('Level : ', levels, '\n', end='\t\t')
    # Denoise
    de_image = denoise_image(I_noise, wname, levels, channels, mode='neigh', show=False)
    display(de_image, 'Denoised image with neigh ')
    de_image = denoise_image(I_noise, wname, levels, channels, mode='univ', show=False)
    display(de_image, 'Denoised image with univ ')

    # Correzione dimensione immagine
    if I.shape != de_image.shape:
        de_image = de_image[:-1, :]
    de_image = skimage.img_as_uint(de_image / 255)
    print('PSNR: ' + '%.4f' % skimage.metrics.peak_signal_noise_ratio(I, de_image), end='\t\t')
    print('SSIM: ' + '%.4f' % skimage.metrics.structural_similarity(I, de_image, multichannel=(channels != 1)))

    plt.show()
    cv2.waitKey(0)
