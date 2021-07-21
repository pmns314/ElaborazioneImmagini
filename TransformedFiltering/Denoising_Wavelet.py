import math

import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.metrics


def show_image_from_coefficients(coeffs, shape, levels, title=None):
    rows, cols = shape
    C = np.ones([rows + int(np.ceil(math.log2(rows) / 2)), cols + int(np.ceil(math.log2(cols) / 2))])
    dim_rows, dim_cols = coeffs[0].shape
    for i in range(levels + 1):
        if i == 0:
            C[:dim_rows, :dim_cols] = coeffs[i]
        else:
            dim_rows, dim_cols = coeffs[i][0].shape
            C[:dim_rows, dim_cols:dim_cols * 2] = coeffs[i][0]
            C[dim_rows:dim_rows * 2, :dim_cols] = coeffs[i][1]
            C[dim_rows:dim_rows * 2, dim_cols:dim_cols * 2] = coeffs[i][2]
    plt.figure()
    plt.imshow(C, vmin=-0.5, vmax=2.5, cmap='jet')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    return


def get_threshold(image, wname):
    coeff = pywt.wavedec2(image, wname, 1)
    A = np.array([np.concatenate(([*coeff]), axis=None)])
    n = (((coeff[0][1]).shape[0]))
    X = A[:, n ** 2 + 1:]
    sigma = np.median(np.abs(X)) / 0.6745
    thr = sigma * math.sqrt(2 * math.log((n * 2) ** 2))
    return thr


def denoise(image, wname, levels, flag_show=None, title=None):
    # Wavelet Transform
    coeff = pywt.wavedec2(image, wname, level=levels)
    # Plot Wavelet Transform
    if flag_show is not None and flag_show[0] is True:
        if title is None:
            t1 = 'Pre Denoising'
        else:
            t1 = title[0]
        show_image_from_coefficients(coeff, image.shape, levels, t1)
    # Get Threshold
    thr = get_threshold(image, wname)
    # Denoising Wavelet Transform
    for i in range(1, len(coeff)):
        for imm in coeff[i]:
            imm[abs(imm) < thr] = 0
    # Plot Wavelet Tranform Denoised
    if flag_show is not None and flag_show[1] is True:
        if title is None:
            t2 = 'Post Denoising'
        else:
            t2 = title[1]
        show_image_from_coefficients(coeff, (256, 256), levels, t2)
    # Reconstruct Image
    return pywt.waverec2(coeff, wname)

if __name__ == '__main__':
    I = cv2.imread('../images/b&w/cameraman.tif', -1)
    I_noise = skimage.util.random_noise(I, 'gaussian')
    cv2.imshow('Noise', I_noise)

    # Trasformata Wavelet
    wname = 'db3'
    I_noise = skimage.img_as_float(I_noise)
    levels = pywt.dwtn_max_level(I_noise.shape, wname)
    for i in range(1, levels+1):
        print('Level : ', i, '\n', end='\t\t')
        # Denoise
        de_image = denoise(I_noise, wname, i, (False, False), title=['Pre_Noise '+str(i), 'Post_denoise '+str(i)])
        plt.figure()
        plt.imshow(de_image, cmap='gray')
        plt.title('Denoised image with level= '+str(i))
        I = skimage.img_as_float64(I)
        print('PSNR: '+'%.4f' % skimage.metrics.peak_signal_noise_ratio(I, de_image), end='\t\t')
        print('SSIM: '+'%.4f' % skimage.metrics.structural_similarity(I, de_image))

    plt.show()