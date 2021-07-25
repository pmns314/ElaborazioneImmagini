""" Mansi Paolo mat. 0622701542"""

import math
import warnings
import pywt
from Utils import *

DEFAULT_LEVEL_NUMBER = 4
DEFAULT_WNAME = 'db10'
DEFAULT_MODE = 'neigh'
DEFAULT_NEIGH_DIM = 3


def show_image_from_coefficients(coeffs, title=None):
    """ Given the coefficients, it shows the wavelet decomposition"""
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
    """ Defines the subplots in which decompose the image and then shows the wavelet transform """
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
    """ Computes the Universal Threshold used for the denoising

    :param coeff - the coefficients of the wavelet decomposition
    :returns thr - the universal threshold
    """
    A = np.array([np.concatenate(([*coeff]), axis=None)])
    n = ((coeff[0][1]).shape[0])
    X = A[:, n ** 2 + 1:]
    # Calculate sigma = MAD/0.6745
    sigma = np.median(np.abs(X)) / 0.6745
    # Calculate universal threshold
    thr = sigma * math.sqrt(2 * math.log((n * 2) ** 2))
    return thr


def denoising_coefficients(coeff, mode, dim_neigh):
    """ Performs the denoising of the wavelets coefficients, either with the universal threshold or with  neighbouring

    :param coeff - the coefficients of the wavelet decomposition
    :param mode - specifies if using the universal threshold or the neighbouring
    :param dim_neigh - in case of the neighbouring, it defines the dimension of the window of the neighbourhood
    """
    warnings.filterwarnings("ignore")
    # Get Threshold
    univ_thr = get_universal_threshold(coeff)

    if mode == 'univ':
        for i in range(1, len(coeff)):
            for imm in coeff[i]:
                # set to 0 the values under the universal threshold
                imm[abs(imm) < univ_thr] = 0

    elif mode == 'neigh':
        mask = np.ones([dim_neigh, dim_neigh])
        for c in range(1, len(coeff)):
            denoised_coeff = [0, 0, 0]
            for i in range(0, 3):
                imm = coeff[c][i]
                # Perform the sum of the elements of the window through the correlation
                S = cv2.filter2D(imm*imm, -1, mask, borderType=cv2.BORDER_REPLICATE)
                # Calculates the shrinking factor
                B = 1 - (univ_thr ** 2 / S)
                B[B < 0] = 0
                # Update the value of the pixel
                denoised_coeff[i] = imm * B
            coeff[c] = tuple(denoised_coeff)
    else:
        raise Exception('Mode not supported')
    return coeff


def _denoising(image, wname, levels, ch, mode, dim_neigh, show):
    """ Performs the wavelet decomposition of the image passed, the denoising of the of those coefficients and then
     the reconstruction of the image

     :param image - the image to be denoised
     :param wname - the name of the wavelet used to decompose
     :param levels - the number of levels of the decomposition
     :param ch - the current channel of the image
     :param mode - the mode of the thresholding
     :param dim_neigh - the dimension of the neighbourhood
     :param show - a boolean flag. if True, it shows the Wavelet decomposition
     """
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


def wavelet_denoising(image, wname=DEFAULT_WNAME, levels=None, mode=DEFAULT_MODE, dim_neigh=DEFAULT_LEVEL_NUMBER,
                      show=False):
    """
    Performs the denoising of the image passed through the thresholding of the wavelet decomposition

    :param image: the image to be denoised
    :param wname: the name of the wavelet used to decompose. By default it uses the 'db10' wavelet
    :param levels: the number of levels of the decomposition. if it's not chosen, it will be assigned 5 if possible, or
                    else the maximum number of levels
    :param mode: the mode of the thresholding. By default it's the neighbouring threshold
    :param dim_neigh: the dimension of the neighbourhood. By default it's a 3x3 window
    :param show: a boolean flag. if True, it shows the Wavelet decomposition
    :returns: the denoised image
    """
    channels = get_channels_number(image)
    # if level is not specified, it is set to the default number
    if levels is None:
        levels = pywt.dwtn_max_level(image.shape[:-1], wname)
        levels = DEFAULT_LEVEL_NUMBER if levels >= DEFAULT_LEVEL_NUMBER else levels
    elif levels == 'max':
        levels = pywt.dwtn_max_level(image.shape[:-1], wname)

    if channels == 1:
        # if the image is b&w the denoising is done on the only existing channel
        im = _denoising(image, wname, levels, -1, mode, dim_neigh, show)
    else:
        # if the image is rgb, the denoising is done on each channel singularly
        im = np.zeros([*image.shape])
        for c in range(channels):
            res = _denoising(image[:, :, c], wname, levels, c + 1, mode, dim_neigh, show)
            # Dimensions adjusting
            if res.shape[0] != image.shape[0]:
                res = res[:-1, :]
            if res.shape[1] != image.shape[1]:
                res = res[:, :-1]
            im[:, :, c] = res
    return im


def plot_images_test(I_original, I_noisy, noisy, I_filtred_univ, I_filtred_neigh):
    """ Plots the four images passed and computes the mse, psnr and ssim indexes"""
    if get_channels_number(I_original) != 1:
        I_original = bgr2rgb(I_original)
        I_noisy = bgr2rgb(I_noisy)
        I_filtred_univ = bgr2rgb(I_filtred_univ)
        I_filtred_neigh = bgr2rgb(I_filtred_neigh)
    # creazione del plot
    fig = plt.figure()
    # plot immagine originale
    fig.add_subplot(2, 2, 1)
    plt.title("Originale", fontweight="bold")
    plt.imshow(I_original, cmap='gray')
    plt.axis("off")
    # plot immagine con rumore
    fig.add_subplot(2, 2, 2)
    plt.title("Rumore " + noisy.value, fontweight="bold")
    plt.imshow(I_noisy, cmap='gray')
    plt.axis("off")
    # plot immagine filtrata
    fig.add_subplot(2, 2, 3)
    plt.title("univ", fontweight="bold")
    plt.imshow(I_filtred_univ, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I_original, I_filtred_univ, False if get_channels_number(I) == 1 else True)), loc="left")

    # plot immagine filtrata con OpenCV
    fig.add_subplot(2, 2, 4)
    plt.title("Neigh", fontweight="bold")
    plt.imshow(I_filtred_neigh, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I_original, I_filtred_neigh, False if get_channels_number(I) == 1 else True)), loc="left")


if __name__ == '__main__':
    print("------------------------- Black and white image --------------------")
    I = cv2.imread('../images/b&w/totem-poles.tif', -1)
    type_noise = Noise.GAUSSIAN
    I_noise = add_noise(I, type_noise)

    # Denoise
    de_image_neigh = wavelet_denoising(I_noise, mode='neigh', show=False)
    de_image_univ = wavelet_denoising(I_noise, mode='univ', show=False)

    plot_images_test(I, I_noise, type_noise, de_image_univ, de_image_neigh)

    print("Univ:\tMSE: %.2f   - PSNR: %.2f  - SSIM: %.2f" % evaluate(I, de_image_univ,
                                                                     False if get_channels_number(
                                                                         I) == 1 else True))

    print("Neigh:\tMSE: %.2f   - PSNR: %.2f   - SSIM: %.2f" % evaluate(I, de_image_neigh,
                                                                       False if get_channels_number(
                                                                           I) == 1 else True))
    print("------------------------- RGB image --------------------------")
    I = cv2.imread('../images/rgb/peppers.png')
    type_noise = Noise.GAUSSIAN
    I_noise = add_noise(I, type_noise)

    # Denoise
    de_image_neigh = wavelet_denoising(I_noise, mode='neigh', show=False)
    de_image_univ = wavelet_denoising(I_noise, mode='univ', show=False)

    plot_images_test(I, I_noise, type_noise, de_image_univ, de_image_neigh)

    print("Univ:\tMSE: %.2f   - PSNR: %.2f   - SSIM: %.2f" % evaluate(I, de_image_univ,
                                                                      False if get_channels_number(I) == 1 else True))

    print("Neigh:\tMSE: %.2f   - PSNR: %.2f   - SSIM: %.2f" % evaluate(I, de_image_neigh,
                                                                       False if get_channels_number(I) == 1 else True))
    plt.show()
