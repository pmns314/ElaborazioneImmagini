from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import numpy as np
import skimage as sk


def print_SSIM(image_true, image_test, message, multichannel=False, dimcut=10, prin=False):
    measure = structural_similarity(image_true[dimcut:-dimcut, dimcut:-dimcut],
                                    image_test[dimcut:-dimcut, dimcut:-dimcut], multichannel=multichannel)
    if prin:
        print(message, measure)
    return measure


def print_PSNR(image_true, image_test, message, dimcut=10, prin=False, multichannel=False):
    if not multichannel:
        measure = peak_signal_noise_ratio(image_true[dimcut:-dimcut, dimcut:-dimcut],
                                          image_test[dimcut:-dimcut, dimcut:-dimcut])
    else:
        measure = PSNR_RGB(image_true, image_test)

    if prin:
        print(message, measure)
    return measure


def PSNR_RGB(image_true, image_test):
    row, col = image_true.shape[0], image_true.shape[1]

    mse_R = mean_squared_error(image_true[:, :, 0], image_test[:, :, 0])
    mse_G = mean_squared_error(image_true[:, :, 1], image_test[:, :, 1])
    mse_B = mean_squared_error(image_true[:, :, 2], image_test[:, :, 2])

    mse_R = np.sum(mse_R) / row * col
    mse_G = np.sum(mse_G) / row * col
    mse_B = np.sum(mse_B) / row * col

    mse = (mse_R + mse_G + mse_B) / 3
    return 10 * np.log10(1 / mse)
