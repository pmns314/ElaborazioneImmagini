""" Salvati Vincenzo mat. 0622701550"""
from Utils import *


def compare_images(I_original, I_noisy, noisy, I_filtred, I_filtred_CV):
    """ Plots the four images passed and computes the mse, psnr and ssim indexes"""
    I_original = im2double(I_original)
    # creation plot
    fig = plt.figure()
    # plot original image
    fig.add_subplot(2, 2, 1)
    plt.title("Originale", fontweight="bold")
    plt.imshow(I, cmap='gray')
    plt.axis("off")
    # plot noised image
    fig.add_subplot(2, 2, 2)
    plt.title("Rumore " + noisy, fontweight="bold")
    plt.imshow(I_noisy, cmap='gray')
    plt.axis("off")
    # plot filtered image
    fig.add_subplot(2, 2, 3)
    plt.title("Filtro (med e var)", fontweight="bold")
    plt.imshow(I_filtred, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I_original, I_filtred, False if get_channels_number(I) == 1 else True)), loc="left")

    # plot filtered image with OpenCV
    fig.add_subplot(2, 2, 4)
    plt.title("Filtro OpenCV", fontweight="bold")
    plt.imshow(I_filtred_CV, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I_original, I_filtred_CV, False if get_channels_number(I) == 1 else True)), loc="left")


def guided_filter(I_guid, I_noisy, epsil=0.01, N=3):
    """ Implements the Guided filter with mean and variance

    :param I_guid - Guide Image
    :param I_noisy - Image to be filtered
    :param epsil - the weight that guides how much following the Guide Image's edges
    :param N - dimension of the neighbourhood window

    :return the filtered image
    """
    h = np.ones(N) / (pow(N, 2))
    # perform mean e correlation
    mean_I = cv2.filter2D(I_guid, -1, h, borderType=cv2.BORDER_REPLICATE)
    mean_p = cv2.filter2D(I_noisy, -1, h, borderType=cv2.BORDER_REPLICATE)
    corr_I = cv2.filter2D(I_guid * I_guid, -1, h, borderType=cv2.BORDER_REPLICATE)
    corr_Ip = cv2.filter2D(I_guid * I_noisy, -1, h, borderType=cv2.BORDER_REPLICATE)
    # perform variance e covariance
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # perform a and b
    a = cov_Ip / (var_I + epsil)
    b = mean_p - a * mean_I
    # perform mean of the assumed values by a and b
    mean_a = cv2.filter2D(a, -1, h, borderType=cv2.BORDER_REPLICATE)
    mean_b = cv2.filter2D(b, -1, h, borderType=cv2.BORDER_REPLICATE)
    # perform linear function
    return mean_a * I_guid + mean_b


def guided_filter_OpenCV(I_guid, I_noisy, epsil=0.01, nhoodSize=3):
    """ Implements the Guided filter with OpenCV functions

    :param I_guid - Guide Image
    :param I_noisy - Image to be filtered
    :param epsil - the weight that guides how much following the Guide Image's edges
    :param nhoodSize - dimension of the neighbourhood window

    :return the filtered image
    """
    I_guid_uint8 = cv2.normalize(src=I_guid, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    I_noisy_uint8 = cv2.normalize(src=I_noisy, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if I_noisy.ndim == 2:
        return cv2.ximgproc.guidedFilter(np.uint8(I_guid_uint8), np.uint8(I_noisy_uint8), nhoodSize, epsil, -1)
    else:
        return cv2.ximgproc.guidedFilter(np.uint8(I_guid_uint8), np.uint8(I_noisy_uint8), nhoodSize, epsil)


def guided_denoising(I_noise, epsil=0.01, nhoodSize=3):
    return guided_filter(I_noise, I_noise, epsil, nhoodSize)


if __name__ == "__main__":

    I = cv2.imread('../images/b&w/cameraman.tif', -1)
    for type_noise in Noise:
        I_noise = add_noise(I, type_noise)
        # show_image(I_noise, 'Immagine con rumore ' + type_noise.value)
        I_fil_guid = im2double(guided_filter(I_noise, I_noise))
        I_fil_guid_OpenCV = im2double(guided_filter_OpenCV(I_noise, I_noise))
        compare_images(I, I_noise, type_noise.value, I_fil_guid, I_fil_guid_OpenCV)

    I = cv2.imread('../images/rgb/peppers.png')
    I = bgr2rgb(I)
    for type_noise in Noise:
        I_noise = add_noise(I, type_noise)
        # show_image(I_noise, 'Immagine con rumore ' + type_noise.value)
        I_fil_guid = im2double(guided_filter(I_noise, I_noise))
        I_fil_guid_OpenCV = im2double(guided_filter_OpenCV(I_noise, I_noise))
        compare_images(I, I_noise, type_noise.value, I_fil_guid, I_fil_guid_OpenCV)

    plt.show()
