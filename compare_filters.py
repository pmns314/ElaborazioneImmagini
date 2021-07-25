from Utils import *
from LinearFiltering.Guided_filter import guided_denoising
from TransformedFiltering.wavelet_denoising import wavelet_denoising
from NonLinearFiltering.anisotropic import anisotropic_denoising


def images_comparation(I, I_noise, linear, non_linear, transformed, type_noise):
    """ Plots the images passed and computes the mse, psnr and ssim indexes"""
    if get_channels_number(I) != 1:
        I = bgr2rgb(I)
        I_noise = bgr2rgb(I_noise)
        linear = bgr2rgb(linear)
        non_linear = bgr2rgb(non_linear)
        transformed = bgr2rgb(transformed)

    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.title("Original", fontweight="bold")
    plt.imshow(I, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(2, 2, 2)
    plt.title("Added Noise: " + type_noise.value, fontweight="bold")
    plt.imshow(I_noise, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(2, 3, 4)
    plt.title("Linear", fontweight="bold")
    plt.imshow(linear, cmap='gray')
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I, linear, False if get_channels_number(I) == 1 else True)), loc="left")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(2, 3, 5)
    plt.title("Non Linear ", fontweight="bold")
    plt.imshow(non_linear, cmap='gray')
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I, non_linear, False if get_channels_number(I) == 1 else True)), loc="left")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(2, 3, 6)
    plt.title("Transformed", fontweight="bold")
    plt.imshow(transformed, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I, transformed, False if get_channels_number(I) == 1 else True)), loc="left")


if __name__ == '__main__':
    # Reading image b&w
    I = cv2.imread('./images/b&w/cameraman.tif', -1)
    for type_noise in Noise:
        I_noise = add_noise(I, type_noise)
        # Denoising
        linear = guided_denoising(I_noise)
        transformed = wavelet_denoising(I_noise)
        non_linear = anisotropic_denoising(I_noise)
        images_comparation(I, I_noise, linear, non_linear, transformed, type_noise)

    I = cv2.imread('./images/rgb/peppers.png')
    for type_noise in Noise:
        I_noise = add_noise(I, type_noise)
        # Denoising
        linear = guided_denoising(I_noise)
        transformed = wavelet_denoising(I_noise)
        non_linear = anisotropic_denoising(I_noise)
        images_comparation(I, I_noise, linear, non_linear, transformed, type_noise)
    plt.show()


