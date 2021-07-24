from Utils import *
from LinearFiltering.Guided_filter import guided_filter
from TransformedFiltering.wavelet_denoising import denoise_image
from NonLinearFiltering.anysotropic import aniso, anisoRGB


def images_comparation(I, I_noise, linear, non_linear, transformed, type_noise):
    """ Plots the images passed and computes the mse, psnr and ssim indexes"""
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("Original", fontweight="bold")
    plt.imshow(I, cmap='gray')

    fig.add_subplot(1, 2, 2)
    plt.title("Added Noise: " + type_noise, fontweight="bold")
    plt.imshow(I_noise, cmap='gray')

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.title("Linear", fontweight="bold")
    plt.imshow(linear, cmap='gray')
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I, linear, False if get_channels_number(I) == 1 else True)), loc="left")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(1, 3, 2)
    plt.title("Non Linear ", fontweight="bold")
    plt.imshow(non_linear, cmap='gray')
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I, non_linear, False if get_channels_number(I) == 1 else True)), loc="left")
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(1, 3, 3)
    plt.title("Transformed", fontweight="bold")
    plt.imshow(transformed, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I, transformed, False if get_channels_number(I) == 1 else True)), loc="left")


if __name__ == '__main__':
    # Reading image
    I = cv2.imread('./images/rgb/cat.jpg', -1)
    type_noise = 'gauss'
    I_noise = add_noise(I, type_noise)
    I = np.uint16(I)

    # Denoising
    linear = guided_filter(I_noise, I_noise)
    transformed = denoise_image(I_noise)
    non_linear = anisoRGB(I_noise, 5, 5, 5, 0.3, 0.3, 0.3)

    images_comparation(bgr2rgb(I), bgr2rgb(I_noise), bgr2rgb(linear), bgr2rgb(non_linear),
                       bgr2rgb(transformed), type_noise)
    plt.show()
