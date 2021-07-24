from wavelet_denoising import *
from Utils import *


def plot_images_neighbourhood_test(I_original, I_noisy, noisy, I_filtred, N):
    """ Plots the images passed and computes the mse, psnr and ssim indexes"""

    fig = plt.figure()

    fig.add_subplot(131)
    plt.title("Originale", fontweight="bold")
    plt.imshow(bgr2rgb(I), cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(132)
    plt.title("Rumore " + noisy.value, fontweight="bold")
    plt.imshow(bgr2rgb(I_noisy), cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    fig.add_subplot(133)
    plt.title("N: " + str(N), fontweight="bold")
    plt.imshow(bgr2rgb(I_filtred), cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I_original, I_filtred, False if get_channels_number(I) == 1 else True)), loc="left")


if __name__ == '__main__':
    # Read Image
    I = cv2.imread('../images/rgb/peppers.png')
    type_noise = Noise.GAUSSIAN

    # Add Noise
    I_noise = add_noise(I, type_noise)

    # Definizione Parametri
    for N in range(1, 10):
        de_image_neigh = wavelet_denoising(I_noise, dim_neigh=N)
        # plot_images_neighbourhood_test(I, I_noise, type_noise, de_image_neigh, N)
        print("N = " + str(N) + ":\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % (
            evaluate(I, de_image_neigh, False if get_channels_number(I) == 1 else True)))

    plt.show()
