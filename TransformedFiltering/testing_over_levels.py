from wavelet_denoising import *
from Utils import *


def plot_images_levels_test(I_original, I_noisy, noisy, I_filtered, level):
    """ Plots the images passed and computes the mse, psnr and ssim indexes"""

    fig = plt.figure()

    fig.add_subplot(131)
    plt.title("Originale", fontweight="bold")
    plt.imshow(bgr2rgb(I_original), cmap='gray')
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
    plt.title("level: " + str(level), fontweight="bold")
    plt.imshow(bgr2rgb(I_filtered), cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (
        evaluate(I_original, I_filtered, False if get_channels_number(I_original) == 1 else True)), loc="left")


if __name__ == '__main__':

    # Read Image
    image = cv2.imread('../images/rgb/cat.jpg')
    for type_noise in Noise:
        print("\n\n Noise: " + type_noise.value)
        # Add Noise
        I_noise = add_noise(image, type_noise)

        # Definizione Parametri
        wname = "db10"
        levels_tot = pywt.dwtn_max_level(I_noise.shape[:-1], wname)

        for levels in range(1, levels_tot + 1):
            de_image_neigh = wavelet_denoising(I_noise, wname=wname, levels=levels)
            plot_images_levels_test(image, I_noise, type_noise, de_image_neigh, levels)
            print("level = " + str(levels) + ":\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % (
                evaluate(image, de_image_neigh, False if get_channels_number(image) == 1 else True)))
        plt.show()
