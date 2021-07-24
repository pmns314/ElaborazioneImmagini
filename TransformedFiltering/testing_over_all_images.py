from wavelet_denoising import *
import os
import warnings
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from LinearFiltering.Guided_filter import im2double


def evaluate(imageA, imageB):
    m = mean_squared_error(imageA, imageB)
    p = cv2.PSNR(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    return m, p, s


def show_images(images, titles, channels=1):
    # creazioen del plot
    fig = plt.figure()
    n = len(images)
    # plot immagine originale
    for i in range(1, n + 1):
        fig.add_subplot(n//2, n//2, i)
        plt.title(titles[i - 1])
        image = cv2.normalize(images[i - 1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if channels == 1:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image[:, :, 2::-1])
        plt.axis("off")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # assign directory
    directory = '../images/b&w'

    # iterate over files in the directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        # Read Image
        I = cv2.imread(f, -1)

        # Add Noise
        I_noise = skimage.util.random_noise(I, 's&p')
        channels = get_channels_number(I_noise)

        # Definizione Parametri
        wname = 'db3'
        levels = pywt.dwtn_max_level(I_noise.shape[:-1], wname)

        # Denoising
        de_image_neigh = denoise_image(I_noise, wname, levels, channels, mode='neigh')
        de_image = denoise_image(I_noise, wname, levels, channels, mode='univ')

        # Agreement dimensions of the images
        if de_image.shape != I.shape:
            de_image = de_image[:I.shape[0], :I.shape[1]]
            de_image_neigh = de_image_neigh[:I.shape[0], :I.shape[1]]

        vals_neigh = evaluate(im2double(I), im2double(de_image_neigh))
        vals_univ = evaluate(im2double(I), im2double(de_image))
        print("Image: ", filename)
        print("\t\t\t\t\t\t MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f" % vals_neigh)
        print("\t\t\t\t\t\t MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f" % vals_univ)
        show_images([I, I_noise, de_image, de_image_neigh], ["Original", "Noise", "Univ", "Neigh"], channels)
    plt.show()
