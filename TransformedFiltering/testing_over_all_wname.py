
from wavelet_denoising import *
import warnings
from testing_over_all_images import evaluate
from LinearFiltering.Guided_filter import im2double

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Read Image
    I = cv2.imread('../images/rgb/cat.jpg')

    # Add Noise
    I_noise = skimage.util.random_noise(I, 'gaussian')
    channels = get_channels_number(I_noise)

    # Definizione Parametri
    wnames = pywt.wavelist(kind='discrete')
    with open("cat.txt", "w") as output_file:
        for wname in wnames:
            levels = pywt.dwtn_max_level(I_noise.shape[:-1], wname)

            # Denoising
            de_image_neigh = denoise_image(I_noise, wname, levels, channels, mode='neigh')

            # Agreement dimensions of the images
            if de_image_neigh.shape != I.shape:
                de_image_neigh = de_image_neigh[:I.shape[0], :I.shape[1]]

            vals_neigh = evaluate(im2double(I), im2double(de_image_neigh))
            print("Wavelet: ", wname)
            print("\t\t\t\t\t\t MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f" % vals_neigh)

            print("Wavelet: ", wname, file=output_file)
            print("\t\t\t\t\t\t MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f" % vals_neigh, file=output_file)

        output_file.close()
    plt.show()
