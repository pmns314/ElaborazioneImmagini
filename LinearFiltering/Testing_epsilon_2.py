import os

from Guided_filter import guided_filter, guided_filter_OpenCV
from Utils import *

# evaluation of epsilon for each image in which is applied the noise altogether
if __name__ == '__main__':
    # assignment directory
    # directory = '../images/b&w'
    directory = '../images/rgb'

    # initialization parameters
    num_noise = 0  # it is calculated automatically
    num_image = 0  # it is calculated automatically
    num_iteration = 1
    noised_images = []

    # choice of noise or write None for analising all noise
    # noise = None
    noise = Noise.SALT_AND_PEPPER

    print("Applicazione dei rumnori alle immagini nella directory prefissata in corso...\n")
    # iteration images in the directory
    for filename in os.listdir(directory):
        # print("Immagine = %s" % filename)
        f = os.path.join(directory, filename)

        # Read Image B&W
        # I = cv2.imread(f, -1)
        # Read Image RGB
        I = cv2.imread(f)

        # BRG to RGB
        if get_channels_number(I) != 1:
            I = bgr2rgb(I)

        # show_image(I, "immagine")

        # iteration noises in order to apply it on the images
        num_noise = 0
        for type_noise in Noise:
            # print("Noise = %s" % type_noise.value)
            if noise is not None:
                if type_noise is noise:
                    noised_images.append(add_noise(I, type_noise))
                    num_noise += 1  # count the number of noises
            else:
                noised_images.append(add_noise(I, type_noise))
                num_noise += 1  # count the number of noises
        # count the number of images
        num_image += 1
    print("Rumore applicato per ogni immagine correttamente!\n")

    print("Valutazione complessiva per ogni epsilon in corso...\n")
    # iteration of epsilon
    for i in np.arange(0.01, 5.81, 0.2):
        # initialization MSE for all images in the directory, all noises, but for each epsilon
        m = 0  # used for guided filter with mean and variance
        m_2 = 0  # used for guided filter with OpenCV

        # initialization PSNR for all images in the directory, all noises, but for each epsilon
        p = 0  # used for guided filter with mean and variance
        p_2 = 0  # used for guided filter with OpenCV

        # initialization SSIM for all images in the directory, all noises, but for each epsilon
        s = 0  # used for guided filter with mean and variance
        s_2 = 0  # used for guided filter with OpenCV

        # iteration images with noise which, individually, are intended both as images with noise and guided images
        for I in noised_images:
            # sum of each MSE, PSNR and SSIM in order to evaluate the mean of them
            for j in range(0, num_iteration):
                # application filters
                I_fil_guid = im2double(guided_filter(I, I, epsil=i))
                I_fil_guid_OpenCV = im2double(guided_filter_OpenCV(I, I, epsil=i))
                # sum of indexes
                m += evaluate(im2double(I), im2double(I_fil_guid), False if get_channels_number(I) == 1 else True)[
                    0]
                m_2 += evaluate(im2double(I), im2double(I_fil_guid_OpenCV),
                                False if get_channels_number(I) == 1 else True)[0]
                p += evaluate(im2double(I), im2double(I_fil_guid), False if get_channels_number(I) == 1 else True)[
                    1]
                p_2 += evaluate(im2double(I), im2double(I_fil_guid_OpenCV),
                                False if get_channels_number(I) == 1 else True)[1]
                s += evaluate(im2double(I), im2double(I_fil_guid), False if get_channels_number(I) == 1 else True)[
                    2]
                s_2 += evaluate(im2double(I), im2double(I_fil_guid_OpenCV),
                                False if get_channels_number(I) == 1 else True)[2]
        # results
        print(
            "Numero di rumori applicati: " + str(
                num_noise) + "\nNumero di immagini su cui è stata effettuate la valutazione: " + str(
                num_image) + "\nNumero di iterazioni effettuate: " + str(num_iteration))
        n = num_noise * num_image * num_iteration
        print("Epsilon = %.2f" % i)
        print("Mean and variance    =   MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f" % (m / n, p / n, s / n))
        print("OpenCV               =   MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f\n" % (m_2 / n, p_2 / n, s_2 / n))
    # end
    print("Termine valutazione.")
