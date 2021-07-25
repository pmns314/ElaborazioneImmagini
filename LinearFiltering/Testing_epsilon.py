import os

from Guided_filter import guided_filter, guided_filter_OpenCV
from Utils import *

# Evaluation of epsilon for each image in which is applied the noise
if __name__ == '__main__':
    # assignment directory
    directory = '../images/b&w'
    # directory = '../images/rgb'

    # initialization parameters
    num_image = 0  # it is calculated automatically
    num_iteration = 1

    # iteration images in the directory
    for filename in os.listdir(directory):
        print("Immagine = %s" % filename)
        f = os.path.join(directory, filename)

        # Read Image B&W
        I = cv2.imread(f, -1)
        # Read Image RGB
        # I = cv2.imread(f)

        # BRG to RGB
        if get_channels_number(I) != 1:
            I = bgr2rgb(I)

        # show_image(I, "immagine")

        # iteration noises in order to apply it on the images
        for type_noise in Noise:
            print("Noisy = %s" % type_noise.value)
            I_noise = add_noise(I, type_noise)
            I_guid = I_noise

            # iteration of epsilon
            for i in np.arange(0.01, 5.81, 0.2):
                # initialization MSE for all images in the directory, all noises, but for each epsilon
                m = 0;  # used for guided filter with mean and variance
                m_2 = 0;  # used for guided filter with OpenCV

                # initialization PSNR for all images in the directory, all noises, but for each epsilon
                p = 0;  # used for guided filter with mean and variance
                p_2 = 0;  # used for guided filter with OpenCV

                # initialization SSIM for all images in the directory, all noises, but for each epsilon
                s = 0;  # used for guided filter with mean and variance
                s_2 = 0;  # used for guided filter with OpenCV

                # sum of each MSE, PSNR and SSIM in order to evaluate the mean of them
                for j in range(0, num_iteration):
                    # application filters
                    I_fil_guid = im2double(guided_filter(I_guid, I_noise, epsil=i))
                    I_fil_guid_OpenCV = im2double(guided_filter_OpenCV(I_guid, I_noise, epsil=i))
                    # sum of indexes
                    m += evaluate(im2double(I), im2double(I_fil_guid), False if get_channels_number(I) == 1 else True)[
                        0]
                    m_2 += \
                        evaluate(im2double(I), im2double(I_fil_guid_OpenCV),
                                 False if get_channels_number(I) == 1 else True)[0]
                    p += evaluate(im2double(I), im2double(I_fil_guid), False if get_channels_number(I) == 1 else True)[
                        1]
                    p_2 += \
                        evaluate(im2double(I), im2double(I_fil_guid_OpenCV),
                                 False if get_channels_number(I) == 1 else True)[1]
                    s += evaluate(im2double(I), im2double(I_fil_guid), False if get_channels_number(I) == 1 else True)[
                        2]
                    s_2 += \
                        evaluate(im2double(I), im2double(I_fil_guid_OpenCV),
                                 False if get_channels_number(I) == 1 else True)[2]
                # results
                print("Numero di iterazioni effettuate: " + str(num_iteration))
                print("Epsilon = %.2f" % (i))
                print("Mean and variance    =   MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f" % (
                    m / num_iteration, p / num_iteration, s / num_iteration))
                print("OpenCV               =   MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f\n" % (
                    m_2 / num_iteration, p_2 / num_iteration, s_2 / num_iteration))
    # end
    print("Termine valutazione.")
