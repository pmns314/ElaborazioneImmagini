from wavelet_denoising import *
from Utils import *

if __name__ == '__main__':

    # Read Image
    I = cv2.imread('../images/rgb/peppers.png')
    type_noise = Noise.GAUSSIAN

    # Add Noise
    I_noise = add_noise(I, type_noise)

    for wname in pywt.wavelist(kind="discrete"):
        de_image_neigh = wavelet_denoising(I_noise, wname=wname)
        print(wname + ":\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % (
            evaluate(I, de_image_neigh, False if get_channels_number(I) == 1 else True)))

    plt.show()
