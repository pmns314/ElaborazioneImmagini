import pywt

from Denoising_Wavelet_rgb import *

if __name__ == '__main__':
    I = cv2.imread('../images/rgb/peppers.png', -1)
    I_noise = skimage.util.random_noise(I, 'gaussian')
    display(I_noise, 'Original with Noise')

    I_noise = skimage.img_as_float(I_noise)


    l = [*pywt.wavelist(kind='discrete')]
    print(l)

    for wname in l:
        levels = pywt.dwtn_max_level(I_noise.shape[:-1], wname)
        channels = get_channels_number(I_noise)
        I = np.uint16(I)
        de_image = denoise(I_noise, wname, levels, channels, (True, True))
        if I.shape != de_image.shape:
            de_image = de_image[:-1, :]
        de_image = skimage.img_as_uint(de_image / 255)
        print("Wname: ", wname, end='\t')
        print('PSNR: ' + '%.4f' % skimage.metrics.peak_signal_noise_ratio(I, de_image), end='\t\t')
        print('SSIM: ' + '%.4f' % skimage.metrics.structural_similarity(I, de_image, multichannel=(channels != 1)))

        plt.show()
    cv2.waitKey(0)