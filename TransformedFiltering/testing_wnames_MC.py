from wavelet_denoising import *
from Utils import *

if __name__ == '__main__':

    # Read Image
    I = cv2.imread('../images/rgb/peppers.png')
    type_noise = Noise.SPECKLE

    I_noise = []
    for i in range(10):
        I_noise.append(add_noise(I, type_noise))
    # Add Noise
    l=[]
    for wname in pywt.wavelist(kind="discrete")[10:]:
        a = 0
        for x in range(10):
            de_image_neigh = wavelet_denoising(I_noise[x], wname=wname)
            vals = evaluate(I, de_image_neigh, False if get_channels_number(I) == 1 else True)
            # print(wname + ":\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % (
            #     vals))
            a+=vals[2]
        l.append((float("%.2f"%(a/10)), wname))

    print("\n\n\n\n")
    l = sorted(l, reverse=True)
    for i in l:
        print(i)
    plt.show()
