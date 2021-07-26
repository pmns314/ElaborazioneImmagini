from wavelet_denoising import *
from Utils import *

if __name__ == '__main__':

    I = cv2.imread('../images/rgb/peppers.png')
    t = 0
    NUM_IT = 50
    names = [[] for _ in range(4)]
    values = [[] for _ in range(4)]
    for type_noise in Noise:
        print(type_noise.value)
        I_noise = []
        for i in range(NUM_IT):
            I_noise.append(add_noise(I, type_noise))

        for wname in pywt.wavelist(kind="discrete"):
            a = 0
            for x in range(NUM_IT):
                de_image_neigh = wavelet_denoising(I_noise[x], wname=wname)
                vals = evaluate(I, de_image_neigh, False if get_channels_number(I) == 1 else True)
                # print(wname + ":\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % (
                #     vals))
                a += vals[2]
            names[t].append(wname)
            values[t].append(float("%.2f" % (a / NUM_IT)))
        t += 1
    np.savetxt('peppers-' + str(NUM_IT) + '.csv', [_ for _ in zip(names[0], *values)], delimiter=';', fmt='%s')
