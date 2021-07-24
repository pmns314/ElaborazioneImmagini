"Dario Cival mat: "

from anysotropic import *
from Utils import *


def conduction_function(image, image_noise, range_it, g1, g2, kappa, option1, option2):
    for i in range_it:
        an = aniso(image_noise, i, kappa, option=1, neightborhood=option1)
        meas = evaluate(image, an, False if get_channels_number(image) == 1 else True)
        g1.append(meas[1])

        an2 = aniso(image_noise, i, kappa, option=2, neightborhood=option2)
        meas2 = evaluate(image, an2, False if get_channels_number(image) == 1 else True)
        g2.append(meas2[1])


def valutation_automatic_parameters(image, sigmas, noise='gaussian', option=1, neightborhood='minimal'):
    """
     visualizza il comportamento dei parametri automatici utilizzati dal filtro anisotropico fissato un certo rumore
    :param image: immagine
    :param sigmas:lista di deviazioni standard
    :param noise:
    :param option: opzioni del filtro anisotropico
    :param neightborhood: direzioni considerate dal filtro anisotropico
    :return:
    """

    I = image.copy()
    if noise == 'gaussian':
        for sigma in sigmas:
            im = add_noise(I, 'gauss', sigma_g=sigma, mean_g=0)
            kappa, mum_it = automatic_parameters(im, option, neightborhood)
            print("sigma: " + str(sigma), "thresh: " + str(kappa), "it: " + str(mum_it),
                  "estimation noise: " + str(estimate_noise(im)))


def valutation_threshold(image, num_iteration, option=1, neightborhood='minimal'):
    """
    visualizza le misure di PSNR e SSIM riguardanti il filtraggio anisotropico per ogni algoritmo di threshold implementato
    :param image: immagine
    :param num_iteration: numero di iterazioni
    :param option: opzioni del filtro anisotropico
    :param neightborhood: direzioni considerate dal filtro anisotropico
    :return:
    """
    k1 = get_gradient_threshold(image)
    k2 = get_gradient_thresh_MAD(image)
    k3 = get_gradient_thresh_morpho(image)
    an1 = aniso(image, num_iteration, k1, option, neightborhood)
    an2 = aniso(image, num_iteration, k2, option, neightborhood)
    an3 = aniso(image, num_iteration, k3, option, neightborhood)

    # VALUTAZIONE THRESHOLD CON APPROCCIO DI PERONA-MALIK
    print("PERONA-MALIK:    MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f " % evaluate(image, an1, False if get_channels_number(
        image) == 1 else True))
    # VALUTAZIONE THRESHOLD CON MAD
    print("MAD:             MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f " % evaluate(image, an2, False if get_channels_number(
        image) == 1 else True))
    # VALUTAZIONE THRESHOLD MORPHO
    print("MORPHO:          MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f " % evaluate(image, an3, False if get_channels_number(
        image) == 1 else True))


# GRAY SCALE IMAGE
#
# cameraman = sk.img_as_float(cv2.imread('images/cameraman.jpg', 0))
# sigmas = np.arange(0, 0.5, 0.025)
#
sigmas = [0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 0.7, 1]
# cameraman_noise = add_gaussian(cameraman, 0, sigmas[2])
# kappa = get_gradient_thresh_morpho(cameraman_noise)
#
# kappas, num_iteration = automatic_parameters(cameraman_noise, 1, 'minimal')
# print(kappas, num_iteration)
# kappa = kappas[0]
# an2 = aniso(cameraman_noise, num_iteration, kappas, 1, 'minimal', single_threshold=False)
#
# print_PSNR(cameraman, an2, 'PSNR an2', prin=True)
# print_SSIM(cameraman, an2, 'SSIM an2', prin=True)
# show_image(an2, 'aniso PERONA-MALIK', True)
# show_image(cameraman_noise, 'noise', True)
# # valutation_threshold(cameraman_noise, 5, option=1, neightborhood='minimal')
# an = aniso(cameraman_noise, 5, get_gradient_thresh_MAD(cameraman_noise))
# print_PSNR(cameraman, an, 'PSNR MAD', prin=True)
# print_SSIM(cameraman, an, 'SSIM MAD', prin=True)
# show_image(an, 'aniso MAD', True)
# # valutation_automatic_parameters(cameraman_noise, sigmas, noise='gaussian', option=1, neightborhood='minimal')

# COLOR IMAGE
I = sk.img_as_float(cv2.imread('../images/rgb/peppers.png'))
I = I[:, :, 2::-1]
I_noise = add_noise(I, 'gauss', sigma_g=sigmas[3], mean_g=0)
show_image(I_noise, 'original')
k_R, k_G, k_B, it_R, it_G, it_B = automatic_parameters_RGB(I_noise, 1, 'minimal')

print(it_R, it_G, it_B)

sp = anisoRGB(I_noise, it_R, it_G, it_B, k_R[0], k_G[0], k_B[0])
sp2 = anisoRGB(I_noise, it_R, it_G, it_B, k_R[0], k_G[0], k_B[0])
print("Aniso RGB:    MSE: %.2f \t PSNR: %.2f \t SSIM: %.2f " % evaluate(I, sp,
                                                                        False if get_channels_number(I) == 1 else True))
show_image(sp, 'sp')

plt.show()
