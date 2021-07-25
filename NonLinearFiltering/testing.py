from NonLinearFiltering.anisotropic import *
from Utils import *


# la funzione di conducibilità esponenziale dalle misure di PSNR, SSIM, e MSE  si è rivelata la più efficiente(le funzioni sono state testate su varie immagini)
def compare_conducibility_function(image_true, image_test, kappa, iter_range, neightborhood):
    """
    Confronta le funzioni di conduibilità, tramite MSE PSNR e SSIM, fissando la soglia del gradiente e variando il numero di iterazioni
    :param image_true: immagine a scala di grigio o RGB sui cui si vuole valutare le funzioni di conducibilità
    :param image_test: immagine di test
    :param kappa: soglia del grandiente usanta dalle funzioni di conducibilità
    :param iter_range: range di iterazione
    :param neightborhood:
     * Se neightborhood = 'minimal' il filtro anisotropico considera solo le variazioni dell'immagine lungo le direzioni {N,S,E,W}
     * Se neightborhood = 'maximal' il filtro anisotropico considera le variazioni dell'immagine lungo le direzioni {N,S,E,W,NE,NW,SE,SW}
    :return:
    """
    for x in iter_range:
        if get_channels_number(image_true) == 3:
            an = anisoRGB(image_test, x, x, x, kappa, kappa, kappa, option=1, neightborhood=neightborhood)
            an2 = anisoRGB(image_test, x, x, x, kappa, kappa, kappa, option=2, neightborhood=neightborhood)

            print("exponential:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an, True))
            print("quadratic:  \t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an2, True))
        else:
            an = aniso(image_test, x, kappa, option=1, neightborhood=neightborhood)
            an2 = aniso(image_test, x, kappa, option=2, neightborhood=neightborhood)
            print("exponential:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an, False))
            print("quadratic:  \t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an2, False))


# La scelta dell'algoritmo di soglia dipende molto dal tipo di elaborazione che si vuole ottenere sull' immagine, in generale algoritmo di soglia adattiva(PERONA-MALIK)  conserva meglio gli edge rispetto
# ai restanti due algoritmi, che causano  un maggior smoothing
def compare_threshold_techniques(image_true, image_test):
    """
    Confronta gli algoritmi di thresholding implemenati dal filtro anisotropico visualizzando  MSE PSNR e SSIM
    :param image_true: immagine a scala di grigio o RGB
    :param image_test: immagine di test
    :return:
    """
    number_it = 5
    if get_channels_number(image_true) == 3:
        # PERONA-MALIK
        k_R, k_G, k_B, it_R, it_G, it_B = automatic_parameters_RGB(image_test, 1, 'minimal')
        an_PER = anisoRGB(image_test, it_R, it_G, it_B, k_R, k_G, k_B, single_threshold=False)

        # MAD
        k_MAD_R, K_MAD_G, K_MAD_B = get_gradient_thresh_MAD_RGB(image_test)
        an_MAD = anisoRGB(image_test, number_it, number_it, number_it, k_MAD_R, K_MAD_G, K_MAD_B)

        # MORPHO
        k_MORPHO_R, k_MORPHO_G, k_MORPHO_B = get_gradient_thresh_morpho_RGB(image_test)
        an_MORPHO = anisoRGB(image_test, number_it, number_it, number_it, k_MORPHO_R, k_MORPHO_G, k_MORPHO_B)
        print("PERONA-MALIK:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_PER, True))
        print("MAD:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_MAD, True))
        print("MORPHO:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_MORPHO, True))

    else:
        # PERONA-MALIK
        kappas, it = automatic_parameters(image_test, 1, 'minimal')
        an_PER = aniso(image_test, it, kappas, single_threshold=False)
        # MAD
        k_MAD = get_gradient_thresh_MAD(image_test)
        an_MAD = aniso(image_test, number_it, k_MAD)
        # MORPHO
        k_MORPHO = get_gradient_thresh_morpho(image_test)
        an_MORPHO = aniso(image_test, number_it, k_MORPHO)
        print("PERONA-MALIK:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_PER, False))
        print("MAD:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_MAD, False))
        print("MORPHO:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_MORPHO, False))


# la scelta tra la modalità minimal e maximal non influisce di molto nel filtraggio
def compare_neightborhood(image_true, image_test, num_it, kappa):
    """
    Confronta le modalità del filtraggio anisotropico
    :param image_true: immagine vera
    :param image_test: immagine di test
    :param num_it: numero di iterazioni
    :param kappa: soglia del gradiente
    :return:
    """
    an_min = aniso(image_test, num_it, kappa, neightborhood='minimal')
    an_max = aniso(image_test, num_it, kappa, neightborhood='maximal')
    print("MINIMAL:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_min, False))
    print("MAXIMAL:\t\tMSE: %.2f\t    PSNR: %.2f\t    SSIM: %.2f" % evaluate(image_true, an_max, False))
    show_image(an_min, 'MIN')
    show_image(an_max, 'MAX')


if __name__ == '__main__':
    print('    ----COMPARE CONDUCIBILITY FUNCTION----')
    print('Gray image')
    im = im2double(cv2.imread('../images/b&w/Napoli.tif', 0))
    I = im2double(add_noise(im, Noise.GAUSSIAN, 0.08))
    compare_conducibility_function(im, I, 0.4, np.arange(1, 21), 'minimal')

    print('\n\nColor image')
    im2 = im2double(cv2.imread('../images/rgb/cat.jpg'))
    im2 = bgr2rgb(im2)
    I2 = im2double(add_noise(im2, Noise.GAUSSIAN, 0.5))
    I2 = bgr2rgb(I2)
    compare_conducibility_function(im2, I2, 0.8, np.arange(1, 21), 'minimal')
    print('-----------------------------------------')
    print('    ----COMPARE THRESHOLD----')
    compare_threshold_techniques(im2, I2)
    print('-----------------------------------------')
    print('    ----COMPARE NEIGHTBORHOOD----')
    compare_neightborhood(im, I, 1, 0.2)
    print('-----------------------------------------')
