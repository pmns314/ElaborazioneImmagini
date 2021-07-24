""" Salvati Vincenzo mat. 0622701550"""
from Utils import *


## calcolo MSE, PSNR e SSIM
def compare_images(I_original, I_noisy, noisy, I_filtred, I_filtred_CV):
    """ Plots the four images passed and computes the mse, psnr and ssim indexes"""
    # creazione del plot
    fig = plt.figure()
    # plot immagine originale
    fig.add_subplot(2, 2, 1)
    plt.title("Originale", fontweight="bold")
    plt.imshow(I, cmap='gray')
    plt.axis("off")
    # plot immagine con rumore
    fig.add_subplot(2, 2, 2)
    plt.title("Rumore " + noisy, fontweight="bold")
    plt.imshow(I_noisy, cmap='gray')
    plt.axis("off")
    # plot immagine filtrata
    fig.add_subplot(2, 2, 3)
    plt.title("Filtro (med e var)", fontweight="bold")
    plt.imshow(I_filtred, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if I_noisy.ndim == 2:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (evaluate(I_original, I_filtred, False)),
                   loc="left")
    else:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (evaluate(I_original, I_filtred, True)),
                   loc="left")
    # plot immagine filtrata con OpenCV
    fig.add_subplot(2, 2, 4)
    plt.title("Filtro OpenCV", fontweight="bold")
    plt.imshow(I_filtred_CV, cmap='gray')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if I_noisy.ndim == 2:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (evaluate(I_original, I_filtred_CV, False)),
                   loc="left")
    else:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (evaluate(I_original, I_filtred_CV, True)),
                   loc="left")


## filtro guidato
def guided_filter(I_guid, I_noisy, epsil=0.5, N=5):
    h = np.ones(N) / (pow(N, 2))
    # calcolo medie e correlazioni
    mean_I = cv2.filter2D(I_guid, -1, h, borderType=cv2.BORDER_REPLICATE)
    mean_p = cv2.filter2D(I_noisy, -1, h, borderType=cv2.BORDER_REPLICATE)
    corr_I = cv2.filter2D(I_guid * I_guid, -1, h, borderType=cv2.BORDER_REPLICATE)
    corr_Ip = cv2.filter2D(I_guid * I_noisy, -1, h, borderType=cv2.BORDER_REPLICATE)
    # calcolo varianza e covarianza
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # calcolo a e b
    a = cov_Ip / (var_I + epsil)
    b = mean_p - a * mean_I
    # calcolo media di valori assunti da a e da b
    mean_a = cv2.filter2D(a, -1, h, borderType=cv2.BORDER_REPLICATE)
    mean_b = cv2.filter2D(b, -1, h, borderType=cv2.BORDER_REPLICATE)
    # calcolo funzione lineare
    return mean_a * I_guid + mean_b


## filtro guidato di OpenCV
def guided_filter_OpenCV(I_guid, I_noisy, nhoodSize, epsil):
    if I_noisy.ndim == 2:
        return cv2.ximgproc.guidedFilter(np.uint8(I_guid), np.uint8(I_noisy), nhoodSize, epsil, -1)
    else:
        return cv2.ximgproc.guidedFilter(np.uint8(I_guid), np.uint8(I_noisy), nhoodSize, epsil)


## applicazione filtri
def filters(I_guid, I_noisy):
    # filtro manuale
    I_fil_guid = im2double(guided_filter(I_guid, I_noisy, 0.5))
    # filtro openCV
    I_guid_uint8 = cv2.normalize(src=I_guid, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    I_noisy_uint8 = cv2.normalize(src=I_noisy, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    I_fil_guid_OpenCV = im2double(guided_filter_OpenCV(I_guid_uint8, I_noisy_uint8, 5, 0.5))
    return I_fil_guid, I_fil_guid_OpenCV


## main
if __name__ == "__main__":
    ## caricamento immagine ##
    img = cv2.imread('../images/b&w/cameraman.tif', -1)
    # img = cv2.imread('../images/rgb/peppers.png')
    # img = cv2.imread('../images/rgb/caster-RGB.tif')

    ## conversione immagine in double ##
    if img.ndim == 2:
        I = im2double(img)
    else:
        I = im2double(img[:, :, 2::-1])

    I_guid = im2double(add_noise(I, "gauss"))

    ## filtraggio di un'immagine con rumore Gaussiano ##
    I_g = im2double(add_noise(I, "gauss"))
    # show_image(I_g, 'Immagine con rumore Gaussiano')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_g)
    compare_images(I, I_g, 'Gaussiano', I_fil_guid, I_fil_guid_OpenCV)

    ## filtraggio di un'immagine con rumore Sale e Pepe ##
    I_sp = im2double(add_noise(I, "s&p"))
    # show_image(I_sp, 'Immagine con rumore Sale e Pepe')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_sp)
    compare_images(I, I_sp, 'Sale e Pepe', I_fil_guid, I_fil_guid_OpenCV)

    ## filtraggio di un'immagine con rumore Poisson ##
    I_poi = im2double(add_noise(I, "poisson"))
    # show_image(I_poi, 'Poisson')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_poi)
    compare_images(I, I_poi, 'Poisson', I_fil_guid, I_fil_guid_OpenCV)

    ## filtraggio di un'immagine con rumore Speckle ##

    I_spe = im2double(add_noise(I, "speckle"))
    # show_image(I_spe, 'Speckle')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_spe)
    compare_images(I, I_spe, 'Speckle', I_fil_guid, I_fil_guid_OpenCV)

    plt.show()
