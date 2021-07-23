""" Salvati Vincenzo mat. 0622701550"""

import numpy as np
import numpy.matlib
import cv2
from matplotlib import pyplot as plt
import skimage as sk
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import copy

## conversione immagine in double
def im2double(im):   
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float')-min_val)/(max_val-min_val)
    return out

## plot dell'immagine
def show_image(I, title):       
    plt.figure()
    plt.imshow(I, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(title)

## calcolo MSE, PSNR e SSIM
def compare_images(I_original, I_noisy, noisy, I_filtred, I_filtred_CV):
    # creazioen del plot
    fig = plt.figure()
    # plot immagine originale
    fig.add_subplot(2, 2, 1)
    plt.title("Originale", fontweight="bold")
    plt.imshow(I, cmap=plt.cm.gray)
    plt.axis("off")
    # plot immagine da comparare
    fig.add_subplot(2, 2, 2)
    plt.title("Rumore " + noisy, fontweight="bold")
    plt.imshow(I_noisy, cmap=plt.cm.gray)
    plt.axis("off")
    # plot immagine originale
    fig.add_subplot(2, 2, 3)
    plt.title("Filtro (med e var)", fontweight="bold")
    plt.imshow(I_filtred, cmap=plt.cm.gray)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if I_noisy.ndim == 2:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (mse(I_original, I_filtred), cv2.PSNR(I_original, I_filtred), ssim(I_original, I_filtred)), loc="left")
    else:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (mse(I_original, I_filtred), cv2.PSNR(I_original, I_filtred), ssim(I_original, I_filtred, multichannel=True)), loc="left")
    # plot immagine da comparare
    fig.add_subplot(2, 2, 4)
    plt.title("Filtro OpenCV", fontweight="bold")
    plt.imshow(I_filtred_CV, cmap=plt.cm.gray)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if I_noisy.ndim == 2:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (mse(I_original, I_filtred_CV), cv2.PSNR(I_original, I_filtred_CV), ssim(I_original, I_filtred_CV)), loc="left")
    else:
        plt.xlabel("   - MSE: %.2f\n   - PSNR: %.2f\n   - SSIM: %.2f" % (mse(I_original, I_filtred_CV), cv2.PSNR(I_original, I_filtred_CV), ssim(I_original, I_filtred_CV, multichannel=True)), loc="left")

## applicazione rumore
def noisy(noise_typ, image):
   if noise_typ == "gauss": # applicazione rumore Gaussiano
       sigma_g = 0.1
       mean_g = 0
       p_sp = 0.05
       ifl = sk.util.img_as_float(copy.deepcopy(image))
       Noise_G = mean_g + sigma_g*np.random.randn(*image.shape)
       return ifl + Noise_G
   elif noise_typ == "s&p": # applicazione rumore Sale e Pepe
       p_sp = 0.05
       I_nb = sk.util.img_as_float(copy.deepcopy(image))
       I_nb[np.random.rand(*I_nb.shape) < p_sp/2] = 0
       aux = np.random.rand(*I_nb.shape)
       I_nb[(aux > p_sp/2) & (aux < p_sp)] = 1
       return I_nb
   elif noise_typ == "poisson": # applicazione rumore Poisson
       vals = len(np.unique(image))
       vals = 2 ** np.ceil(np.log2(vals))
       noisy = np.random.poisson(image * vals) / float(vals)
       return noisy
   elif noise_typ == "speckle": # applicazione rumore Speckle
       row,col,ch = image.shape
       gauss = np.random.randn(row,col,ch)
       gauss = gauss.reshape(row,col,ch)        
       noisy = image + image * gauss
       return noisy

## filtro guidato
def guided_filter(I_guid, I_noisy, epsil):
    N = 3;
    h = np.ones(N)/(pow(N,2));
    # calcolo medie e correlazioni
    mean_I = cv2.filter2D(I_guid, -1, h, borderType=cv2.BORDER_REPLICATE);
    mean_p = cv2.filter2D(I_noisy, -1, h, borderType=cv2.BORDER_REPLICATE);
    corr_I = cv2.filter2D(I_guid*I_guid, -1, h, borderType=cv2.BORDER_REPLICATE);
    corr_Ip = cv2.filter2D(I_guid*I_noisy, -1, h, borderType=cv2.BORDER_REPLICATE);
    # calcolo varianza e covarianza
    var_I = corr_I-mean_I*mean_I;
    cov_Ip = corr_Ip-mean_I*mean_p;
    # calcolo a e b
    a = cov_Ip/(var_I+epsil);
    b = mean_p - a*mean_I;
    # calcolo media di valori assunti da a e da b
    mean_a = cv2.filter2D(a, -1, h, borderType=cv2.BORDER_REPLICATE);
    mean_b = cv2.filter2D(b, -1, h, borderType=cv2.BORDER_REPLICATE);
    # calcolo funzione lineare
    return mean_a*I_guid + mean_b;

## filtro guidato di OpenCV
def guided_filter_OpenCV(I_guid, I_noisy, nhoodSize, epsil): 
    if I_noisy.ndim == 2:
        return cv2.ximgproc.guidedFilter(np.uint8(I_guid), np.uint8(I_noisy), nhoodSize, epsil, -1);
    else:
        return cv2.ximgproc.guidedFilter(np.uint8(I_guid), np.uint8(I_noisy), nhoodSize, epsil);

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
    img = cv2.imread('cameraman.tif', -1)
    img = cv2.imread('peppers.png')  
    img = cv2.imread('caster-RGB.tif')
    
    ## conversione immagine in double ##
    if img.ndim == 2:
        I = im2double(img) 
    else:
        I = im2double(img[:,:,2::-1])
    
    I_guid = im2double(noisy("gauss", I));
        
    ## filtraggio di un'immagine con rumore Gaussiano ##
    I_g = im2double(noisy("gauss", I)) 
    #show_image(I_g, 'Immagine con rumore Gaussiano')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_g)
    compare_images(I, I_g, 'Gaussiano', I_fil_guid, I_fil_guid_OpenCV)
    
    ## filtraggio di un'immagine con rumore Sale e Pepe ##
    I_sp = im2double(noisy("s&p", I))
    #show_image(I_sp, 'Immagine con rumore Sale e Pepe')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_sp)
    compare_images(I, I_sp, 'Sale e Pepe', I_fil_guid, I_fil_guid_OpenCV)
    
    ## filtraggio di un'immagine con rumore Poisson ##
    I_poi = im2double(noisy("poisson", I))
    #show_image(I_poi, 'Poisson')
    I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_poi)
    compare_images(I, I_poi, 'Poisson', I_fil_guid, I_fil_guid_OpenCV)
    
    ## filtraggio di un'immagine con rumore Speckle (solo a colori) ##
    if img.ndim != 2:
        I_spe = im2double(noisy("speckle", I))
        #show_image(I_spe, 'Speckle')
        I_fil_guid, I_fil_guid_OpenCV = filters(I_guid, I_spe)
        compare_images(I, I_spe, 'Speckle', I_fil_guid, I_fil_guid_OpenCV)
