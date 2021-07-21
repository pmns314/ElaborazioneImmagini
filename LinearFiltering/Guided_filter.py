""" Salvati Vincenzo mat. 0622701550"""

import numpy as np
import numpy.matlib
import cv2
from matplotlib import pyplot as plt
import skimage as sk
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import copy


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def show_image(I, title):
    plt.figure()
    plt.imshow(I, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(title)


def compare_images(imageA, imageB, tit1, tit2):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mean_squared_error(imageA, imageB)
    p = cv2.PSNR(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure()
    plt.suptitle("\nMSE: %.2f, PSNR: %.2f, SSIM: %.2f" % (m, p, s))
    # show first image
    fig.add_subplot(1, 2, 1)
    plt.title(tit1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    fig.add_subplot(1, 2, 2)
    plt.title(tit2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")


def add_gaussian_noise(image, sigma_g, mean_g, p_sp):
    ifl = sk.util.img_as_float(copy.deepcopy(image))
    Noise_G = mean_g + sigma_g * np.random.randn(*image.shape)  # * unpack operator
    return ifl + Noise_G


def add_sale_pepe_noise(image, p_sp):
    I_nb = sk.util.img_as_float(copy.deepcopy(image))
    I_nb[np.random.rand(*I_nb.shape) < p_sp / 2] = 0
    aux = np.random.rand(*I_nb.shape)
    I_nb[(aux > p_sp / 2) & (aux < p_sp)] = 1
    return I_nb


def guided_filter(image, I_guid, epsil):
    N = 3;
    h = np.ones(N) / (pow(N, 2));

    mean_I = cv2.filter2D(image, -1, h, borderType=cv2.BORDER_REPLICATE);
    mean_p = cv2.filter2D(I_g, -1, h, borderType=cv2.BORDER_REPLICATE);
    corr_I = cv2.filter2D(I * I, -1, h, borderType=cv2.BORDER_REPLICATE);
    corr_Ip = cv2.filter2D(I * I_g, -1, h, borderType=cv2.BORDER_REPLICATE);

    var_I = corr_I - mean_I * mean_I;
    cov_Ip = corr_Ip - mean_I * mean_p;

    a = cov_Ip / (var_I + epsil);
    b = mean_p - a * mean_I;

    mean_a = cv2.filter2D(a, -1, h, borderType=cv2.BORDER_REPLICATE);
    mean_b = cv2.filter2D(b, -1, h, borderType=cv2.BORDER_REPLICATE);

    return mean_a * I + mean_b;


if __name__ == '__main__':
    img = cv2.imread('..\images\cameraman.tif', -1)  # Read in your image
    I = im2double(img)  # Convert to normalized floating point
    I_g = im2double(add_gaussian_noise(I, 0.1, 0, 0.05))  # image to smooth
    I_sp = im2double(add_sale_pepe_noise(I, 0.05))
    # show_image(I_g, 'Immagine con rumore Gaussiano')
    # show_image(I_sp, 'Immagine con rumore Sale e Pepe')
    compare_images(I, I_g, 'Originale', 'Rumore Gaussiano')
    compare_images(I, I_sp, 'Originale', 'Rumore Sale e Pepe')
    I_fil_guid = im2double(guided_filter(I, I_g, 0.01))
    compare_images(I, I_fil_guid, 'Originale', 'Filtro guidato')

    plt.show()
