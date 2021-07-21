"Dario Cival mat: 0622701620"

from Morphological import *
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
import skimage as sk
from scipy import signal

from OrdFilter import *
from skimage.metrics import *

if __name__ == '__main__':

    image = cv2.imread('../images/cameraman.tif', cv2.IMREAD_GRAYSCALE)
    image = sk.img_as_float(image)
    N = 5
    p = 0.05

    cv2.imshow('cameraman', image)
    cv2.imshow('salt and pepper', add_salt_pepper(image, 0.05))
    image_sp = add_salt_pepper(image, p)
    strle = disk(3)
    image_gaussian = add_gaussian(image, 0, 0.05)
    cv2.imshow('gaussian', image_gaussian)
    # cv2.imshow('erosion cameraman', erosion(image, strle))
    # cv2.imshow('dilatation image', dilatation(image, strle))
    # cv2.imshow('opening image', open(image_sp, strle))
    # cv2.imshow('median image gaussian', median_open_close(image_gaussian, strle))
    open_close = median_open_close(image_sp, strle)
    cv2.imshow('denoised image with open-closing filter', open_close)
    median_filter = sk.img_as_float(cv2.medianBlur(sk.img_as_ubyte(image_sp), N))

    cv2.imshow('median filtering', median_filter)

    # FILTRO BITONICO
    nh = 3
    strle = disk(nh + 1)
    dimcut = 10
    o = 0.1
    sigma = 0.65 * nh / np.sqrt(2)
    i_filt_morph_bit_sp = bitonic_filter(image_sp, nh, o, strle)
    i_filt_morph_bit_ga = bitonic_filter(image_gaussian, nh, o, strle)

    cv2.imshow('denoising with bitonic filter *salt and pepper*', i_filt_morph_bit_sp[dimcut:-dimcut, dimcut:-dimcut])
    cv2.imshow('denoising with bitonic filter *gaussian*', i_filt_morph_bit_ga[dimcut:-dimcut, dimcut:-dimcut])
    
    # PSNR
    # SSIM
    print("PSNR bitonic *salt and pepper*", peak_signal_noise_ratio(image[dimcut:-dimcut, dimcut:-dimcut],
                                                                    i_filt_morph_bit_sp[dimcut:-dimcut,
                                                                    dimcut:-dimcut]))
    print("PSNR bitonic *gaussian*",
          peak_signal_noise_ratio(image[dimcut:-dimcut, dimcut:-dimcut],
                                  i_filt_morph_bit_ga[dimcut:-dimcut, dimcut:-dimcut]))

    print("SSIM bitonic *salt and pepper*", structural_similarity(image[dimcut:-dimcut, dimcut:-dimcut],
                                                                  i_filt_morph_bit_sp[dimcut:-dimcut,
                                                                  dimcut:-dimcut]))
    print("SSIM bitonic *gaussian*",
          structural_similarity(image[dimcut:-dimcut, dimcut:-dimcut], i_filt_morph_bit_ga[dimcut:-dimcut, dimcut:-dimcut]))

    # print(peak_signal_noise_ratio(image[dimcut:-dimcut, dimcut:-dimcut], open_close[dimcut:-dimcut, dimcut:-dimcut]))
    # print(peak_signal_noise_ratio(image[dimcut:-dimcut, dimcut:-dimcut], median_filter[dimcut:-dimcut, dimcut:-dimcut]))

    print("**************************")
    any_filter_sp = anysotropic_filter(image_sp)
    any_filter_ga = anysotropic_filter(image_gaussian)

    cv2.imshow('denoising with anysotropic filter *salt and pepper*', any_filter_ga)
    cv2.imshow('denoising with anysotropic filter *gaussian*', any_filter_ga)

    print("PSNR anystropic *salt and pepper*",
          peak_signal_noise_ratio(image[dimcut:-dimcut, dimcut:-dimcut], any_filter_sp[dimcut:-dimcut, dimcut:-dimcut]))
    print("PSNR anystropic *gaussian*",
          peak_signal_noise_ratio(image[dimcut:-dimcut, dimcut:-dimcut], any_filter_ga[dimcut:-dimcut, dimcut:-dimcut]))
    print("SSIM anysotropic *salt and pepper*",
          structural_similarity(image[dimcut:-dimcut, dimcut:-dimcut], any_filter_sp[dimcut:-dimcut, dimcut:-dimcut]))
    print("SSIM anysotropic *gaussian*",
          structural_similarity(image[dimcut:-dimcut, dimcut:-dimcut], any_filter_ga[dimcut:-dimcut, dimcut:-dimcut]))

    cv2.waitKey(0)
