# import necessary packages
import numpy as np
import cv2
import os
import sys
import tqdm
from colour_balance import balanceColour

def brighten(img, val):
    """
    Function to increase the brightness of an input image

    Keyword arguments:
    img -- the input image for which the brightness should be increased
    val -- magnitude of the operation
    """

    # convert image to HSV colour space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # increase brightness of pixels
    lim = 255 - val
    v[v > lim] = 255
    v[v <= lim] += val

    # decrease saturation of image
    s = s*0.5
    s = np.clip(s, 0, 255)

    # merge channels
    hsv_merge = cv2.merge((h, s, v))

    # convert image to BGR and return
    return cv2.cvtColor(hsv_merge.astype(np.uint8), cv2.COLOR_HSV2BGR)

def equalizeHistogramColour(img, clip_limit, tile_size):
    """
    Function to equalize the histogram of colour images

    Keyword arguments:
    img -- input image
    clip_limit -- clip limit for equalization operation
    tile_size -- tile size for equalization operation
    """

    # convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # split channels
    l, a, b = cv2.split(lab)

    # apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l = clahe.apply(l)

    # merge channels
    lab = cv2.merge((l,a,b))

    # convert to BGR
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # return brightened image
    return brighten(bgr, 60)

def equalizeHistogram(img, clip_limit, tile_size):
    """
    Function to equalize the histogram of grayscale images

    Keyword arguments:
    img -- input image
    clip_limit -- clip limit for equalization operation
    tile_size -- tile size for equalization operation
    """

    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
    l = clahe.apply(img)

    # return brightened image
    return np.where((255-l) < 75, 255, l + 75)

def equalizeImage(img, clipLimit, tileSize, name, debug, debug_directory,
    params):
    """
    Function to perform equalization operations on an image

    Keyword arguments:
    img -- the image to be equalized
    clipLimit -- clip limit for equalization operation
    tileSize -- tile size for equalization operation
    name -- image file name
    debug -- bool flag indicating where to output equalized images
    debug_directory -- directory where debug images should be written
    params -- parameters for bilateral filter
    """

    # denoise using bilateral filter
    img_filter = cv2.bilateralFilter(img.copy(), params[0], params[1], params[2])

    # balance colour
    img_filter = balanceColour(img_filter, 5)
    img = balanceColour(img, 5)

    # equqlize image according to specified parameters
    img_eq_gray = equalizeHistogram(img_filter, clipLimit, tileSize)
    img_eq = equalizeHistogramColour(img, clipLimit, tileSize)

    # if debugging write to directory
    if(debug):
        cv2.imwrite(debug_directory + name, img_eq)

    # return equalized images
    return img_eq_gray, img_eq

def equalizeTemplate(templatePath, clipLimit, tileSize, upperBorder, lowerBorder,
    params):
    """
    Function to crop and equalize a template image according to parameters

    Keyword arguments:
    templatePath -- path to template image
    clipLimit -- clip limit for equalization operation
    tileSize -- tile size for equalization operation
    upperBorder -- upper crop parameter
    lowerBorder -- lower crop parameter
    params -- parameters for bilateral filter
    """

    # import and crop template
    template = cv2.imread(templatePath)
    h_temp = template.shape[:2][0]
    template = template[upperBorder:(h_temp-lowerBorder), :, :]

    # get denoised template
    template_noise = cv2.bilateralFilter(template.copy(), params[0], params[1], params[2])

    # return equalized template
    return equalizeHistogram(template, clipLimit, tileSize), equalizeHistogram(template_noise, clipLimit, tileSize)

def equalizeImageSet(images_filtered, filtered_names, templatePath, upperBorder,
    lowerBorder, clipLimit, tileSize, debug, debug_directory_img,
    debug_directory_template, params):
    """
    Function to equalize a set of images

    Keyword arguments:
    images_filtered -- set of images with night images removed
    filtered_names -- list of corresponding image file names
    templatePath -- path to template image
    upperBorder -- upper crop parameter
    lowerBorder -- lower crop parameter
    clipLimit -- clip limit for equalization operation
    tileSize -- tile size for equalization operation
    debug -- bool flag indicating where to output equalized images
    debug_directory_img -- directory where debug images should be written
    debug_directory_template -- directory where template image should be written
    params -- parameters for bilateral filter
    """

    # list for equalized images
    images_equalized = list()
    images_filtered_eq = list()

    # iterator
    index = 0

    # iterate through images
    for img in tqdm.tqdm(images_filtered):
        # equalize image according to specified parameters
        img_eq_gray, img_eq = equalizeImage(img, clipLimit, tileSize, filtered_names[index],
            debug, debug_directory_img, params)

        # add to lists
        images_filtered[index] = img_eq
        images_equalized.append(img_eq_gray)

        # update iterator
        index += 1

    # equalize template
    template_eq, template_reduced_noise = equalizeTemplate(templatePath, clipLimit, tileSize,
        upperBorder, lowerBorder, params)

    # if debugging write to directory
    if(debug):
        cv2.imwrite(debug_directory_template + os.path.split(templatePath)[1], template_eq)

    # return equalized images
    return images_equalized, images_filtered, template_eq, template_reduced_noise

def unpackArgs(args):
    """
    Function to unpack arguments explicitly and call equalizeImage function

    Keyword arguments:
    args -- function arguments passed by parallel equalization function
    """
    return equalizeImage(*args)

def equalizeImageSetParallel(pool, images_filtered, filtered_names, templatePath,
    upperBorder, lowerBorder, clipLimit, tileSize, debug, debug_directory_img,
    debug_directory_template, params):
    """
    Function to equalize a set of images using a parallel pool for computing

    Keyword arguments:
    pool -- the parallel pool to be used for computing
    images_filtered -- set of images with night images removed
    filtered_names -- list of corresponding image file names
    templatePath -- path to template image
    upperBorder -- upper crop parameter
    lowerBorder -- lower crop parameter
    clipLimit -- clip limit for equalization operation
    tileSize -- tile size for equalization operation
    debug -- bool flag indicating where to output equalized images
    debug_directory_img -- directory where debug images should be written
    debug_directory_template -- directory where template image should be written
    params -- parameters for bilateral filter
    """

    # setup lists
    images_equalized = list()
    images_filtered_eq = list()

    # create task list for pool
    tasks = list()
    for i, img in enumerate(images_filtered):
        tasks.append((img, clipLimit, tileSize, filtered_names[i], debug,
            debug_directory_img, params))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total = len(tasks)):
        images_equalized.append(i[0])
        images_filtered_eq.append(i[1])
        pass

    # equalize template
    template_eq, template_reduced_noise = equalizeTemplate(templatePath, clipLimit, tileSize,
        upperBorder, lowerBorder, params)

    # if debugging write to directory
    if(debug):
        cv2.imwrite(debug_directory_template + os.path.split(templatePath)[1], template_eq)

    # return equalized images
    return images_equalized, images_filtered_eq, template_eq, template_reduced_noise
