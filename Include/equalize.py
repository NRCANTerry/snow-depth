# import necessary packages
import numpy as np
import cv2
import os
import sys
import tqdm
from colour_balance import balanceColour

# image resizing parameters
maxHeight = 1080.0
maxWidth = 1920.0

def brighten(img, val):
    '''
    Increases the brightness of an input image
    @param image the image to be brightened
    @param val the magnitude of the operation
    @type image cv2.image
    @type val int
    @return the modified image
    @rtype cv2.image
    '''

    # convert image to HSV colour space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # increase brightness of pixels
    lim = 255 - val
    v[v > lim] = 255
    v[v <= lim] += val

    # merge channels
    hsv_merge = cv2.merge((h, s, v))

    # convert image to BGR and return
    return cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)

def equalizeHistogramColour(img, clip_limit, tile_size):
    '''
    Equalize histogram of colour images
    @param img the image to be equalized
    @param clip_limit the clip limit for the equalization
    @param tile_size the tile size for the equalization
    @type img cv2.image
    @type clip_limit float
    @type tile_size list(a, b)
    @return equalized image
    @rtype cv2.image
    '''

    # convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # split channels
    l, a, b = cv2.split(lab)

    # apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
    l = clahe.apply(l)

    # merge channels
    lab = cv2.merge((l,a,b))

    # convert to BGR
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # return brightened image
    return brighten(bgr, 50)

def equalizeHistogram(img, clip_limit, tile_size):
    '''
    Equalize histogram of grayscale images
    @param img the image to be equalized
    @param clip_limit the clip limit for the equalization
    @param tile_size the tile size for the equalization
    @type img cv2.image
    @type clip_limit float
    @type tile_size list(a, b)
    @return equalized image
    @rtype cv2.image
    '''

    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
    l = clahe.apply(img)

    # return brightened image
    return np.where((255-l) < 75, 255, l + 75)

def equalizeImage(img, clipLimit, tileSize, name, debug, debug_directory):
    '''
    Perform equalization operations on an image
    @param img the image to be equalized
    @param clipLimit the clip limit for the equalization
    @param tileSize the tile size for the equalization
    @param name the name of the image
    @param debug flag indicating whether image should be written to debug directory
    @param debug_directory the directory where debug image should be written
    @type img cv2.image
    @type clipLimit float
    @type tileSize list(a, b)
    @type name string
    @type debug bool
    @type debug_directory string
    @return img_eq_gray grayscale equalized image
    @return img_eq colour equalized image
    '''

    # denoise using bilateral filter
    img_filter = cv2.bilateralFilter(img.copy(), 9, 75, 75)

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

def equalizeTemplate(templatePath, clipLimit, tileSize, upperBorder, lowerBorder):
    '''
    Crop and equalize template according to parameters
    @param templatePath path to template image
    @param clipLimit the clip limit for the equalization
    @param tileSize the tile size for the equalization
    @param upperBorder upper crop parameter
    @param lowerBorder lower crop parameter
    @type templatePath string
    @type clipLimit float
    @type tileSize list(a, b)
    @type upperBorder int
    @type lowerBorder int
    @return template_eq equalized template image
    @rtype cv2.image
    '''

    # import and crop template
    template = cv2.imread(templatePath)
    h_temp = template.shape[:2][0]
    template = template[upperBorder:(h_temp-lowerBorder), :, :]

    # resize template
    #h, w = template.shape[:2]
    #resizeFactor = min(maxWidth/float(w), maxHeight/float(h))
    #template = cv2.resize(template, None, fx=resizeFactor, fy=resizeFactor)

    # get denoised template
    template_noise = cv2.bilateralFilter(template.copy(), 9, 75, 75)

    # return equalized template
    return equalizeHistogram(template, clipLimit, tileSize), equalizeHistogram(template_noise, clipLimit, tileSize)

def equalizeImageSet(images_filtered, filtered_names, templatePath, upperBorder,
    lowerBorder, clipLimit, tileSize, debug, debug_directory_img,
    debug_directory_template):
    '''
    Equalize sets of images without using a parallel pool
    @param images_filtered set of images with night images removed
    @param filtered_names corresponding names of images
    @param templatePath path to template image
    @param upperBorder upper crop parameter
    @param lowerBorder lower crop parameter
    @param clipLimit the clip limit for equalization
    @param tileSize the tile size for equalization
    @param debug flag indicating whether debugging mode is selected
    @param debug_directory_img path where equalized images are written
    @param debug_directory_template path where equalized template is written

    @type images_filtered list(cv2.image)
    @type filtered_names list(string)
    @type templatePath string
    @type upperBorder int
    @type lowerBorder int
    @type clipLimit float
    @type tileSize list(a, b)
    @type debug bool
    @type debug_directory_img string
    @type debug_directory_template string

    @return images_equalized list of grayscale equalized images
    @return images_filtered list of colour equalized images
    @return template_eq equalized template
    @rtype images_equalized list(cv2.image)
    @rtype images_filtered list(cv2.iamge)
    @rtype template_eq cv2.image
    '''

    # list for equalized images
    images_equalized = list()
    images_filtered_eq = list()

    # iterator
    index = 0

    # iterate through images
    for img in tqdm.tqdm(images_filtered):
        # equalize image according to specified parameters
        img_eq_gray, img_eq = equalizeImage(img, clipLimit, tileSize, filtered_names[index],
            debug, debug_directory_img)

        # add to lists
        images_filtered[index] = img_eq
        images_equalized.append(img_eq_gray)

        # update iterator
        index += 1

    # equalize template
    template_eq, template_reduced_noise = equalizeTemplate(templatePath, clipLimit, tileSize,
        upperBorder, lowerBorder)

    # if debugging write to directory
    if(debug):
        cv2.imwrite(debug_directory_template + os.path.split(templatePath)[1], template_eq)

    # return equalized images
    return images_equalized, images_filtered, template_eq, template_reduced_noise

def unpackArgs(args):
    '''
    Function to unpack arguments explicitly
    @param args function arguments
    @type args arguments
    @return output of equalizeImage function
    @rtype list
    '''
    return equalizeImage(*args)

def equalizeImageSetParallel(pool, images_filtered, filtered_names, templatePath,
    upperBorder, lowerBorder, clipLimit, tileSize, debug, debug_directory_img,
    debug_directory_template):
    '''
    Equalize sets of images using a parallel pool
    @param pool paralell pool to be used
    @param images_filtered set of images with night images removed
    @param filtered_names corresponding names of images
    @param templatePath path to template image
    @param upperBorder upper crop parameter
    @param lowerBorder lower crop parameter
    @param clipLimit the clip limit for equalization
    @param tileSize the tile size for equalization
    @param debug flag indicating whether debugging mode is selected
    @param debug_directory_img path where equalized images are written
    @param debug_directory_template path where equalized template is written

    @type pool multiprocessing.Pool
    @type images_filtered list(cv2.image)
    @type filtered_names list(string)
    @type templatePath string
    @type upperBorder int
    @type lowerBorder int
    @type clipLimit float
    @type tileSize list(a, b)
    @type debug bool
    @type debug_directory_img string
    @type debug_directory_template string

    @return images_equalized list of grayscale equalized images
    @return images_filtered list of colour equalized images
    @return template_eq equalized template
    @rtype images_equalized list(cv2.image)
    @rtype images_filtered list(cv2.iamge)
    @rtype template_eq cv2.image
    '''

    # setup lists
    images_equalized = list()
    images_filtered_eq = list()

    # create task list for pool
    tasks = list()
    for i, img in enumerate(images_filtered):
        tasks.append((img, clipLimit, tileSize, filtered_names[i], debug, debug_directory_img))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total = len(tasks)):
        images_equalized.append(i[0])
        images_filtered_eq.append(i[1])
        pass

    # equalize template
    template_eq, template_reduced_noise = equalizeTemplate(templatePath, clipLimit, tileSize,
        upperBorder, lowerBorder)

    # if debugging write to directory
    if(debug):
        cv2.imwrite(debug_directory_template + os.path.split(templatePath)[1], template_eq)

    # return equalized images
    return images_equalized, images_filtered_eq, template_eq, template_reduced_noise
