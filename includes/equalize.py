# import necessary modules
import numpy as np
import cv2
import os
import sys
import tqdm

# function to increase the brightness of an image
def increase_brightness(img, val):
    # convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # increase value of pixels
    lim = 255 - val
    v[v > lim] = 255
    v[v <= lim] += val

    # merge channels
    hsv_merge = cv2.merge((h,s,v))

    # convert image to BGR and return
    img = cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)
    return img

# function to apply adaptive histogram equalization
def equalize_hist_colour(img, clip_limit, tile_size):
    # convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # split image
    l, a, b = cv2.split(lab)

    # apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
    l = clahe.apply(l)

    # merge channels
    lab = cv2.merge((l,a,b))

    # convert to BGR
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # increase brigh#tness
    bgr = increase_brightness(bgr, 50)

    # return image
    return bgr

# function to apply adaptive histogram equalization
def equalize_hist(img, clip_limit, tile_size):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
    l = clahe.apply(img)

    # increase brigh#tness
    bgr = np.where((255-l) < 75, 255, l + 75)

    # return image
    return bgr

# function to perform all equalizations in pool
def equalizeParallel(img, clipLimit, tileSize, name, debug, debug_directory):
    # equalize image according to specified parameters
    img_eq_gray = equalize_hist(img, clipLimit, tileSize)
    img_eq = equalize_hist_colour(img, clipLimit, tileSize)

    # if debugging write to directory
    if(debug):
        cv2.imwrite((debug_directory + name), img_eq)

    # return equalized images
    return img_eq_gray, img_eq

# function to unpack arguments explicitly
def unpackArgs(args):
    # and call equalization function
    return equalizeParallel(*args)

# parallel function to equalize images
def equalizeImagesParallel(pool, manager, images_filtered, filtered_names, templatePath, upperBorder,
    lowerBorder, clipLimit, tileSize, debug, debug_directory_img, debug_directory_template):
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

    # equalize and crop template
    template = cv2.imread(templatePath)
    h_temp = template.shape[:2][0]
    template = template[upperBorder:(h_temp-lowerBorder), :, :]
    template_eq = equalize_hist(template, clipLimit, tileSize)

    # if debugging write to directory
    if(debug):
        cv2.imwrite((debug_directory_template + os.path.split(templatePath)[1]), template_eq)

    # return equalized images
    return images_equalized, images_filtered_eq, template_eq

# non parallel function to equalize small batches of images
def equalizeImages(images_filtered, filtered_names, templatePath, upperBorder,
    lowerBorder, clipLimit, tileSize, debug, debug_directory_img, debug_directory_template):
    # list for equalized images
    images_equalized = list()

    # iterator
    index = 0

    # iterate through images
    for img in tqdm.tqdm(images_filtered):
        # equalize image according to specified parameters
        img_eq_gray = equalize_hist(img, clipLimit, tileSize)
        img_eq = equalize_hist_colour(img, clipLimit, tileSize)

        # replace image in list with equalized image
        images_filtered[index] = img_eq

        # add to list
        images_equalized.append(img_eq_gray)

        # if debugging write to directory
        if(debug):
            cv2.imwrite((debug_directory_img + filtered_names[index]), img_eq)

        # update iterator
        index += 1

    # equalize and crop template
    template = cv2.imread(templatePath)
    h_temp = template.shape[:2][0]
    template = template[upperBorder:(h_temp-lowerBorder), :, :]
    template_eq = equalize_hist(template, clipLimit, tileSize)

    # if debugging write to directory
    if(debug):
        cv2.imwrite((debug_directory_template + os.path.split(templatePath)[1]), template_eq)

    # return equalized images
    return images_equalized, images_filtered, template_eq
