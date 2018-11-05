# import necessary modules
import cv2
import numpy as np
from progress_bar import progress
import os
import statistics
from colour_balance import balanceColour
import tqdm
import time
import multiprocessing as m

# threshold for day image
PERCENTAGE_THRESHOLD = 0.5
HIST_LOW = 50
HIST_HIGH = 200

# function that returns whether an image is day or not
def isDay(img_name, directory, upperBorder, lowerBorder):
    # import image
    img = cv2.imread(directory + img_name)

    # get height and width
    h, w = img.shape[:2]

    # crop image
    img = img[int(upperBorder):int(h-lowerBorder), :, :]

    # get total number of pixels in image
    sum_total = img.shape[:2][0] * img.shape[:2][1]

    # variable for number of pixels in range
    num_px = 0

    # get historgram for colour image
    for i in range(0, 3):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])

        # determine number of pixels below in middle of histogram
        num_px += float(np.sum(histr[HIST_LOW:HIST_HIGH]))

    # get mean percentage
    mean_pct = (num_px / 3.0) / float(sum_total)

    # determine if image is night or day based on the mean percetange
    # of pixels in the "dark" range
    if(mean_pct > PERCENTAGE_THRESHOLD):
        return balanceColour(img, 1), img_name
    else:
        return None

# function to unpack arguments explicitly
def unpackArgs(args):
    # and call isDay function
    return isDay(*args)

# parallel function to filter out night images
def filterNightParallel(pool, manager, directory, upperBorder, lowerBorder):
    # setup lists
    images_filtered = list()
    filtered_names = list()

    # get file names
    images = tuple([file_name for file_name in os.listdir(directory)])

    # create list for pool
    tasks = list()
    for name in images:
        tasks.append((name, directory, upperBorder, lowerBorder))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total = len(tasks)):
        if i != None:
            images_filtered.append(i[0])
            filtered_names.append(i[1])
        pass

    # return lists
    return images_filtered, filtered_names

# non parallel function to process small batches of night images
def filterNight(directory, upperBorder, lowerBorder):
    # get file names
    images = [file_name for file_name in os.listdir(directory)]

    # determine number of images
    num_images = len(images)

    # lists for filtered images
    images_filtered = list()
    filtered_names = list()

    # iterate through images
    for img_name in tqdm.tqdm(images):
        # import image
        img = cv2.imread(directory + img_name)

        # get height and width
        h, w = img.shape[:2]

        # crop image
        img = img[upperBorder:(h-lowerBorder), :, :]

        # get total number of pixels in image
        sum_total = img.shape[:2][0] * img.shape[:2][1]

        # variable for number of pixels in range
        num_px = 0

        # get historgram for colour image
        for i in range(0, 3):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])

            # determine number of pixels below in middle of histogram
            num_px += float(np.sum(histr[HIST_LOW:HIST_HIGH]))

        # get mean percentage
        mean_pct = (num_px / 3.0) / float(sum_total)

        # determine if image is night or day based on the mean percetange
        # of pixels in the "dark" range
        if(mean_pct > PERCENTAGE_THRESHOLD):
            # add image to lists
            images_filtered.append(balanceColour(img, 1))
            filtered_names.append(img_name)

    # return lists
    return images_filtered, filtered_names
