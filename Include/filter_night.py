# import necessary modules
import cv2
import os
import tqdm
import multiprocessing as m
from numpy import squeeze, asarray
from colour_balance import balanceColour
from PIL import Image
from datetime import datetime
from datetime import time
import numpy as np

# Constant
MAX_NIGHT = 2 # maximum difference between weighted means for night image
maxHeight = 1080.0 # resize parameters
maxWidth = 1920.0
maxWidth4K = 3840
maxHeight4K = 2160

def calculate_weighted_means(data):
    """
    Function that calculates the weighted means for the histogram data and
    returns the weighted meansd for red, green and blue channels

    Keyword arguments:
    data -- the colour histogram data
    """

    # weighted mean variables
    weighted_red = 0
    weighted_green = 0
    weighted_blue = 0

    # iterate through histogram intensities
    for i in range(256):
        # add to weighted means
        weighted_red += (data["red"][i] * i)
        weighted_green += (data["green"][i] * i)
        weighted_blue += (data["blue"][i] * i)

    # calculate means
    weighted_red /= sum(data["red"])
    weighted_green /= sum(data["green"])
    weighted_blue /= sum(data["blue"])

    # return data
    return [weighted_red, weighted_blue, weighted_green]

def get_histogram(img):
    """
    Function that calculates the histogram of an input image

    Keyword arguments:
    img -- input image for which a histogram should be calculated
    """

    # calculate histogram
    blue = cv2.calcHist([img], [0], None, [256], [0, 256])
    green = cv2.calcHist([img], [1], None, [256], [0, 256])
    red = cv2.calcHist([img], [2], None, [256], [0, 256])

    # return computed histogram data
    return {
        "red": squeeze(asarray(blue.astype(int))),
        "green": squeeze(asarray(green.astype(int))),
        "blue": squeeze(asarray(red.astype(int)))
    }

def isDay(img, name):
    """
    Function that returns whether an image is taken during the day or night

    Keyword arguments:
    img -- image for which the time should be determined
    name -- image file name
    """

    # compute histogram
    hist = get_histogram(img)

    # get weighted means
    wMeans = calculate_weighted_means(hist)

    # determine maximum difference between weighted means
    max_diff = abs(wMeans[1] - wMeans[0])
    for i in range(0, len(wMeans)):
        for j in range(i+1, len(wMeans)):
            if(abs(wMeans[j] - wMeans[i]) > max_diff):
                max_diff = abs(wMeans[j] - wMeans[i])

    # return whether image is day or night
    return (max_diff > MAX_NIGHT, img, name)

def filterNight(directory, upperBorder, lowerBorder, dateRange, imageSummary):
    """
    Function that returns a filtered list of daytime images

    Keyword arguments:
    directory -- path to the directory containing the input images
    upperBorder -- upper crop parameter
    lowerBorder -- lower crop parameter
    dateRange -- list containing dates of images user would like to process
    imageSummary -- dictionary containing information about each run
    """

    # get file names
    images = [file_name for file_name in os.listdir(directory)]

    # lists for filtered images
    images_filtered = list()
    filtered_names = list()

    # iterate through images
    for img_name in tqdm.tqdm(images):
        # if date range selected ensure image falls in date range
        if(dateRange[3]):
            # import using PIL
            pil_im = Image.open(directory + img_name)

            # check EXIF data
            exif = pil_im._getexif()
            exif = exif[36867] if exif is not None else None
            date = datetime.strptime(exif, '%Y:%m:%d %H:%M:%S') if exif is not None else None

            # check for all filled scenario (i.e. user specifies start and
            # end dates + times)
            if dateRange[0] is not None and dateRange[1] is not None and \
                all(v is not None for v in dateRange[2]):
                # update datetime objects
                dateRange[0] = dateRange[0].replace(hour=dateRange[2][0],
                    minute=dateRange[2][1])
                dateRange[1] = dateRange[1].replace(hour=dateRange[2][2],
                    minute=dateRange[2][3])

                # reset time range
                dateRange[2] = [None, None, None, None]

            # create time boundaries
            if dateRange[2] != [None, None, None, None]:
                lower_time = time(dateRange[2][0], dateRange[2][1])
                upper_time = lower_time

                if dateRange[2][2] != None and dateRange[2][3] != None:
                    upper_time = time(dateRange[2][2], dateRange[2][3])

                current_time = time(date.hour, date.minute)

            # check if image is valid
            if date is None: # if no exif data available
                # convert img to cv2
                img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            elif(
                ((dateRange[0] is None and dateRange[1] is None) or
                    dateRange[0] <= date <= dateRange[1]) # date check
                and (dateRange[2] == [None, None, None, None] or (lower_time <= current_time
                    and current_time <= upper_time)) # time check
            ):
                # convert img to cv2
                img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            # if image isn't in the date range skip it
            else:
                continue

        # if no date range selected
        else:
            # import image
            img = cv2.imread(directory + img_name)

        # get height and width
        h, w = img.shape[:2]

        # crop image
        img = img[upperBorder:(h-lowerBorder), :, :]

        # determine whether image is day or night
        output = isDay(img, img_name)
        if(output[0]):
            # constrain resolution of input images to 4K
            h, w = output[1].shape[:2]
            if(w > maxWidth4K or h > maxHeight4K):
                factor = min(maxWidth4K/float(w), maxHeight4K/float(h))
                output[1] = cv2.resize(output[1], None, fx=factor, fy=factor)

            # add images to list
            images_filtered.append(output[1])
            filtered_names.append(img_name)

        # add to individual summary
        imageSummary[img_name] = {"Valid Image": output[0]}

    # return lists
    return images_filtered, filtered_names, imageSummary

def unpackArgs(args):
    """
    Function to unpack arguments explicitly and call isDay function

    Keyword arguments:
    args -- function arguments passed by parallel filtering night function
    """
    return isDay(*args)

def filterNightParallel(pool, directory, upperBorder, lowerBorder, imageSummary):
    """
    Function that returns a filtered list of daytime images using parallel pool
    for computing

    Keyword arguments:
    pool -- the parallel pool used for computing
    directory -- path to the directory containing the input images
    upperBorder -- upper crop parameter
    lowerBorder -- lower crop parameter
    dateRange -- list containing dates of images user would like to process
    imageSummary -- dictionary containing information about each run
    """

    # get file names
    images = tuple([file_name for file_name in os.listdir(directory)])

    # lists for filtered images
    images_filtered = list()
    filtered_names = list()

    # create list for pool
    tasks = list()
    for name in images:
        # import image
        img = cv2.imread(directory + name)

        # get height and width
        h, w = img.shape[:2]

        # crop image
        img = img[upperBorder:(h-lowerBorder), :, :]

        tasks.append((img, name))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total=len(tasks)):
        if i[0]: # if day image
            # add to lists
            images_filtered.append(i[1])
            filtered_names.append(i[2])

        # add to individual summary
        imageSummary[i[2]] = {"Valid Image": i[0]}

    # return lists
    return images_filtered, filtered_names, imageSummary
