# import necessary modules
import cv2
import os
import tqdm
import multiprocessing as m
from numpy import squeeze, asarray
from colour_balance import balanceColour

# Constant
MAX_NIGHT = 2 # maximum difference between weighted means for night image

def calculate_weighted_means(data):
    '''
    Calculates the weighted means for the histogram data
    @param data the colour histogram data
    @type data dict(red, green, blue)
    @return the weighted means for red, green and blue channels
    @rtype list(red, green, blue)
    '''

    # weighted mean variables
    weighted_red = 0
    weighted_green = 0
    weighted_blue = 0

    # iterate through histogram intensities
    for i in xrange(256):
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
    '''
    Calculates the histogram data for the image
    @param img the image to compute the histogram for
    @type img cv2 image
    @return the calculated histogram data
    @rtype dict(red, green, blue)
    '''

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
    '''
    Returns whether an image is taken during the day or night
    @param img the image to compute the histogram for
    @param name the name of the image
    @type img cv2 image
    @type name string
    @return whether the image is day or not
    @return original image
    @return image name
    @rtype bool
    @rtype image cv2.image
    @rtype name string
    '''

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

def filterNight(directory, upperBorder, lowerBorder):
    '''
    Returns filtered list of day images
    @param directory the path to the directory containing the image
    @param upperBorder upper crop parameter
    @param lowerBorder lower crop parameter
    @type directory string
    @type upperBorder int
    @type lowerBorder int
    @return list of filtered images
    @return list of filtered names
    @rtype list(images)
    @rtype list(string)
    '''

    # get file names
    images = [file_name for file_name in os.listdir(directory)]

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

        # determine whether image is day or night
        if(isDay(img, img_name)[0]):
            # add image to lists
            images_filtered.append(balanceColour(img, 5))
            filtered_names.append(img_name)

    # return lists
    return images_filtered, filtered_names

def unpackArgs(args):
    '''
    Function to unpack arguments explicitly
    @param args function arguments
    @type args arguments
    @return output of isDay function
    @rtype bool
    '''
    return isDay(*args)

def filterNightParallel(pool, directory, upperBorder, lowerBorder):
    '''
    Returns filtered list of day images using parallel pool for computing
    @param pool the parallel pool used for computing
    @param directory the path to the directory containing the image
    @param upperBorder upper crop parameter
    @param lowerBorder lower crop parameter
    @type pool multiprocessing pool
    @type directory string
    @type upperBorder int
    @type lowerBorder int
    @return list of filtered images
    @return list of filtered names
    @rtype list(images)
    @rtype list(string)
    '''

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
            images_filtered.append(balanceColour(i[1], 5))
            filtered_names.append(i[2])
        pass

    # return lists
    return images_filtered, filtered_names
