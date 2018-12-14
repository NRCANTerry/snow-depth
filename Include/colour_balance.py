# import necessary modules
import cv2
import numpy as np
import math

def applyMask(matrix, mask, fill_value):
    """
    Function to apply mask to matrix

    Keyword arguments:
    matrix -- the matrix to which the mask will be applied
    mask -- mask to apply
    fill_value -- degree of fill
    """

    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def applyThreshold(matrix, low_val, high_val):
    """
    Function to apply threshold to matrix

    Keyword arguments:
    matrix -- the matrix to be thresholded
    low_val -- lower value for mask
    high_val -- high value for mask)
    """

    low_mask = matrix < low_val
    matrix = applyMask(matrix, low_mask, low_val)

    high_mask = matrix > high_val
    matrix = applyMask(matrix, high_mask, high_val)

    return matrix

def balanceColour(img, percent):
    """
    Function to correct the colour balance of an image
    - scale the histograms of R, G, B channels to cover entire 0-255 scale

    Keyword arguments:
    img -- input image
    percent -- degree of colour correction
    """

    # ensure that image is colour and percent input is valid
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    # determine half percentage
    half_percent = percent / 200.0

    # split in R, G and B channels
    channels = cv2.split(img)

    # create list for output channels
    outputChannels = list()

    # iterate through channels
    for channel in channels:
        # ensure that channel has appropriate shape
        assert len(channel.shape) == 2

        # find the low and high percentile values
        height, width = channel.shape
        size = width * height
        flat = channel.reshape(size)
        assert len(flat.shape) == 1

        # sort
        flat = np.sort(flat)

        # determine number columns
        n_cols = flat.shape[0]

        # determine low and high values
        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        # saturate below the low percentile and above the high percentile
        thresholded = applyThreshold(channel, low_val, high_val)

        # scale channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255,
            cv2.NORM_MINMAX)

        # add scaled channel to output list
        outputChannels.append(normalized)

    # return balanced image
    return cv2.merge(outputChannels)
