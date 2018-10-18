# import necessary modules
import cv2
import numpy as np
import math

# function to apply mask
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

# function to apply threshold
def apply_threshold(matrix, low_val, high_val):
    low_mask = matrix < low_val
    matrix = apply_mask(matrix, low_mask, low_val)

    high_mask = matrix > high_val
    matrix = apply_mask(matrix, high_mask, high_val)

    return matrix

# function that is used to correct the colour balance of an image
# works by scaling the histograms of each of the channels (R, G, B) so that
# they cover the entire 0-255 scale
def balanceColour(img, percent):
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
        thresholded = apply_threshold(channel, low_val, high_val)

        # scale channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255,
            cv2.NORM_MINMAX)

        # add scaled channel to output list
        outputChannels.append(normalized)

    # return balanced image
    return cv2.merge(outputChannels)
