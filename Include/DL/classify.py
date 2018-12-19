# import necessary packages
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import cv2

def classify(img, model):
    """
    Function to classify input image based on model

    Keyword arguments:
    img -- the image to be classified
    model -- the lenet model to be used
    """

    # ensure width and height are non-zero
    height, width = img.shape[:2]
    if not height > 0 or not width > 0:
        return False, 1.0

    # pre-process image
    img = cv2.resize(img, (28, 28))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # classify input image
    (notValidPct, validPct) = model.predict(img)[0]

    # determine valid / notValid
    valid = True if validPct > notValidPct else False
    prob = validPct if valid else notValidPct

    # return validity and probability
    return valid, prob
