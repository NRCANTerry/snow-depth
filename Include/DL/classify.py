# import necessary packages
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import cv2

def classify(img, model):
    '''
    Function to classify input image based on model
    @param img the image to be classified
    @param model the lenet model to be used
    @type img cv2.image
    @type model keras.models

    @return valid whether the input image is valid according to the model
    @return prob the probability of the input image being valid/invalid
    @rtype valid bool
    @rtype prob float
    '''

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
