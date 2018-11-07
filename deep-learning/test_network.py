# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()

# load image
image = cv2.imread(args.image)
img_orig = image.copy()

# pre-process image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load pre-trained neural network
print("Loading neural network...")
model = load_model(args.model)

# classify the input image
(notBlob, blob) = model.predict(image)[0]

# build the label
label = "Blob" if blob > notBlob else "Not Blob"
proba = blob if blob > notBlob else notBlob
label = "%s: %0.2f%%" % (label, proba*100.0)

# add output to image
output = imutils.resize(img_orig, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show output image
cv2.imshow("Output", output)
cv2.waitKey()
