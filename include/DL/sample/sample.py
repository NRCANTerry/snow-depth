# import necessary packages
import sys
sys.path.append('C:\\Users\\tbaricia\\Documents\\GitHub\\snow-depth\\include\\DL')
from lenet import LeNet
from classify import classify
from keras.models import load_model
import cv2
import time

# create neural network
#LeNet("test_model.model", "./valid/", "./invalid/")

# test network
img = cv2.imread("test.JPG")
model = load_model("test_model.model")
print(classify(img, model)) #~0.06s
