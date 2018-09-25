import numpy as np
import cv2

img = cv2.imread('IMG_0001.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
channels = cv2.split(img2)
print len(channels)

# create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(channels[0])

cv2.merge(channels, img2)
img2 = cv2.cvtColor(img2, cv2.COLOR_YCR_CB2BGR)

cv2.imshow("1", img)
cv2.imshow("2", img2)
cv2.waitKey()