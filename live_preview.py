# import necessary modules
import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
	pass

cv2.namedWindow('Colourbars')
cv2.namedWindow('hsv_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('hsv_img', 1200,1200)
cv2.namedWindow('orig', cv2.WINDOW_NORMAL)
cv2.resizeWindow('orig', 1200,1200)

# create trackbars
cv2.createTrackbar("Hue 1", "Colourbars", 0, 180, nothing)
cv2.createTrackbar("Hue 2", "Colourbars", 0, 180, nothing)
cv2.createTrackbar("Sat 1", "Colourbars", 0, 255, nothing)
cv2.createTrackbar("Sat 2", "Colourbars", 0, 255, nothing)
cv2.createTrackbar("Val 1", "Colourbars", 0, 255, nothing)
cv2.createTrackbar("Val 2", "Colourbars", 0, 255, nothing)

img_name = "MFDC3594.JPG"
img = cv2.imread(img_name)
img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

titles = ['Original Image', 'HSV Thresholded']

while(1):
	# get tracker positions
	h1 = cv2.getTrackbarPos("Hue 1", "Colourbars")
	h2 = cv2.getTrackbarPos("Hue 2", "Colourbars")
	s1 = cv2.getTrackbarPos("Sat 1", "Colourbars")
	s2 = cv2.getTrackbarPos("Sat 2", "Colourbars")
	v1 = cv2.getTrackbarPos("Val 1", "Colourbars")
	v2 = cv2.getTrackbarPos("Val 2", "Colourbars")

	cv2.resizeWindow('hsv_img', 600,600)
	cv2.namedWindow('orig', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('orig', 600,600)

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
	cv2.imshow("orig", img)
	cv2.imshow("hsv_img", mask)

	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27:
		break

cv2.destroyAllWindows()