# import necessary modules
import cv2
import numpy as np
from matplotlib import pyplot as plt
import Tkinter as tk
import tkFileDialog

def nothing(x):
	pass

def HSVPreview():
	# select image file
	root = tk.Tk()
	root.withdraw()
	filename = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

	# open slider window
	cv2.namedWindow("HSV Sliders")
	cv2.resizeWindow("HSV Sliders", 500, 350)

	# open comparison window
	cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

	# create trackbars
	cv2.createTrackbar("Hue 1", "HSV Sliders", 0, 180, nothing)
	cv2.createTrackbar("Sat 1", "HSV Sliders", 0, 255, nothing)
	cv2.createTrackbar("Val 1", "HSV Sliders", 0, 255, nothing)
	cv2.createTrackbar("Hue 2", "HSV Sliders", 0, 180, nothing)
	cv2.createTrackbar("Sat 2", "HSV Sliders", 0, 255, nothing)
	cv2.createTrackbar("Val 2", "HSV Sliders", 0, 255, nothing)

	# open image
	img = cv2.imread(filename)

	# resize image to 1/4 of original size
	img = cv2.resize(img, (0,0), None, 0.25, 0.25)

	# repeat until user exits
	while(1):
		# get tracker positions
		h1 = cv2.getTrackbarPos("Hue 1", "HSV Sliders")
		h2 = cv2.getTrackbarPos("Hue 2", "HSV Sliders")
		s1 = cv2.getTrackbarPos("Sat 1", "HSV Sliders")
		s2 = cv2.getTrackbarPos("Sat 2", "HSV Sliders")
		v1 = cv2.getTrackbarPos("Val 1", "HSV Sliders")
		v2 = cv2.getTrackbarPos("Val 2", "HSV Sliders")

		# convert image to HSV
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# apply HSV mask
		mask = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

		# create horizontal stack
		numpy_horizontal = np.hstack((img, mask))

		# display image to user
		cv2.imshow("Comparison", numpy_horizontal)

		# exit when user presses esc
		k = cv2.waitKey(1) & 0xFF
		if k == ord('m'):
			mode = not mode
		elif k == 27:
			break

	# close all windows on exit
	cv2.destroyAllWindows()