# import necessary modules
import numpy as np
import cv2
import os
import sys

# function to increase the brightness of an image
def increase_brightness(img, val):
	# convert image to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	# increase value of pixels
	lim = 255 - val
	v[v > lim] = 255
	v[v <= lim] += val

	# merge channels
	hsv_merge = cv2.merge((h,s,v))

	# convert image to BGR and return
	img = cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)
	return img

# function to apply adaptive histogram equalization
def equalize_hist(img, clip_limit, tile_size):
	# convert to LAB
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

	# split image
	l, a, b = cv2.split(lab)

	# apply adaptive histogram equalization
	clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
	l = clahe.apply(l)

	# merge channels
	lab = cv2.merge((l,a,b))

	# convert to BGR
	bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

	# increase brightness
	bgr = increase_brightness(bgr, 40)

	# return image
	return bgr
