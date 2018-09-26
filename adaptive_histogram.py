import numpy as np
import cv2

def increase_brigthness(img, value):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	lim = 255 - value
	v[v > lim] = 255
	v[v <= lim] += value

	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

def brighten_shadows(img, value):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	print(np.amin(v))

	v[v < 125] += value

	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

# read in image
img = cv2.imread('images/IMG_0003.jpg')

# convert to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# split image
lab_planes = cv2.split(lab)

# apply adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])

# merge channels
lab = cv2.merge(lab_planes)

# convert to RGB
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
bgr2 = increase_brigthness(bgr, 30)

vis = np.concatenate((img, bgr2), axis = 1)
vis = cv2.resize(vis, None, fx = 0.25, fy = 0.25)

cv2.imshow("Image", vis)
cv2.waitKey()