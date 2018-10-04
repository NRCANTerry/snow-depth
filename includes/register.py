# import necessary packages
import cv2
import numpy as np

# global variables
MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.25

# function to align image to template
def alignImages(img, template):

	# apply median blur to highlight foreground featurse
	img_blur = cv2.medianBlur(img, 5)
	template_blur = cv2.medianBlur(template, 5)

	# convert images to grayscale
	img1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(nfeatures = MAX_FEATURES)
	kp1, desc1 = orb.detectAndCompute(img_blur, None)
	kp2, desc2 = orb.detectAndCompute(template_blur, None)

	# create brute-force matcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

	# match the descriptors
	matches = bf.match(desc1, desc2)

	# sort matches by score
	matches = sorted(matches, key = lambda x: x.distance)

	# remove poor matches
	matches = matches[:int(len(matches)*GOOD_MATCH_PERCENT)]

	# draw top matches
	imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# convert numpy arrays to list
	points1 = points1.tolist()
	points2 = points2.tolist()

	# filter points to exclude poor point matching
	# points which differ by more than 150 pixels in x or y domains
	# are removed from the point lists
	for i, point in enumerate(points1):
		if(abs(point[1] - points2[i][1]) != 0 and abs(point[0] - points2[i][0]) > 150 and \
		(abs(point[0] - points2[i][0]) / abs(point[1] - points2[i][1])) < 25):
			points1.pop(i)
			points2.pop(i)

	# convert lists to numpy arrays
	points1 = np.asarray(points1)
	points2 = np.asarray(points2)

	# find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# use homography
	height, width, channels = template.shape
	imgReg = cv2.warpPerspective(img, h, (width, height))

	return imgReg, h, imgMatches
