# import necessary packages
import cv2
import numpy as np

# global variables
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.10

# function to align image to template
def alignImages(img, template):

	# convert images to grayscale
	img1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(MAX_FEATURES)
	kp1, desc1 = orb.detectAndCompute(img1Gray, None)
	kp2, desc2 = orb.detectAndCompute(img2Gray, None)

	# match features
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(desc1, desc2, None)

	# sort matches by score
	matches.sort(key = lambda x: x.distance, reverse = False)

	# remove poor matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	# draw top matches
	imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# use homography
	height, width, channels = template.shape
	imgReg = cv2.warpPerspective(img, h, (width, height))

	return imgReg, h, imgMatches
