# import necessary packages
import cv2
import numpy as np
from operator import attrgetter
import math
from scipy import ndimage

# global variables
MAX_FEATURES = 50000

# function to align image to template
def alignImages3(img, template):
	# apply median blur to highlight foreground features
	img_blur = cv2.medianBlur(img, 49)
	template_blur = cv2.medianBlur(template, 49)

	# convert images to grayscale
	img1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(nfeatures = MAX_FEATURES)
	kp1, desc1 = orb.detectAndCompute(img1Gray, None)
	kp2, desc2 = orb.detectAndCompute(img2Gray, None)

	# create brute-force matcher object and match descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(desc1, desc2)

	# sort matches by score and remove poor matches
	# matches with a score greater than 30 are removed
	matches = sorted(matches, key = lambda x: x.distance)
	#min_match = min(matches, key=attrgetter('distance')).distance
	#max_match = max(matches, key=attrgetter('distance')).distance
	#matches = [x for x in matches if x.distance <= min_match*1.5]
	matches = matches[:4]

	# draw top matches
	imgMatches = cv2.drawMatches(img1Gray, kp1, img2Gray, kp2, matches, None)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# determine homography
	# apply RANSAC-based robust method first then Least-Median robust method
	#RANSAC_h, RANSAC_mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	#LMEDS_h, LMEDS_mask = cv2.findHomography(points1, points2, cv2.LMEDS, mask = RANSAC_mask)
	test, _ = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC, confidence = 0.999)

	# use homography
	height, width, channels = template.shape
	#imgReg = cv2.warpPerspective(img, LMEDS_h, (width, height))
	imgReg = cv2.warpAffine(img, test, (width, height))

	# convert registered image to grayscale
	imgRegGray = cv2.cvtColor(imgReg, cv2.COLOR_BGR2GRAY)

	# define ECC motion model
	#warp_mode = cv2.MOTION_HOMOGRAPHY
	warp_mode = cv2.MOTION_AFFINE

	# define 3x3 matrix
	#warp_matrix = np.eye(3, 3, dtype=np.float32)
	warp_matrix = np.eye(2, 3, dtype=np.float32)

	# specify the number of iterations and threshold
	number_iterations = 250
	termination_thresh = 1e-10

	# define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

	# run ECC algorithm (results are stored in warp matrix)
	#warp_matrix = cv2.findTransformECC(img2Gray, imgRegGray, warp_matrix, warp_mode, criteria)[1]
	warp_matrix = cv2.findTransformECC(img2Gray, img1Gray, warp_matrix, warp_mode, criteria)[1]

	# align image
	#imgECCAligned = cv2.warpPerspective(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	#imgECCAligned = cv2.warpPerspective(img, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	imgECCAligned = cv2.warpAffine(img, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	# return aligned image and matches
	return imgECCAligned, imgMatches, test
