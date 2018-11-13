# import necessary packages
import cv2
import numpy as np
from operator import attrgetter
import math
from scipy import ndimage
import sys

# global variables
MAX_FEATURES = int(1e6)#int(5e8)

# the third image is unaltered and will be subjected to the warp
def alignImages3(img, template):
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# apply median blur to highlight foreground features
	img1Gray = cv2.medianBlur(img, 5)
	img2Gray = cv2.medianBlur(template, 5)

	# denoise grayscale image
	img1Gray = cv2.fastNlMeansDenoising(img1Gray,None,3,10,7)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(nfeatures = MAX_FEATURES)
	kp1, desc1 = orb.detectAndCompute(img1Gray, None)
	kp2, desc2 = orb.detectAndCompute(img2Gray, None)

	# create brute-force matcher object and match descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(desc1, desc2)
	bf2 = cv2.BFMatcher(cv2.NORM_HAMMING)
	kmatches = bf2.knnMatch(desc1, desc2, k=2)

	good = []
	for m, n in kmatches:
		if m.distance < 0.5*n.distance:
			good.append([m])

	# sort matches by score and remove poor matches
	# matches with a score greater than 30 are removed
	matches = sorted(matches, key = lambda x: x.distance)
	matches = [x for x in matches if x.distance <= 30]
	if len(matches) > len(good):
		matches = matches[:len(good)]
	#if len(matches) > 100:
	#	matches = matches[:100]

	# draw top matches
	imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None)
	kimgMatches = cv2.drawMatchesKnn(img, kp1, template, kp2, good, None, flags=2)

	cv2.imshow("orig", cv2.resize(imgMatches, None, fx = 0.25, fy = 0.25))
	cv2.imshow("K", cv2.resize(kimgMatches, None, fx = 0.25, fy = 0.25))
	cv2.waitKey()
	#sys.exit()

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	matches = good

	# extract location of good matches
	points1k = np.zeros((len(matches), 2), dtype = np.float32)
	points2k = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1k[i, :] = kp1[match[0].queryIdx].pt
		points2k[i, :] = kp2[match[0].trainIdx].pt

	# determine affine 2D transformation
	# apply RANSAC robust method
	affine_matrix = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC)[0]
	affine_matrixk = cv2.estimateAffine2D(points1k, points2k, method = cv2.RANSAC)[0]
	height, width = template.shape

	# apply registration
	imgReg = cv2.warpAffine(img, affine_matrix, (width, height))
	imgRegGray = cv2.warpAffine(img1Gray, affine_matrix, (width, height))
	#imgReg = img
	#imgRegGray = img1Gray

	imgRegk = cv2.warpAffine(img, affine_matrixk, (width, height))
	imgRegGrayk = cv2.warpAffine(img1Gray, affine_matrixk, (width, height))

	# define ECC motion model
	warp_mode = cv2.MOTION_AFFINE

	# define 2x3 matrix
	warp_matrix = np.eye(2, 3, dtype=np.float32)

	# specify the number of iterations and threshold
	number_iterations = 250
	termination_thresh = 1e-10

	# define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

	# run ECC algorithm (results are stored in warp matrix)
	warp_matrix = cv2.findTransformECC(img2Gray, imgRegGray, warp_matrix, warp_mode, criteria)[1]

	# align image
	imgECCAligned = cv2.warpAffine(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	warp_matrixk = cv2.findTransformECC(img2Gray, imgRegGrayk, warp_matrix, warp_mode, criteria)[1]

	# align image
	imgECCAlignedk = cv2.warpAffine(imgRegk, warp_matrixk, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	return imgECCAligned, imgMatches, affine_matrix, imgECCAlignedk
