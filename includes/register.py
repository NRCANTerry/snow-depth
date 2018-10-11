# import necessary packages
import cv2
import numpy as np

# global variables
MAX_FEATURES = 5000

# function to align image to template
def alignImages(img, template):
	# apply median blur to highlight foreground featurse
	img_blur = cv2.medianBlur(img, 5)
	template_blur = cv2.medianBlur(template, 5)

	# convert images to grayscale
	img1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# denoise grayscale image
	img1Gray = cv2.fastNlMeansDenoising(img1Gray,None,4,10,7)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(nfeatures = MAX_FEATURES, scaleFactor = 1.1,
		nlevels = 10)
	kp1, desc1 = orb.detectAndCompute(img_blur, None)
	kp2, desc2 = orb.detectAndCompute(template_blur, None)

	# create brute-force matcher object and match descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(desc1, desc2)

	# sort matches by score and remove poor matches
	# matches with a score greater than 30 are removed
	matches = sorted(matches, key = lambda x: x.distance)
	matches = [x for x in matches if x.distance <= 30]
	matches = matches[:100] if len(matches)>100 else matches

	# draw top matches
	imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# determine homography
	# apply RANSAC-based robust method first then Least-Median robust method
	RANSAC_h, RANSAC_mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	LMEDS_h, LMEDS_mask = cv2.findHomography(points1, points2, cv2.LMEDS, mask = RANSAC_mask)

	# use homography
	height, width, channels = template.shape
	imgReg = cv2.warpPerspective(img, LMEDS_h, (width, height))

	# convert registered image to grayscale
	imgRegGray = cv2.cvtColor(imgReg, cv2.COLOR_BGR2GRAY)

	# define ECC motion model
	warp_mode = cv2.MOTION_HOMOGRAPHY

	# define 3x3 matrix
	warp_matrix = np.eye(3, 3, dtype=np.float32)

	# specify the number of iterations and threshold
	number_iterations = 250
	termination_thresh = 1e-4

	# define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

	# run ECC algorithm (results are stored in warp matrix)
	warp_matrix = cv2.findTransformECC(img2Gray, imgRegGray, warp_matrix, warp_mode, criteria)[1]

	# align image
	imgECCAligned = cv2.warpPerspective(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	# return aligned image and matches
	return imgECCAligned, imgMatches
