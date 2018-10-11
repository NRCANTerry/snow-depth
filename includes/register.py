# import necessary packages
import cv2
import numpy as np

# global variables
MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.05

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
	# keep matches with score less than 30 with a maximum of 100 matches
	matches = [x for x in matches if x.distance <= 29]
	if(len(matches) > 100):
		matches = matches[:100]

	# draw top matches
	imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# convert numpy arrays to list
	'''
	points1 = points1.tolist()
	points2 = points2.tolist()

	# determine ratio of y change to x change for each set of points
	ratios = list()

	# iterate through list of points
	for i, point in enumerate(points1):
		# calculate ratio
		ratios.append((point[1] - points2[i][1]) * (point[0] - points2[i][0]))

	# convert list of ratios to numpy type
	ratios = np.asarray(ratios)

	# filter points based on ratios
	d = np.abs(ratios - np.median(ratios))
	mdev = np.median(d)
	s = d/mdev if mdev else 0
	points1 = np.asarray(points1)[s<1]
	points2 = np.asarray(points2)[s<1]
	matches = np.asarray(matches)[s<1] if len(np.asarray(matches)[s<1]) > 1 else np.asarray(matches)[s<1][0]	#h2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100)
	'''
	h, mask_new = cv2.findHomography(points1, points2, cv2.RANSAC)
	h2, mask2 = cv2.findHomography(points1, points2, cv2.LMEDS, mask = mask_new)

	'''
	# convert numpy arrays to list
	points1 = points1.tolist()
	points2 = points2.tolist()

	# determine ratio of y change to x change for each set of points
	ratios = list()

	# iterate through list of points
	for i, point in enumerate(points1):
		# calculate ratio
		ratios.append((point[1] - points2[i][1]) * (point[0] - points2[i][0]))

	# convert list of ratios to numpy type
	ratios = np.asarray(ratios)

	# filter points based on ratios
	d = np.abs(ratios - np.median(ratios))
	mdev = np.median(d)
	s = d/mdev if mdev else 0
	points1 = np.asarray(points1)[s<1]
	points2 = np.asarray(points2)[s<1]
	matches = np.asarray(matches)[s<1] if len(np.asarray(matches)[s<1]) > 1 else np.asarray(matches)[s<1][0]
	'''
	#imgMatches = cv2.drawMatches(img, list(kp1), template, list(kp2), list(matches), None)
	'''
	# find homography
	#h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 1) #, cv2.LMEDS)
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, )
	h, mask = cv2.findHomography(points1, points2, cv2.LMEDS, mask = mask)

	print h
	'''
	# use homography
	height, width, channels = template.shape
	imgReg = cv2.warpPerspective(img, h2, (width, height))

	return imgReg, h2, imgMatches

def alignImages2(img, template):

	# convert images to grayscale
	im1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# Find size of image1
	sz = im2_gray.shape

	# Define the motion model
	warp_mode = cv2.MOTION_TRANSLATION

	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
	    warp_matrix = np.eye(3, 3, dtype=np.float32)
	else :
	    warp_matrix = np.eye(2, 3, dtype=np.float32)

	# Specify the number of iterations.
	number_of_iterations = 5000;

	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10;

	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

	if warp_mode == cv2.MOTION_HOMOGRAPHY :
	    # Use warpPerspective for Homography
	    im2_aligned = cv2.warpPerspective (img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	else :
	    # Use warpAffine for Translation, Euclidean and Affine
	    im2_aligned = cv2.warpAffine(img, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

	return im2_aligned
