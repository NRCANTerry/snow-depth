# import necessary packages
import cv2
import numpy as np
from progress_bar import progress
from sklearn.metrics import mean_squared_error
import json

# global variables
MAX_FEATURES = int(1e7)

# function to align image to template
# the first image and template are already grayscale from clahe application
# the third image is unaltered and will be subjected to the warp
def alignImages(imgs, template, img_names, imgs_apply, debug_directory_registered,
	debug_directory_matches, debug, dataset, dataset_enabled, MAX_ROTATION,
	MAX_TRANSLATION, MAX_SCALING, NUM_STD_DEV):

	# adjust scaling from % to absolute
	MAX_SCALING /= 100

	# determine maximum mean squared error for non-initialized dataset
	max_mean_squared_error = 1e10
	zero_matrix = np.zeros((2,3), dtype=np.float32)

	# create affine matrix according to specified rotation, translation and scale
	alpha = MAX_SCALING * np.cos(np.deg2rad(MAX_ROTATION))
	beta = MAX_SCALING * np.sin(np.deg2rad(MAX_ROTATION))
	affine_transform_matrix = np.array([
		[alpha, beta, MAX_TRANSLATION],
		[-beta, alpha, MAX_TRANSLATION]])

	# determine mean squared error
	max_mean_squared_error = np.sum(np.square(abs(affine_transform_matrix) - zero_matrix))

	# number of images
	num_images = len(imgs)

	# blur template
	#img2Gray = cv2.medianBlur(template, 5)
	img2Gray = cv2.bilateralFilter(template, 9, 75, 75)

	# create output list for images
	registeredImages = list()

	# contains output data for JSON file
	registration_output = {}

	# filtered image names
	images_names_registered = list()

	# iterate through imagse
	for count, img in enumerate(imgs):
		# update progress bar
		progress(count + 1, num_images, status=img_names[count])

		# flags for whether image was aligned
		ORB_aligned_flag = False
		ECC_aligned_flag = False

		# get application image
		img_apply = imgs_apply[count]

		# apply median blur to highlight foreground features
		#img1Gray = cv2.medianBlur(img, 5)
		img1Gray = cv2.bilateralFilter(img, 9, 75, 75)

		# denoise grayscale image
		img1Gray = cv2.fastNlMeansDenoising(img1Gray,None,3,10,7)

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
		matches = [x for x in matches if x.distance <= 30]
		if len(matches) > 100:
			matches = matches[:100]

		# draw top matches
		imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None, flags=2)

		# extract location of good matches
		points1 = np.zeros((len(matches), 2), dtype = np.float32)
		points2 = np.zeros((len(matches), 2), dtype = np.float32)

		for i, match in enumerate(matches):
			points1[i, :] = kp1[match.queryIdx].pt
			points2[i, :] = kp2[match.trainIdx].pt

		# determine affine 2D transformation
		# apply RANSAC robust method
		affine_matrix = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC)[0]
		height, width = template.shape

		# set registered images to original images
		# will be warped if the affine matrix is within spec
		imgReg = img_apply
		imgRegGray = img1Gray

		# get mean squared error between affine matrix and zero matrix
		mean_squared_error = np.sum(np.square(abs(affine_matrix) - zero_matrix))

		# update dataset
		# if dataset isn't enabled, append mean squared error to dataset
		if(not dataset_enabled and mean_squared_error <= max_mean_squared_error):
			dataset[1].append(mean_squared_error)

			# apply registration
			imgReg = cv2.warpAffine(img_apply, affine_matrix, (width, height))
			imgRegGray = cv2.warpAffine(img1Gray, affine_matrix, (width, height))

			# update flag
			ORB_aligned_flag = True

		# if dataset is enabled, compare matrix to mean
		elif mean_squared_error <= max_mean_squared_error:
			# get mean and standard deviation from dataset
			mean = dataset[0][0]
			std_dev = dataset[0][1]

			# if mean squared error is within defined range (number of standard deviations)
			if (mean_squared_error <= (mean+(std_dev*NUM_STD_DEV))):
				# apply registration
				imgReg = cv2.warpAffine(img_apply, affine_matrix, (width, height))
				imgRegGray = cv2.warpAffine(img1Gray, affine_matrix, (width, height))

				# update data set
				num_vals_dataset = dataset[0][2]
				new_vals_dataset = num_vals_dataset + 1
				new_mean = ((mean * num_vals_dataset) + mean_squared_error) / new_vals_dataset
				new_std_dev = np.sqrt(pow(std_dev, 2) + ((((mean_squared_error - mean) * (mean_squared_error - new_mean)) - \
				 			pow(std_dev, 2)) / new_vals_dataset))
				dataset = np.array([[new_mean, new_std_dev, new_vals_dataset], []])

				# update flag
				ORB_aligned_flag = True

		# overly large mean squared error
		else:
			# use unaligned images
			imgReg = img_apply
			imgRegGray = img1Gray

		# define ECC motion model
		warp_mode = cv2.MOTION_AFFINE

		# define 2x3 matrix
		warp_matrix = np.eye(2, 3, dtype=np.float32)

		# specify the number of iterations and threshold
		number_iterations = 500
		termination_thresh = 1e-7 if mean_squared_error <= max_mean_squared_error else 1e-9

		# define termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

		# run ECC algorithm (results are stored in warp matrix)
		warp_matrix = cv2.findTransformECC(img2Gray, imgRegGray, warp_matrix, warp_mode, criteria)[1]

		# only check if dataset enabled
		if(dataset_enabled):
			# compare warp matrix to data set
			mean_squared_error = np.sum(np.square(abs(warp_matrix) - zero_matrix))

			# get mean and standard deviation from dataset
			mean = dataset[0][0]
			std_dev = dataset[0][1]

			# align image if warp is within spec
			if (mean_squared_error <= (mean+(std_dev*NUM_STD_DEV))):
				# align image
				imgECCAligned = cv2.warpAffine(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

				# update flag
				ECC_aligned_flag = True
			# else use ORB registered image
			else:
				imgECCAligned = imgReg

		# else align image
		else:
			# align image
			imgECCAligned = cv2.warpAffine(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

			# update flag
			ECC_aligned_flag = True

		# only if image was aligned (is not the same as input image)
		if(ORB_aligned_flag or ECC_aligned_flag):
			# append to list
			registeredImages.append(imgECCAligned)

			# add image name to filtered list
			images_names_registered.append(img_names[count])

			# write images to debug directories
			cv2.imwrite(debug_directory_registered + img_names[count], imgECCAligned)
			cv2.imwrite(debug_directory_matches + img_names[count], imgMatches)

		# if in debugging mode
		if(debug):
			# add data to output
			registration_output[img_names[count]] = {
				"ORB Aligned": ORB_aligned_flag,
				"ORB Matrix": affine_matrix.tolist(),
				"ECC Aligned": ECC_aligned_flag,
				"ECC Matrix": warp_matrix.tolist()
			}

	# if in debugging mode
	if(debug):
		# output JSON file
		file = open(debug_directory_registered + 'registered.json', 'w')
		json.dump(registration_output, file, sort_keys=True, indent=4, separators=(',', ': '))

	# return list of registered images
	return registeredImages, dataset, images_names_registered
