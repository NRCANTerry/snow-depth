# import necessary packages
import cv2
import numy as np
import json
from order_points import orderPoints
import os
from get_tensor import getTensor
import statistics
import tqdm

# parameters
median_kernel_size = 5
dilate_kernel = (5,5)

TRAINING_DATA = True # temporary flag to output training data for neural network

def valid(img, coordinates, hsvRanges, blobSizes, upper_border, debug, name,
    debug_directory, dataset, dataset_enabled, NUM_STD_DEV):
    '''
    Function to determine whether stakes in an image are valid (still standing)
    '''

    # create copy of image
    img_copy = img.copy()
    img_ref_blobs = img.copy()

    # determine whether single or double HSV range
    numRanges = len(hsvRanges)

    # create list for blob coordinates on stake
    blobCoordsStake = list()

    # create list for actual blob coordinates
    actualCoordsStake = list()

    # create list for stake tensors
    actualTensorsStake = list()

    # reduce noise in image by local smoothing
    img_blur = cv2.medianBlur(img_copy, median_kernel_size)

    # identify coloured regions in image
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, hsvRanges[0], hsvRanges[1])

    # apply second mask if required
    if(numRanges == 4):
    	mask_hsv2 = cv2.inRange(hsv, hsvRanges[2], hsvRanges[3])
    	mask_hsv = cv2.bitwise_or(mask_hsv, mask_hsv2)

    # erosion followed by dilation to reduce noise
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dilate_kernel)
	mask_open = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)

    # iterate through stakes
    for j, stake in enumerate(coordinates):
        # create bool list for blobs for each stake
		validBlobs = list()

		# get blob size range for stake
		blob_size_range = blobSizes[j]

		# list to store actual coordinates of blobs
		actualCoords = list()

		# iterate through roi in each stake
		for i, rectangle in enumerate(stake):
            # skip stakes
			if(i == 0):
				continue

			# blob counter
			num_blobs = 0

			# create a zero image
			mask = np.zeros(mask_open.shape, np.uint8)

			# get points
			top_left = (rectangle[0][0], rectangle[0][1]-upper_border)
			bottom_right = (rectangle[1][0], rectangle[1][1]-upper_border)

			# copy ROI to zero image
			mask[int(top_left[1]):int(bottom_right[1]),int(top_left[0]):int(bottom_right[0])] = \
				mask_open[int(top_left[1]):int(bottom_right[1]),int(top_left[0]):int(bottom_right[0])]

			# find final coloured polygon regions
			contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
			contour_index = 0

			# iterate through contours
			for k, cnt in enumerate(contours):
				# filter by area
				if(blob_size_range[0] <= cv2.contourArea(cnt) <= blob_size_range[1]):
					# increment blob counter
					num_blobs += 1
					contour_index = k

			# add to valid blob list if one valid blob found
			if(num_blobs == 1):
				validBlobs.append(True)

				# update lowest/highest blob variables
				rect = cv2.minAreaRect(contours[contour_index])
				coords = cv2.boxPoints(rect)
				box = np.array(coords, dtype = "int")

				# add to list of points for stake
				actualCoords.append(orderPoints(box, False))

				# if in debugging mode draw green (valid) rectangle
				if(debug):
					cv2.rectangle(img_copy, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
                        (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 255, 0), 3)

					# write to training folder
					train_img = img_copy[int(rectangle[0][1])-upper_border:int(rectangle[1][1])-upper_border,
						int(rectangle[0][0]):int(rectangle[1][0])]
					train_name = "%d.JPG" % validIndex
					cv2.imwrite(validPath + train_name, train_img)
					validIndex += 1

			# else add invalid blob
			else:
				validBlobs.append(False)

				# add False to list
				actualCoords.append(False)

				# if in debugging mode draw red (invalid) rectangle
				if(debug):
					cv2.rectangle(img_copy, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
                        (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 0, 255), 3)

					# write to training folder
					train_img = img_copy[int(rectangle[0][1])-upper_border:int(rectangle[1][1])-upper_border,
						int(rectangle[0][0]):int(rectangle[1][0])]
					train_name = "%d.JPG" % invalidIndex
					cv2.imwrite(invalidPath + train_name, train_img)
					invalidIndex += 1

		# determine number of valid blobs on stake
		validBlobsOnStake = validBlobs.count(True)

def getValidStakes(imgs, coordinates, hsvRanges, blobSizes, upper_border, debug,
	img_names, debug_directory, dataset, dataset_enabled, NUM_STD_DEV):
    '''
    Function to get valid stakes from a set of images
    '''

    # create directories for training images
	if(debug):
		validPath = debug_directory + "valid/"
		invalidPath = debug_directory + "invalid/"
		os.mkdir(validPath)
		os.mkdir(invalidPath)

    # indexes for image names
	validIndex = 0
	invalidIndex = 0

    # contains output data
    stake_output = {}

    # dictionary for images
    validImages = dict()

    # dictionary for blob coordinates
    blobCoords = dict()

    # dictionary for stake tensor
    actualTensors = dict()

    # iterate through images
    for img_ in tqdm.tqdm(imgs):
