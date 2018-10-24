# import necessary packages
import cv2
import numpy as np
import json
from progress_bar import progress
from order_points import orderPoints
import os
from get_tensor import getTensor
import statistics

# parameters
median_kernel_size = 5
dilate_kernel = (5,5)

# number of standard deviations away from the mean the tensor can be
NUM_STD_DEV = 5

# function to determine which stakes are valid
# verify that blobs are still within reference windows
# need at least two blobs to have a valid stake
# returns a dictionary indicating which stakes in each image are valid
def getValidStakes(imgs, coordinates, hsvRanges, blobSizes, upper_border, debug,
	img_names, debug_directory, dataset, dataset_enabled):

	# contains output data
	stake_output = {}

	# number of images
	num_images = len(imgs)

	# create bool dictionary for images
	validImages = dict()

	# dictionary for blob coordinates
	blobCoords = dict()

	# dictionary for blob indexes
	blobIndexes = dict()

	# iterate through images
	for count, img_ in enumerate(imgs):
		# update progress bar
		progress(count + 1, num_images, status=img_names[count])

		# duplicate image
		img = img_.copy()
		img_low_blob = img.copy()

		# determine whether single or double HSV range
		numRanges = len(hsvRanges)

		# create bool list for stakes
		validStakes = list()

		# create list for blob coordinates on stakes
		blobCoordsStake = list()

		# create list for blob indexes on stakes
		blobIndexesStake = list()

		# create list for actual blob coordinates
		actualCoordsStake = list()

		# reduce noise in image by local smoothing
		img_blur = cv2.medianBlur(img, median_kernel_size)

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
				mask[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]] = \
					mask_open[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

				# find final coloured polygon regions
				#contours = cv2.findContours(mask_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
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
						cv2.rectangle(img, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
	                        (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 255, 0), 3)

				# else add invalid blob
				else:
					validBlobs.append(False)

					# add False to list
					actualCoords.append(False)

					# if in debugging mode draw red (invalid) rectangle
					if(debug):
						cv2.rectangle(img, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
	                        (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 0, 255), 3)

			# determine number of valid blobs on stake
			validBlobsOnStake = validBlobs.count(True)

			# determine tensor for stake
			tensors_low = list()
			tensors_high = list()

			# mean tensor
			mean_tensor = 0

			# get top and bottom tensor
			# if more than 2 blobs on bottom part of stake
			if(validBlobs[0:3].count(True) >= 2):
				for x in range(0, 4):
					for y in range(x+1, 4):
						# if valid blob, calculate tensor
						if(validBlobs[x] and validBlobs[y]):
							tensors_low.append(getTensor(actualCoords[x][1], actualCoords[y][1],
								((y-x) * (80+56))))

				# get median
				if(len(tensors_low) > 0):
					median_tensor_low = statistics.median(tensors_low)
					mean_tensor += median_tensor_low

			if(len(validBlobs) >= 6 and validBlobs[4:7].count(True) >= 2):
				# determine number of blobs on stake
				num_blobs_on_stake = len(validBlobs)
				if(len(validBlobs) > 8): num_blobs_on_stake = 8

				for x in range(4, num_blobs_on_stake):
					for y in range(x+1, num_blobs_on_stake):
						# if valid blob, calculate tensor
						if(validBlobs[x] and validBlobs[y]):
							tensors_high.append(getTensor(actualCoords[x][1], actualCoords[y][1],
								((y-x) * (80+56))))

				# get median
				if(len(tensors_high) > 0):
					median_tensor_high = statistics.median(tensors_high)

					if(mean_tensor != 0):
						mean_tensor = (mean_tensor + median_tensor_high) / 2.0
					else:
						mean_tensor += median_tensor_high

			# flag to indicate whether stake is valid based on tensor comparison
			tensorValid = True

			# update dataset
			# if dataset isn't enabled, append tensor to dataset
			if(not dataset_enabled[j]):
				dataset[j][1].append(mean_tensor)

			# if dataset is enabled, compare tensor to mean
			else:
				# get mean and standard deviation from dataset
				mean = dataset[j][0][0]
				std_dev = dataset[j][0][1]

				# if tensor measurement is within defined range
				if((mean-(std_dev*NUM_STD_DEV)) <= mean_tensor and
					mean_tensor <= (mean+(std_dev*NUM_STD_DEV))):
					# update data set
					num_vals_dataset = dataset[j][0][2]
					new_vals_dataset = num_vals_dataset + 1
					new_mean = ((mean * num_vals_dataset) + mean_tensor) / new_vals_dataset
					new_std_dev = np.sqrt(pow(std_dev, 2) + ((((mean_tensor - mean) * (mean_tensor - new_mean)) - \
					 			pow(std_dev, 2)) / new_vals_dataset))
					dataset[j] = np.array([[new_mean, new_std_dev, new_vals_dataset], []])

				# update flag to indicate bad tensor match
				else:
					tensorValid = False

			# determine if stake is valid
			# need at minimum 2 blobs for stake to be valid
			validStake = (validBlobsOnStake >= 2 and tensorValid)

			# if in debugging mode draw appropriate rectangle around stake
			if(validStake and debug):
				# green rectangle
				cv2.rectangle(img, (int(stake[0][0][0]), int(stake[0][0][1])-upper_border),
					(int(stake[0][1][0]), int(stake[0][1][1])-upper_border), (0, 255, 0), 3)
			elif(debug):
				# red rectangle
				cv2.rectangle(img, (int(stake[0][0][0]), int(stake[0][0][1])-upper_border),
					(int(stake[0][1][0]), int(stake[0][1][1])-upper_border), (0, 0, 255), 3)

			# if more than 2 valid blobs list stake as valid
			validStakes.append(validStake)

			# add lowest blob to list
			if validStake:
				# order coordinates and append to list
				validCoordinates = [t for t in actualCoords if t != False]
				ordered_coordinates_low = validCoordinates[0]
				ordered_coordinates_high = validCoordinates[len(validCoordinates)-1]
				blobCoordsStake.append(list(ordered_coordinates_low + ordered_coordinates_high))
				actualCoordsStake.append(actualCoords)

				# write labelled image if in debugging mode
				if(debug):
					# draw rectangles
					cv2.rectangle(img_low_blob, tuple(ordered_coordinates_low[0]), tuple(ordered_coordinates_low[2]),
						(0,255,0), 3)
					cv2.rectangle(img_low_blob, tuple(ordered_coordinates_high[0]), tuple(ordered_coordinates_high[2]),
						(0,255,0), 3)
			else:
				# if stake is invalid add zero box
				blobCoordsStake.append([0,0,0,0,0,0,0,0])
				actualCoordsStake.append(False)

		# if in debugging mode
		if(debug):
			# write images to debug directory
			filename, file_extension = os.path.splitext(img_names[count])
			cv2.imwrite(debug_directory + img_names[count], img)
			cv2.imwrite(debug_directory + filename + '-boxes' + file_extension, img_low_blob)

			# create temporary dictionary
			stake_dict = dict()
			stake_dict_coords_low = dict()
			stake_dict_coords_high = dict()

			# add data to output
			for x in range(0, len(coordinates)):
				stake_dict['stake' + str(x)] = validStakes[x]
				stake_dict_coords_low['stake' + str(x)] = blobCoordsStake[x][0:4]
				stake_dict_coords_high['stake' + str(x)] = blobCoordsStake[x][4:8]

			stake_output[img_names[count]] = {
				"validity": stake_dict,
				"lower blob": stake_dict_coords_low,
				"upper blob": stake_dict_coords_high
			}

		# add data to return dictionaries
		validImages[img_names[count]] = validStakes
		blobCoords[img_names[count]] = actualCoordsStake

	# if in debugging mode
	if(debug):
		# output JSON file
		file = open(debug_directory + 'stakes.json', 'w')
		json.dump(stake_output, file, sort_keys=True, indent=4, separators=(',', ': '))

	# return list of valid stakes
	return validImages, blobCoords, dataset
