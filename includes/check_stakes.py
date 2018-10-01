import cv2
import numpy as np
import json
from progress_bar import progress

# parameters
median_kernel_size = 5
dilate_kernel = (5,5)

# function to determine which stakes are valid
# verify that blobs are still within reference windows
# need at least two blobs to have a valid stake
# returns a dictionary indicating which stakes in each image are valid
def getValidStakes(imgs, coordinates, hsvRanges, min_area, max_area, upper_border, debug, 
	img_names, debug_directory):

	# contains output data
	stake_output = {}

	# number of images
	num_images = len(imgs)

	# create bool dictionary for images
	validImages = dict()

	# iterate through images
	for count, img in enumerate(imgs):
		# update progress bar
		progress(count + 1, num_images, status=img_names[count])

		# determine whether single or double HSV range
		numRanges = len(hsvRanges)

		# create bool list for stakes
		validStakes = list()

		# iterate through stakes
		for j, stake in enumerate(coordinates):
			# create bool list for blobs for each stake
			validBlobs = list()

			# iterate through roi in each stake
			for i, rectangle in enumerate(stake):
				# skip stakes
				if(i == 0):
					continue

				# blob counter
				num_blobs = 0

				# create a zero image
				mask = np.zeros(img.shape, np.uint8)

				# get points
				top_left = (rectangle[0][0], rectangle[0][1]-upper_border)
				bottom_right = (rectangle[1][0], rectangle[1][1]-upper_border)

				# copy ROI to zero image
				mask[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]] = \
					img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

				# reduce noise in image by local smoothing
				img_blur = cv2.medianBlur(mask, median_kernel_size)

				# identify coloured regions in image
				hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
				mask = cv2.inRange(hsv, hsvRanges[0], hsvRanges[1])

				# apply second mask if required
				if(numRanges == 4):
					mask2 = cv2.inRange(hsv, hsvRanges[2], hsvRanges[3])
					mask = cv2.bitwise_or(mask, mask2)

				# erosion followed by dilation to reduce noise
				kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dilate_kernel)
				mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

				# find final coloured polygon regions
				contours = cv2.findContours(mask_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
				print(len(contours))

				# iterate through contours
				for cnt in contours:
					# filter by area
					if(min_area <= cv2.contourArea(cnt) <= max_area):
						# increment blob counter
						num_blobs += 1

				# add to valid blob list if one valid blob found
				if(num_blobs == 1):
					validBlobs.append(True)

					# if in debugging mode draw green (valid) rectangle
					if(debug):
						cv2.rectangle(img, (rectangle[0][0], rectangle[0][1]-upper_border), 
	                        (rectangle[1][0], rectangle[1][1]-upper_border), (0, 255, 0), 3)

				# else add invalid blob
				else:
					validBlobs.append(False)

					# if in debugging mode draw red (invalid) rectangle
					if(debug):
						cv2.rectangle(img, (rectangle[0][0], rectangle[0][1]-upper_border), 
	                        (rectangle[1][0], rectangle[1][1]-upper_border), (0, 0, 255), 3)

			# determine number of valid blobs on stake
			validBlobsOnStake = validBlobs.count(True)

			# determine if stake is valid
			validStake = False
			if(validBlobsOnStake >= 2):
				validStake = True

			# if in debugging mode draw appropriate rectangle around stake
			if(validStake and debug):
				# green rectangle
				cv2.rectangle(img, (stake[0][0][0], stake[0][0][1]-upper_border), 
					(stake[0][1][0], stake[0][1][1]-upper_border), (0, 255, 0), 3)
			elif(debug):
				# red rectangle
				cv2.rectangle(img, (stake[0][0][0], stake[0][0][1]-upper_border), 
					(stake[0][1][0], stake[0][1][1]-upper_border), (0, 0, 255), 3)

			# if more than 2 valid blobs list stake as valid
			validStakes.append(validStake)

		# if in debugging mode
		if(debug):
			# write image to debug directory
			cv2.imwrite(debug_directory + img_names[count], img)

		# create temporary dictionary
		stake_dict = dict()

		# add data to output
		for x in range(0, len(coordinates)):
			stake_dict['stake' + str(x)] = validStakes[x] 

		stake_output[img_names[count]] = stake_dict

		# add data to return dictionary
		validImages[img_names[count]] = validStakes

	# output JSON file
	file = open(debug_directory + 'stakes.json', 'w')
	json.dump(stake_output, file, sort_keys=True, indent=4, separators=(',', ': '))

	# return list of valid stakes
	return validImages