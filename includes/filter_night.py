# import necessary modules
import cv2
import numpy as np

# parameters
median_kernel_size = 5
dilate_kernel = (5,5)
min_pixel_count = 100

# function to determine if an image was taken at night
def isDay(img, hsvRanges, blobSizes):
	# determine whether single or double HSV range
	numRanges = len(hsvRanges)

	# get image shape
	h, w = img.shape[:2]

	# reduce noise in image by local smoothing
	img = cv2.medianBlur(img, median_kernel_size)

	# identify coloured regions in image
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, hsvRanges[0], hsvRanges[1])

	# apply second mask if required
	if(numRanges == 4):
		mask2 = cv2.inRange(hsv, hsvRanges[2], hsvRanges[3])
		mask = cv2.bitwise_or(mask, mask2)

	# erosion followed by dilation to reduce noise
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dilate_kernel)
	mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	# find final coloured polygon regions
	mask_filtered = np.zeros((h,w), dtype=np.uint8)
	contours = cv2.findContours(mask_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

	# set range as smallest to largest blob sizes
	min_area = min(blobSizes, key = lambda y: y[0])[0]
	max_area = max(blobSizes, key = lambda y: y[1])[1]

	# filter by area
	for cnt in contours:
		contour_area = cv2.contourArea(cnt)
		if(min_area <= contour_area <= max_area):
			cv2.drawContours(mask_filtered, [cnt], 0, 255, -1)

	# count number of pixels within polygons
	pixel_count = np.count_nonzero(mask_filtered)

	return pixel_count > min_pixel_count
