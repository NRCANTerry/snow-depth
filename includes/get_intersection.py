# import necessary packages
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
from order_points import orderPoints
from progress_bar import progress
import math
import os

# function that returns the intersection of lines defined by two points
def lineIntersections(pt1, pt2, ptA, ptB):
	# tolerance
	DET_TOLERANCE = 0.00000001

	# first line
	x1, y1 = pt1
	x2, y2 = pt2
	dx1 = x2 - x1
	dy1 = y2 - y1

	# second line
	xA, yA = ptA
	xB, yB = ptB
	dx = xB - xA
	dy = yB - yA

	# calculate determinant
	# if DET is too small, lines are parallel
	DET = (-dx1 * dy + dy1 * dx)
	if math.fabs(DET) < DET_TOLERANCE: return (0,0)

	# find inverse determinant
	DETinv = 1.0/DET

	# find the sacalar amount along the "self" and input segments
	r = DETinv * (-dy  * (xA-x1) +  dx * (yA-y1))
	s = DETinv * (-dy1 * (xA-x1) + dx1 * (yA-y1))

	# return point of intersection
	xi = (x1 + r*dx1 + xA + s*dx)/2.0
	yi = (y1 + r*dy1 + yA + s*dy)/2.0
	return xi, yi

# function to determine the intersection point of stakes
# returns a dictionary indicating the coordinates of the
# intersection points for each stake
def getIntersections(imgs, boxCoords, stakeValidity, roiCoordinates, threshold, img_names, debug, debug_directory):

	# contains output data
	intersection_output = {}

	# number of images
	num_images = len(imgs)

	# create dictionary for images
	intersectionCoordinates = dict()

	# iterate through images
	for count, img_ in enumerate(imgs):
		# update progress bar
		progress(count + 1, num_images, status=img_names[count])

		# convert image to gray
		img_write = img_
		img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

		# get lowest blob coordinates
		coords = boxCoords[img_names[count]]

		# iterate through stakes
		for i, box in enumerate(coords):
			# only continue if stake is valid
			if(stakeValidity[img_names[count]][i]):
				# list for three different point combinations
				# measure intersection point using lines along left edge,
				# centroid and right edge of lowest blob
				coordinateCombinations = list()

				# points are already ordered
				# determine middle of box
				middleTop = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
				middleBottom = ((box[6][0] + box[7][0]) / 2, (box[6][1] + box[7][1]) / 2)

				# add combinations to list
				coordinateCombinations.append((middleTop, middleBottom)) # middle
				coordinateCombinations.append((box[0], box[7])) # left
				coordinateCombinations.append((box[1], box[6])) # right

				# iterate through combinations
				for j, points in enumerate(coordinateCombinations):
					# get points
					x0, y0 = points[0][0], points[0][1]
					x1, y1 = points[1][0], points[1][1]

					# get endpoint for line
					# intersection of line between points on blob with line defining bottom of stake
					x1, y1 = (lineIntersections((x0,y0), (x1,y1), (roiCoordinates[i][0][0][0], 
						roiCoordinates[i][0][1][1]), tuple(roiCoordinates[i][0][1])))
					
					# make a line with "num" points
					num = 500
					x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

					# extract values along the line
					lineVals = ndimage.map_coordinates(np.transpose(img), np.vstack((x,y)))

					# plot and save
					fig, axes = plt.subplots(nrows = 2)
					axes[0].imshow(img)
					axes[0].plot([x0, x1], [y0, y1], 'ro-')
					axes[0].axis('image')
					axes[1].plot(lineVals)

					filename, file_extension = os.path.splitext(img_names[count])
					plt.savefig((debug_directory + filename + 'stake' + str(i) + '-' + str(j) + file_extension))

					coords = [a for a, v in enumerate(lineVals) if v > threshold]

					first_coord = 0
					for k, coord in enumerate(coords):
						if ((coords[i+10] - coord) < 15):# and (coord - coords[i-2]) > 40:
							first_coord = coord
							break

					cv2.line(img_write, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0),2)
					cv2.circle(img_write, (int(x[first_coord]), int(y[first_coord])), 5, (0,255,0), 3)

				cv2.imwrite(debug_directory + img_names[count], img_write)