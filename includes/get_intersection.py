# import necessary packages
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
from order_points import orderPoints
from progress_bar import progress
import math
import os
import json
import statistics

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

	# contains output data for JSON file
	intersection_output = {}

	# number of images
	num_images = len(imgs)

	# create output dictionary for images
	intersectionCoordinates = dict()

	# iterate through images
	for count, img_ in enumerate(imgs):
		# update progress bar
		progress(count + 1, num_images, status=img_names[count])

		# convert image to gray
		img_write = img_.copy()
		img = cv2.cvtColor(img_.copy(), cv2.COLOR_BGR2GRAY)

		# get top and bottom blob coordinates
		blob_coords = boxCoords[img_names[count]]

		# create list for coordinates on stakes
		stake_intersections = list()

		# iterate through stakes
		for i, box in enumerate(blob_coords):
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

				# combination names list
				combination_names = ["middle", "left", "right"]

				# dictionary containing coordinates
				coordinates = dict()

				# iterate through combinations
				for j, points in enumerate(coordinateCombinations):
					# get points
					x0, y0 = points[0][0], points[0][1]
					x1, y1 = points[1][0], points[1][1]

					if j == 1:
						x0 += 5
						x1 += 5
					elif j == 2:
						x0 -= 5
						x1 -= 5

					# get endpoint for line
					# intersection of line between points on blob with line defining bottom of stake
					x1, y1 = (lineIntersections((x0,y0), (x1,y1), (roiCoordinates[i][0][0][0],
						roiCoordinates[i][0][1][1]), tuple(roiCoordinates[i][0][1])))

					# make a line with "num" points
					num = 2000
					x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

					# extract values along the line
					lineVals = ndimage.map_coordinates(np.transpose(img), np.vstack((x,y)))

					# get coordinates that have intensity greater than threshold
					# and aren't a colour other than white
					coords = [a for a, v in enumerate(lineVals) if v > threshold]

					# further filter those values to get intersection point

					#coords_filtered = [a for a, v in enumerate(lineVals) if (v > threshold and
					#	img_[int(y[v]),int(x[v])][1] > 150 and \
		 		#		abs(img_[int(y[v]),int(x[v])][0] - \
				#		img_[int(y[v]),int(x[v])][2]) <= 50)]
				#	print "Coords filtered ", coords_filtered

					first_coord = 0
					for k, coord in enumerate(coords):
						if ((len(coords) > k+100) and (coords[k+100] - coord) <= 105 and y[coord] > box[7][1]):
							first_coord = coord
							break
					if(first_coord != 0):
						cv2.line(img_write, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0),2)
						cv2.circle(img_write, (int(x[first_coord]), int(y[first_coord])), 5, (0,255,0), 3)

					# add coordinates to dictionary
					if(first_coord != 0):
						coordinates[combination_names[j]] = (x[first_coord], y[first_coord])
					else:
						coordinates[combination_names[j]] = (False, False)

					# if in debugging mode
					if debug:
						# plot and save
						fig, axes = plt.subplots(nrows = 2)
						axes[0].imshow(img)
						axes[0].plot([x0, x1], [y0, y1], 'ro-')
						axes[0].axis('image')
						axes[1].plot(lineVals)
						axes[1].axvline(x=first_coord,color='g')

						filename, file_extension = os.path.splitext(img_names[count])
						plt.savefig((debug_directory + filename + 'stake' + str(i) + '-' + str(j) + file_extension))
						plt.close()

				# calculate median intersection point and filter out combinations where no intersection point was found
				y_vals = [x[1] for x in [coordinates["left"], coordinates["right"], coordinates["middle"]]]
				y_vals = [x for x in y_vals if x != False]
				x_vals = [x[0] for x in [coordinates["left"], coordinates["right"], coordinates["middle"]]]
				x_vals = [x for x in x_vals if x != False]

				# append to dictionary
				if(len(x_vals) > 0 and len(y_vals) > 0):
					median_y = statistics.median(y_vals)
					median_x = statistics.median(x_vals)
					coordinates["average"] = [median_x, median_y]
				# if no intersection point append False to dictionary
				else:
					coordinates["average"] = [False, False]

				# add to stake coordinates list
				stake_intersections.append(coordinates)

			# if stake isn't valid append empty dictionary
			else:
				stake_intersections.append(dict())

		# if in debugging mode
		if(debug):
			# create temporary dictionary
			stake_dict = dict()

			# add data to output
			for x in range(0, len(blob_coords)):
				stake_dict['stake' + str(x)] = stake_intersections[x]

			# add data to output
			intersection_output[img_names[count]] = stake_dict

			cv2.imwrite(debug_directory + img_names[count], img_write)

		# add data to return dictionary
		intersectionCoordinates[img_names[count]] = stake_intersections

	# if in debugging mode
	if(debug):
		# output JSON file
		file = open(debug_directory + 'stakes.json', 'w')
		json.dump(intersection_output, file, sort_keys=True, indent=4, separators=(',', ': '))

	# return dictionary
	return intersectionCoordinates
