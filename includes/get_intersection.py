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
from scipy.signal import find_peaks
from scipy import signal
from scipy import ndimage

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

# function to adjust move intersection lines towards the centre of the stake
# preventing incorrect snow depth measurements
def adjustCoords(x0, x1, degree, status):
	if(status == 1):
		return x0+5, x1+5
	elif(status == 2):
		return x0-5, x1-5
	else:
		return x0, x1

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
		img_hsv = cv2.split(cv2.cvtColor(img_.copy(), cv2.COLOR_BGR2HSV))[1]

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
					x0, x1 = adjustCoords(points[0][0], points[1][0], 5, j)
					y0, y1 = points[0][1], points[1][1]

					# get endpoint for line
					# intersection of line between points on blob with line defining bottom of stake
					x1, y1 = (lineIntersections((x0,y0), (x1,y1), (roiCoordinates[i][0][0][0],
						roiCoordinates[i][0][1][1]), tuple(roiCoordinates[i][0][1])))
					y0 = points[1][1]
					x0, x1 = adjustCoords(points[1][0], x1, 5, j)

					# make a line with "num" points
					num = 1000
					x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

					# extract values along the line
					lineVals = ndimage.map_coordinates(np.transpose(img), np.vstack((x,y)))

					# apply gaussian filter to smooth line
					lineVals_smooth = ndimage.filters.gaussian_filter1d(lineVals, 10)

					# append zero to signal to create peak
					lineVals_smooth = np.append(lineVals_smooth, 0)

					# determine peaks and properties
					peaks, properties = find_peaks(lineVals_smooth, height=100, prominence=1, width=10)

					# get sorted indexes (decreasing distance down the line)
					sorted_index = np.argsort(peaks)
					sorted_index = sorted_index[::-1]

					# index of selected peak in sorted list of peaks
					selected_peak = -1

					# iterate through peaks from bottom to top
					for index in sorted_index:
						# only check if there is more than 1 peak remaining
						if(index > 0):
							# check that peak is isolated (doesn't have peak immediately next to it
							# of similar size)
							if(properties["left_ips"][index] - properties["right_ips"][index-1] > 50
								or properties["peak_heights"][index-1] < properties["peak_heights"][index-1] * 0.5):
								selected_peak = index
								break
						# else select the only peak remaining
						else:
							# determine if this is a no snow case
							# must see mostly snow after peak (50% coverage)
							# snow threshold is 75% of peak
							peak_index = peaks[index]
							peak_intensity = lineVals[peak_index]
							peak_range = lineVals[peak_index:]
							snow_cover = float(len(np.where(peak_range > peak_intensity * 0.75)[0])) / float(len(peak_range)) if \
								peak_intensity * 0.75 < 140 else float(len(np.where(peak_range > 140)[0])) / float(len(peak_range))

							if(snow_cover > 0.5 or float(len(peak_range)) / float(len(lineVals)) < 0.15):
								selected_peak = 0
							else:
								selected_peak = -1
							break

					# if a snow case was found
					if(selected_peak != -1):
						# determine peak index in lineVals array
						peak_index_line = peaks[selected_peak]

						# determine threshold for finding stake
						# average of intensity at left edge of peak and intensity at base of peak
						left_edge_index = properties["left_ips"][selected_peak]
						left_edge_intensity = lineVals[int(left_edge_index)]
						left_base_index = properties["left_bases"][selected_peak]
						left_base_intensity = lineVals[int(left_base_index)]
						stake_threshold = (float(left_edge_intensity) - float(left_base_intensity)) / 2.0 + \
											float(left_base_intensity)

						# restrict stake threshold
						stake_threshold = 65 if stake_threshold < 65 else stake_threshold
						stake_threshold = 115 if stake_threshold > 115 else stake_threshold

						# determine index of intersection point
						intersection_index = 0
						for t, intensity in enumerate(reversed(lineVals[:peak_index_line])):
							if(intensity < stake_threshold):
								intersection_index = int(peak_index_line-t)
								break

						# overlay debugging points
						if(debug):
							cv2.line(img_write, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0),2)
							cv2.circle(img_write, (int(x[intersection_index]), int(y[intersection_index])), 5, (0,255,0), 3)

					# add coordinates to dictionary
					if(selected_peak != -1 and intersection_index != 0):
						coordinates[combination_names[j]] = (x[intersection_index], y[intersection_index])
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
						axes[1].plot(peak_index_line, lineVals[peak_index_line], "x")

						# only show if valid intersction point found
						if selected_peak != -1:
							axes[1].vlines(x=peak_index_line, ymin=lineVals[peak_index_line] - properties["prominences"][selected_peak],
								ymax=lineVals[peak_index_line], color="C1")
							axes[1].hlines(y=properties["width_heights"][selected_peak], xmin=properties["left_ips"][selected_peak],
								xmax=properties["right_ips"][selected_peak], color = "C1")
							axes[1].axvline(x=properties["left_bases"][selected_peak], color = 'b')
							axes[1].axvline(x=properties["left_ips"][selected_peak], color = 'y')
							axes[1].axvline(x=intersection_index,color='r')

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
