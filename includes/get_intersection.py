# import necessary packages
import numpy as np
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
from scipy import integrate

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

# function to adjust intersection lines towards the centre of the stake
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
def getIntersections(imgs, boxCoords, stakeValidity, roiCoordinates, img_names,
    debug, debug_directory):

    # create directory for pyplot images
    if(debug):
        signal_dir = debug_directory + "signals/"
        os.mkdir(signal_dir)

    # contains output data for JSON file
    intersection_output = {}

    # number of images
    num_images = len(imgs)

    # create output dictionaries for images
    # first dictionary holds intersection coordinates and second the distances
    # from the intersection coordinates to the upper and lower reference blobs
    intersectionCoordinates = dict()
    intersectionDistances = dict()

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

        # create list for distanes between reference blobs and intersection point
        stake_distances = list()

        # iterate through stakes
        for i, box in enumerate(blob_coords):

            # only continue if stake is valid
            if(stakeValidity[img_names[count]][i]):
                # list for three different point combinations
                # measure intersection point using lines along left edge,
                # centroid and right edge of lowest blob
                coordinateCombinations = list()

                # get valid coordinates from stake
                validCoordinates = [t for t in box if t != False]
                bottomBlob = validCoordinates[0]
                topBlob = validCoordinates[len(validCoordinates)-1]

                # determine centre of upper and lower reference blobs
                middleTop = (float(topBlob[0][0] + topBlob[2][0]) / 2.0, \
                    float(topBlob[0][1] + topBlob[2][1]) / 2.0)
                middleBottom = (float(bottomBlob[0][0] + bottomBlob[2][0]) / 2.0, \
                    float(bottomBlob[0][1] + bottomBlob[2][1]) / 2.0)

                # add combinations to list
                coordinateCombinations.append((middleTop, middleBottom)) # middle
                coordinateCombinations.append((topBlob[0], bottomBlob[3])) # left
                coordinateCombinations.append((topBlob[1], bottomBlob[2])) # right

                # combination names list
                combination_names = ["middle", "left", "right"]

                # dictionary containing coordinates
                coordinates = dict()

                # iterate through combinations
                for j, points in enumerate(coordinateCombinations):
                    # get points
                    x0, x1 = adjustCoords(points[0][0], points[1][0], 5, j)
                    y0, y1 = points[0][1], points[1][1]

                    # calculate line length
                    num = 1000 + ((roiCoordinates[i][1][1][1]-y1) * 4)

                    # get endpoint for line
                    # intersection of line between points on blob with line defining bottom of stake
                    x1, y1 = (lineIntersections((x0,y0), (x1,y1), (roiCoordinates[i][0][0][0],
                        roiCoordinates[i][0][1][1]), tuple(roiCoordinates[i][0][1])))
                    y0 = points[1][1]
                    x0, x1 = adjustCoords(points[1][0], x1, 5, j)

                    # make a line with "num" points
                    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

                    # extract values along the line
                    lineVals = ndimage.map_coordinates(np.transpose(img), np.vstack((x,y))).astype(np.float32)

                    # apply gaussian filter to smooth line
                    lineVals_smooth = ndimage.filters.gaussian_filter1d(lineVals, 10)

                    # append zero to signal to create peak
                    lineVals_smooth = np.append(lineVals_smooth, 0)

                    # determine peaks and properties
                    peaks, properties = find_peaks(lineVals_smooth, height=90, width=10)
                    peakWidthsOutput = signal.peak_widths(lineVals_smooth, peaks, rel_height = 0.5)
                    peakWidths = peakWidthsOutput[0]
                    minLineVal = min(lineVals)
                    maxLineVal = float(max(lineVals)) * 0.75

                    # get sorted indexes (decreasing distance down the line)
                    sorted_index = np.argsort(peaks)
                    last_index = sorted_index[len(sorted_index)-1]

                    # index of selected peak in sorted list of peaks
                    selected_peak = -1
                    major_peak = -1 # select larger peak for threshold calculation

                    # iterate through peaks from top to bottom
                    for index in sorted_index:
                        # determine snow cover before peak
                        peak_index = peaks[index]
                        left_edge = properties["left_ips"][index]
                        peak_range = lineVals[:int(left_edge)]
                        peak_intensity = lineVals[peak_index]
                        snow_threshold = peak_intensity * 0.65 if peak_intensity < 200 else 125
                        stake_cover = float(len(np.where(peak_range < snow_threshold)[0])) / float(len(peak_range))

                        # determine snow cover after peak
                        peak_range = lineVals[int(left_edge):]
                        snow_cover = float(len(np.where(peak_range > snow_threshold)[0])) / float(len(peak_range))

                        # get peak width and next peak width
                        peak_width = peakWidths[index]
                        if index != last_index:
                            peak_width_next = peakWidths[index+1]

                        # get proximity to next peak
                        if index != last_index:
                            proximity_peak = properties["left_ips"][index+1] - properties["right_ips"][index]

                        # get size of next peak
                        if(index != last_index): next_peak_height = lineVals[peaks[index+1]]

                        # get rgb value of peak
                        rgb_peak = img_[int(y[peak_index]), int(x[peak_index])]
                        rgb_proximity = np.amax(rgb_peak.astype(np.int32)) - np.amin(rgb_peak.astype(np.int32))
                        rgb_min = np.amin(rgb_peak.astype(np.int32))

                        # if peak meets conditions select it
                        if (
                            index != last_index # peak isn't last
                            and stake_cover > 0.5 # majority stake before peak
                            and (snow_cover > 0.5 or peak_width > 150 or (snow_cover > 0.35 and peak_width > 100)) # snow after peak
                            and (peak_intensity > maxLineVal or (next_peak_height > maxLineVal and proximity_peak < 100
                                and float(peak_intensity) / float(next_peak_height) > 0.5)) # peak is sufficiently large
                            and (peak_width > 100 or ((peak_width + peak_width_next > 100) and proximity_peak < 125
                                and (float(peak_width) / float(peak_width_next) > 0.20))) # peak is sufficiently wide
                            and rgb_proximity < 40 and rgb_min > 200 # peak is white in colour
                        ):
                            # select peak
                            selected_peak = index

                            # determine major peak
                            if(peak_intensity < maxLineVal and next_peak_height > maxLineVal and proximity_peak < 100):
                                major_peak = index + 1
                            else:
                                major_peak = index

                            # break loop
                            break

                        # last peak intersection conditions
                        elif (
                            index == last_index # last peak
                            and stake_cover > 0.4 # stake before peak
                            and peak_intensity > float(maxLineVal) * 0.75 # large enough
                            and (snow_cover > 0.33 or peak_index > float(len(lineVals)) * 0.75) # enough snow afterwards or near end
                            and rgb_proximity < 40 and rgb_min > 200 # peak is white in colour
                        ):
                            selected_peak = index
                            major_peak = index
                            break

                        # if peak resembling grass/ground is found exit
                        elif (
                            stake_cover > 0.4
                            and peak_intensity > float(maxLineVal) * 0.75
                            and (snow_cover > 0.33 or peak_index > float(len(lineVals)) * 0.75)
                            and rgb_proximity > 50 and rgb_min < 190 # peak is not white
                        ): break # exit

                    # calculate gradient of line
                    line_gradients_full = np.gradient(lineVals)
                    integral_full = integrate.cumtrapz(lineVals)

                    # if a snow case was found
                    if(selected_peak != -1):
                        # determine peak index in lineVals array
                        peak_index_line = np.uint32(peaks[selected_peak])

                        # determine threshold for finding stake
                        # average of intensity at left edge of peak and intensity at base of peak
                        left_edge_index = properties["left_ips"][major_peak]
                        left_edge_intensity = lineVals[int(left_edge_index)]
                        left_base_index = properties["left_bases"][selected_peak]
                        left_base_intensity = lineVals[int(left_base_index)] if lineVals[peak_index_line] < left_edge_index * 1.5 \
                             else lineVals[peak_index_line]
                        stake_threshold = (float(left_edge_intensity) - float(left_base_intensity)) / 2.0 + \
                                            float(left_base_intensity)

                        # restrict stake threshold
                        stake_threshold = 65 if stake_threshold < 65 else stake_threshold
                        stake_threshold = 125 if stake_threshold > 125 else stake_threshold
                        if(stake_threshold > 105 and max(lineVals) < 235): stake_threshold = 105

                        # determine index of intersection point
                        intersection_index = 0

                        # calculate gradients
                        line_gradients = np.gradient(lineVals)[0:peak_index_line][::-1]

                        # determine threshold for drop in intensity
                        # varies based on lighting conditions
                        maximum_drop = max(x for x in line_gradients_full if x < 10)
                        drop_threshold = maximum_drop * 0.333

                        # iterate through points prior to peak
                        for t, intensity in enumerate(reversed(lineVals[:peak_index_line])):
                            # calculate maximum drop near point
                            max_drop = max(line_gradients.tolist()[t-25:t+25]) if (t>25 and t<(len(line_gradients)-25)) \
                                else 0

                            # converted index
                            conv_index = peak_index_line-t

                            # if below threshold or large drop
                            if(
                                (intensity < stake_threshold
                                    and (max_drop > drop_threshold or max_drop == 0))
                                or (line_gradients[t] > 25 and min(lineVals[conv_index-25:conv_index].tolist()) \
                                    < stake_threshold * 1.25)
                            ):
                                intersection_index = conv_index
                                break

                        # overlay debugging points
                        if(debug):
                            cv2.line(img_write, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0),2)
                            cv2.circle(img_write, (int(x[intersection_index]), int(y[intersection_index])), 5, (0,255,0), 3)

                    else: peak_index_line = 0

                    # add coordinates to dictionary
                    if(selected_peak != -1 and intersection_index != 0):
                        coordinates[combination_names[j]] = (x[intersection_index], y[intersection_index])
                    else:
                        coordinates[combination_names[j]] = (False, False)

                    # if in debugging mode
                    if debug:
                        # plot and save
                        fig, axes = plt.subplots(nrows = 4)
                        axes[0].imshow(img)
                        axes[0].plot([x0, x1], [y0, y1], 'ro-')
                        axes[0].axis('image')
                        axes[1].plot(lineVals)
                        axes[1].plot(peaks, lineVals[peaks], "x")
                        axes[1].plot(peak_index_line, lineVals[peak_index_line], "x")
                        axes[2].plot(line_gradients_full)
                        axes[3].plot(integral_full)

                        # only show if valid intersction point found
                        if selected_peak != -1:
                            axes[1].vlines(x=peak_index_line, ymin=lineVals[peak_index_line] - properties["prominences"][selected_peak],
                                ymax=lineVals[peak_index_line], color="C1")
                            axes[1].hlines(y=properties["width_heights"][selected_peak], xmin=properties["left_ips"][selected_peak],
                                xmax=properties["right_ips"][selected_peak], color = "C1")
                            axes[1].hlines(*peakWidthsOutput[1:], color = "C2")
                            axes[1].axvline(x=properties["left_bases"][selected_peak], color = 'b')
                            axes[1].axvline(x=properties["left_ips"][selected_peak], color = 'y')
                            axes[1].axvline(x=intersection_index,color='r')

                        filename, file_extension = os.path.splitext(img_names[count])
                        plt.savefig((signal_dir + filename + 'stake' + str(i) + '-' + str(j) + file_extension))
                        plt.close()

                # calculate median intersection point and filter out combinations where no intersection point was found
                y_vals = [x[1] for x in [coordinates["left"], coordinates["right"], coordinates["middle"]]]
                y_vals = [x for x in y_vals if x != False]
                x_vals = [x[0] for x in [coordinates["left"], coordinates["right"], coordinates["middle"]]]
                x_vals = [x for x in x_vals if x != False]

                # append to dictionary
                if(len(x_vals) > 1 and len(y_vals) > 1):
                    median_y = statistics.median(y_vals)
                    median_x = statistics.median(x_vals)
                    coordinates["average"] = [median_x, median_y]
                # if no intersection point append False to dictionary
                else:
                    coordinates["average"] = [False, False]

                # add to stake coordinates list
                stake_intersections.append(coordinates)

                # add distances to list
                distances_list = list()
                if(coordinates["average"] != [False, False]):
                    num_blobs = len(box)
                    validDistances = [t for t in box if t != False]
                    offset = abs(float(validDistances[0][2][0] - validDistances[0][0][0])) / num_blobs
                    for q, v in enumerate(box):
                        if(v != False):#and selected_peak != -1
                            # calculate centre of blob
                            middle = (float(v[0][0] + v[2][0]) / 2.0, float(v[0][1] + v[2][1]) / 2.0)
                            distances_list.append(math.hypot(coordinates["average"][0] - middle[0], \
                                coordinates["average"][1] - middle[1]))

                            # overlay debugging points
                            cv2.circle(img_write, (int(middle[0]), int(middle[1])), 5, (0,255,255), 3)
                            cv2.line(img_write, (int(middle[0] + (q-(num_blobs/2.0)) * offset), int(middle[1])), \
                                (int(median_x + (q-(num_blobs/2.0)) * offset), int(median_y)), (0,255,255), 2)
                        else:
                            distances_list.append(False)

                # add to stake distance list
                stake_distances.append(distances_list)

            # if stake isn't valid append empty dictionary and list
            else:
                stake_intersections.append(dict())
                stake_distances.append(list())

        # if in debugging mode
        if(debug):
            # create temporary dictionaries
            stake_dict = dict()
            stake_dict_dist = dict()

            # add data to output
            for x in range(0, len(blob_coords)):
                stake_dict['stake' + str(x)] = stake_intersections[x]
                stake_dict_dist['stake' + str(x)] = stake_distances[x]

            # add data to output
            intersection_output[img_names[count]] = {
                "Coordinates": stake_dict,
                "Measurements": stake_dict_dist
            }

            # output image to debug directory
            cv2.imwrite(debug_directory + img_names[count], img_write)

        # add data to return dictionaries
        intersectionCoordinates[img_names[count]] = stake_intersections
        intersectionDistances[img_names[count]] = stake_distances

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory + 'stakes.json', 'w')
        json.dump(intersection_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # return dictionary
    return intersectionCoordinates, intersectionDistances
