# import necessary packages
import numpy as np
from matplotlib import pyplot as plt
import cv2
from order_points import orderPoints
import math
import os
import json
import statistics
from scipy.signal import find_peaks
from scipy import signal
from scipy import ndimage
import tqdm
import time

def lineIntersections(pt1, pt2, ptA, ptB):
    '''
    Function that returns the intersection of lines defined by two points
    @param pt1 first point of line A
    @param pt2 second point of line A
    @param ptA first point on line B
    @param ptB second point on line B

    @type pt1 float
    @type pt2 float
    @type ptA float
    @type ptB float

    @return xi, yi coordinates of intersection point
    @rtype (float, float)
    '''

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

def adjustCoords(x0, x1, degree, type):
    '''
    Function to adjust intersection lines towards the centre of the stake
    preventing incorrect snow depth measurements
    @param x0 first x coordinate
    @param x1 second x coordinate
    @param degree magnitude of adjustment
    @param type indicator for type of adjustment
    '''

    if(type == 1):
        return x0+degree, x1+degree # adjust right
    elif(type == 2):
        return x0-degree, x1-degree # adjust left
    else:
        return x0, x1

def intersect(img, boxCoords, stakeValidity, roiCoordinates, name, debug,
    debug_directory, signal_dir, params, tensors, upper_border, signal_var,
    template_tensors):
    '''
    Function to get intersection coordinates and distances for an image
    '''

    # convert image to grayscale
    img_write = img.copy()
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # create list for coordinates on stake
    stake_intersections = list()

    # create list for distances between blobs and intersection point
    stake_distances = list()

    # iterate through stakes
    for i, box in enumerate(boxCoords):
        # if stake is valid
        if(stakeValidity[i]):
            # list for three difference point combinations
            # measure intersection point using lines along left edge, centre
            # and right edge of lowest blob
            coordinateCombinations = list()

            # get valid blob coordinates from stake
            validCoordinates = [t for t in box if t != False]
            bottomBlob = validCoordinates[0]
            topBlob = validCoordinates[len(validCoordinates)-1]

            # combination names list
            combination_names = ["middle", "left", "right"]

            # dictionary containing coordinates
            coordinates = dict()

            # determine x and y coordinates of three lines
            x0, x1 = ((topBlob[0][0] + topBlob[2][0]) / 2.0, (bottomBlob[0][0] + \
                bottomBlob[2][0]) / 2.0)
            y0, y1 = ((topBlob[0][1] + topBlob[1][1]) / 2.0, (bottomBlob[0][1] +  \
                bottomBlob[1][1]) / 2.0)
            temp = (x1, y1)

            # determine degree to move line in
            lineShift = float(abs(topBlob[0][0] - topBlob[2][0])) / 3.0

            # get endpoint for line
            # intersection of line between points on blob with line defining bottom of stake
            x1, y1 = (lineIntersections((x0,y0), (x1,y1), (roiCoordinates[i][0][0][0],
                roiCoordinates[i][0][1][1]), tuple(roiCoordinates[i][0][1])))
            x0, y0 = temp # start line from bottom blob

            # calculate line length
            line_length = np.hypot(x1-x0, y1-y0)

            # adjust line length so 1pt represents 1mm
            line_length *= tensors[i] if tensors[i] != True else template_tensors[i]

            # add combinations to list
            coordinateCombinations.extend([
                ((x0, y0), (x1, y1)), # middle
                ((x0-lineShift, y0), (x1-lineShift, y1)), # left
                ((x0+lineShift, y0), (x1+lineShift, y1)) # right
            ])

            # isolate bottom of stake roi
            stake_bottom_roi = img_write.copy()[int(y0):int(roiCoordinates[i][0][1][1]),
                int(roiCoordinates[i][0][0][0]):int(roiCoordinates[i][0][1][0])]

            # find edges in roi
            gray_roi = cv2.cvtColor(stake_bottom_roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            edges = cv2.Canny(gray_roi, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))

            # calculate parameters for Hough Line Transform
            maxLineGap = 75.0 / tensors[i] if tensors[i] != True else template_tensors[i] # 75mm
            minLineLength = 20.0 / tensors[i] if tensors[i] != True else template_tensors[i] # 20mm

            # Remove all but vertical edges
            edgeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, edgeKernel)

            # find lines
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=100,
                maxLineGap=maxLineGap, minLineLength=minLineLength)

            # isolate lines along stake boundary
            v_stake_lines = list()
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        x_coord = x1 + roiCoordinates[i][0][0][0]
                        blob_shift = lineShift*2

                        # if within stake boundary
                        if (x_coord < bottomBlob[0][0] and x_coord > bottomBlob[0][0] - blob_shift) or \
                            (x_coord > bottomBlob[2][0] and x_coord < bottomBlob[2][0] + blob_shift):
                            v_stake_lines.append(line)

            # find lowest stake line (near intersection point)
            lowest_edge_y = -1
            if params[5]: # find lowest stake edge if using robust intersection
                for line in v_stake_lines:
                    for x1, y1, x2, y2 in line:
                        # draw line on debugging image
                        if debug:
                            cv2.line(img_write, (int(x1+roiCoordinates[i][0][0][0]), int(y1+y0)),
                                (int(x2+roiCoordinates[i][0][0][0]), int(y2+y0)), (0, 255, 0), 2)

                        # update lowest edge variable
                        if y2 + y0 > lowest_edge_y and y2 > y1:
                            lowest_edge_y = y2 + y0
                        elif y1 + y0 > lowest_edge_y:
                            lowest_edge_y = y1 + y0

            # iterate through combinations
            for j, ((x0, y0), (x1, y1)) in enumerate(coordinateCombinations):
                # make a line with 1pt per mm
                x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)

                # overlay line on image
                if(debug):
                    cv2.line(img_write, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0),2)

                # extract values along the line
                lineVals = ndimage.map_coordinates(np.transpose(img), np.vstack((x,y))).astype(np.float32)

                # apply gaussian filter to smooth line
                lineVals_smooth = ndimage.filters.gaussian_filter1d(lineVals, 10)

                # append zero to signal to create peak
                lineVals_smooth = np.append(lineVals_smooth, 0)

                # determine peaks and properties
                peaks, properties = find_peaks(lineVals_smooth, height=params[0], width=10)
                peakWidthsOutput = signal.peak_widths(lineVals_smooth, peaks, rel_height = 0.75)
                peakWidths = peakWidthsOutput[0]
                minLineVal = min(lineVals)
                maxLineVal = float(max(lineVals)) * 0.65

                # get sorted indexes (decreasing distance down the line)
                sorted_index = np.argsort(peaks)
                last_index = sorted_index[len(sorted_index)-1] if len(sorted_index) > 0 else 0

                # index of selected peak in sorted list of peaks
                selected_peak = -1
                major_peak = -1 # select larger peak for threshold calculation

                # determine index of last known stake edge
                index_edge = 0
                if lowest_edge_y != -1:
                    index_edge = min(range(len(y)), key=lambda i: abs(y[i]-lowest_edge_y))

                # iterate through peaks from top to bottom
                for index in sorted_index:
                    # determine snow cover before peak
                    peak_index = peaks[index]
                    left_edge = properties["left_ips"][index]
                    peak_range = lineVals[:int(left_edge)]
                    peak_intensity = lineVals[peak_index]
                    snow_threshold = peak_intensity * params[1] if peak_intensity < 200 else params[2]
                    stake_cover = float(len(np.where(peak_range < snow_threshold)[0])) / float(len(peak_range))
                    right_edge = properties["right_ips"][index]

                    # determine snow cover after peak
                    peak_range = lineVals[int(left_edge):]
                    snow_cover = float(len(np.where(peak_range > snow_threshold)[0])) / (float(len(peak_range)) - \
                        float(len(np.where(peak_range == 0)[0]))) # don't count image border

                    # get peak width and next peak width
                    peak_width = peakWidths[index]
                    if index != last_index:
                        peak_width_next = peakWidths[index+1]

                    # get proximity to next peak
                    if index != last_index:
                        proximity_peak = properties["left_ips"][index+1] - properties["right_ips"][index]

                    # get size of next peak
                    if(index != last_index): next_peak_height = lineVals[peaks[index+1]]

                    # determine if stake is visible beyond peak
                    if index != last_index:
                        minimum_between_peaks = np.amin(lineVals[peak_index:peaks[index+1]])
                        distance_between_peaks = properties["left_ips"][index+1] - properties["right_ips"][index]

                        # determine if there is a significant amount of stake before the edge index
                        edgeValid = True # flag for whether major amount of stake remaining
                        if right_edge < index_edge:
                            remaining_range = lineVals[int(right_edge):int(index_edge)]
                            remaining_stake = (float(len(np.where(remaining_range < 100)[0])) - float(len(np.where(remaining_range == 0)[0]))) \
                                / float(len(remaining_range))
                            index_edge_prop = index_edge / float(line_length) # don't consider this check if stake edge extends to bottom
                            if remaining_stake > 0.4 and index_edge_prop < 0.75: edgeValid = False # if more than 40% of remaining range is stake

                    # if peak meets conditions select it
                    if (
                        index != last_index # peak isn't last
                        and stake_cover > params[3] # majority stake before peak
                        and (snow_cover > params[4] or (snow_cover > params[4] * 0.666 and peak_width > 150)) # snow after peak
                        and (peak_intensity > maxLineVal or (next_peak_height > maxLineVal and proximity_peak < 75
                            and float(peak_intensity) / float(next_peak_height) > 0.5) or (y[int(right_edge)] > lowest_edge_y and lowest_edge_y != -1))
                        and (peak_width > 50 or ((peak_width + peak_width_next > 75) and proximity_peak < 100)
                            or (minimum_between_peaks < 100 and y[int(right_edge)] > lowest_edge_y and lowest_edge_y != -1 and peak_width > 25))
                        and (minimum_between_peaks > 100 or distance_between_peaks > 200 or (y[int(right_edge)]>lowest_edge_y and lowest_edge_y != -1))
                        and edgeValid # ensure no more large amounts of stake remain
                    ):
                        # select peak
                        selected_peak = index

                        # determine major peak
                        if(peak_intensity < maxLineVal and next_peak_height > maxLineVal and proximity_peak < 100):
                            major_peak = index + 1
                        else:
                            major_peak = index
                        break

                    # last peak intersection conditions
                    elif (
                        index == last_index # last peak
                        and stake_cover > params[3] # stake before peak
                        and peak_intensity > float(maxLineVal) * 0.75 # large enough
                        and (snow_cover > params[4] * 0.666 or peak_index > float(len(lineVals)) * 0.75) # enough snow afterwards or near end
                    ):
                        selected_peak = index
                        major_peak = index
                        break

                # calculate gradient of line
                line_gradients_full = np.gradient(lineVals)

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
                            #or (line_gradients[t] > 25 and min(lineVals[conv_index-25:conv_index].tolist()) \
                            #    < stake_threshold * 1.5)
                        ):
                            # if intersection index is beyond edge of strong peak decrease threshold slightly
                            stake_threshold_updated = False
                            if intersection_index == 0 and max(lineVals) < 230: # don't run on bright scenes
                                for a, robustEdge in enumerate(peakWidthsOutput[2][selected_peak:]):
                                    if conv_index > robustEdge and lineVals[int(robustEdge)] > 65 and \
                                        abs(robustEdge-conv_index) < 50:
                                        stake_threshold = lineVals[int(robustEdge)] + \
                                            (abs(stake_threshold - lineVals[int(robustEdge)]) / 4.0)
                                        conv_index = robustEdge
                                        stake_threshold_updated = True

                                # if edge estimate and robust peak edge agree
                                #if conv_index - index_edge < 20:
                                #    intersection_index = int(conv_index)
                                #    break

                                if stake_threshold_updated:
                                    intersection_index = int(conv_index)
                                    continue # run again

                            # if there is a large drop nearby choose it
                            if not stake_threshold_updated:
                            #    max_nearby = max(line_gradients.tolist()[t:t+10]) if (t<len(line_gradients)-25) else 0
                            #    if max_nearby > line_gradients[peak_index_line-t] * 2.0 and max(lineVals) < 240:
                            #        if(lineVals[int(conv_index - np.argmax(line_gradients.tolist()[t:t+10]))] > 55):
                            #            conv_index -= np.argmax(line_gradients.tolist()[t:t+10])

                                # set intersection index
                                intersection_index = conv_index
                                break

                    # overlay debugging points
                    if(debug):
                        cv2.circle(img_write, (int(x[intersection_index]), int(y[intersection_index])), 5, (0,255,0), 3)

                else: peak_index_line = 0

                # add coordinates to dictionary
                if(selected_peak != -1 and intersection_index != 0):
                    coordinates[combination_names[j]] = (x[intersection_index], y[intersection_index])
                else:
                    coordinates[combination_names[j]] = (False, False)

                # if in debugging mode
                if debug and signal_var:
                    # plot and save
                    fig, axes = plt.subplots(nrows = 2)
                    axes[0].imshow(img)
                    axes[0].plot([x0, x1], [y0, y1], 'ro-')
                    axes[0].axis('image')
                    axes[1].plot(lineVals)
                    axes[1].plot(peaks, lineVals[peaks], "x")
                    axes[1].plot(peak_index_line, lineVals[peak_index_line], "x")

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
                        axes[1].axvline(x=index_edge, color='pink')

                    filename, file_extension = os.path.splitext(name)
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

    # create temporary dictionaries
    stake_dict = dict()
    stake_dict_dist = dict()

    # if in debugging mode
    if(debug):
        # add data to output
        for x in range(0, len(boxCoords)):
            stake_dict['stake' + str(x)] = stake_intersections[x]
            stake_dict_dist['stake' + str(x)] = stake_distances[x]

        # output image to debug directory
        cv2.imwrite(debug_directory + name, img_write)

    # return stake intersections, distances and JSON output
    return stake_intersections, stake_distances, stake_dict, stake_dict_dist, name

def getIntersections(imgs, boxCoords, stakeValidity, roiCoordinates, img_names,
    debug, debug_directory, params, tensors, upper_border, imageSummary, signal_var,
    template_tensors):
    '''
    Function to get intersection coordinates and distances for an image set
    '''

    # create directory for signal images
    if(debug):
        signal_dir = debug_directory + "signals/"
        os.mkdir(signal_dir)

    # contains output data for JSON file
    intersection_output = {}

    # output dictionaries for images
    # intersectionCoordinates holds intersection coordinates
    # intersectionDistances holds distances from blobs to intersection point
    intersectionCoordinates = dict()
    intersectionDistances = dict()

    # iterator
    count = 0

    # iterate through images
    for img_ in tqdm.tqdm(imgs):
        # get image name
        imgName = img_names[count]

        # get intersection points, distances and JSON output
        stake_intersections, stake_distances, stake_dict, stake_dict_dist, _ = intersect(img_,
            boxCoords[imgName], stakeValidity[imgName], roiCoordinates, imgName,
            debug, debug_directory, signal_dir, params, tensors[imgName], upper_border,
            signal_var, template_tensors)

        if(debug):
            # add data to output
            intersection_output[imgName] = {
                "Coordinates": stake_dict,
                "Measurements": stake_dict_dist
            }

        # add data to return dictionaries
        intersectionCoordinates[imgName] = stake_intersections
        intersectionDistances[imgName] = stake_distances

        # add to image summary
        imageSummary[imgName][" "] = ""
        imageSummary[imgName]["Stake (Intersection Points)"] = "x (px)                 y (px)  "
        for e, point in enumerate(stake_intersections):
            if "average" in point.keys() and not False in point["average"]: # valid point found
                x, y = np.round(point["average"], 2)
                imageSummary[imgName]["   %d " % (e+1)] = "%0.2f              %0.2f" % (x, y)
            else:
                imageSummary[imgName]["   %d " % (e+1)] = "%s                     %s    " % ("n/a", "n/a")

        # increment iterator
        count += 1

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory + 'stakes.json', 'w')
        json.dump(intersection_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # return dictionaries
    return intersectionCoordinates, intersectionDistances, imageSummary

def unpackArgs(args):
    '''
    Function to unpack arguments explicitly
    @param args function arguments
    @type args arguments
    @return output of intersect function
    @rtype list
    '''
    return intersect(*args)

def getIntersectionsParallel(pool, imgs, boxCoords, stakeValidity, roiCoordinates,
    img_names, debug, debug_directory, params, tensors, upper_border, imageSummary,
    signal_var, template_tensors):
    '''
    Function to get intersection coordinates and distances for an image set using
        a parallel pool to improve efficiency
    '''

    # create directory for signal images
    if(debug):
        signal_dir = debug_directory + "signals/"
        os.mkdir(signal_dir)

    # contains output data for JSON file
    intersection_output = {}

    # output dictionaries for images
    # intersectionCoordinates holds intersection coordinates
    # intersectionDistances holds distances from blobs to intersection point
    intersectionCoordinates = dict()
    intersectionDistances = dict()

    # create task list for pool
    tasks = list()
    for i, img in enumerate(imgs):
        imgName = img_names[i]
        tasks.append((img, boxCoords[imgName], stakeValidity[imgName], roiCoordinates,
        imgName, debug, debug_directory, signal_dir, params, tensors[imgName], upper_border,
        signal_var, template_tensors))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total=len(tasks)):
        # unpack outupt
        stake_intersections, stake_distances, stake_dict, stake_dict_dist, imgName = i

        if(debug):
            # add data to output
            intersection_output[imgName] = {
                "Coordinates": stake_dict,
                "Measurements": stake_dict_dist
            }

        # add data to return dictionaries
        intersectionCoordinates[imgName] = stake_intersections
        intersectionDistances[imgName] = stake_distances

        # add to image summary
        imageSummary[imgName][" "] = ""
        imageSummary[imgName]["Stake (Intersection Points)"] = "x (px)                 y (px)  "
        for e, point in enumerate(stake_intersections):
            if "average" in point.keys() and not False in point["average"]: # valid point found
                x, y = np.round(point["average"], 2)
                imageSummary[imgName]["   %d " % (e+1)] = "%0.2f              %0.2f" % (x, y)
            else:
                imageSummary[imgName]["   %d " % (e+1)] = "%s                     %s    " % ("n/a", "n/a")

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory + 'stakes.json', 'w')
        json.dump(intersection_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # return dictionaries
    return intersectionCoordinates, intersectionDistances, imageSummary
