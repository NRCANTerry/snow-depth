# add to python path
import sys

sys.path.append('./includes')

# import necessary modules
import cv2
import os
import numpy as np
import json
import imutils
from progress_bar import progress
from equalize import equalize_hist
from register import alignImages
from filter_night import isDay
from check_stakes import getValidStakes
from GUI import GUI
from get_intersection import getIntersections
import Tkinter as tk
import datetime
import time

root = tk.Tk()
gui = GUI(root)
root.mainloop()

# get parameters
params = gui.getValues()

# start timer
start = time.time()

# ---------------------------------------------------------------------------------
# Get parameters from GUI
# ---------------------------------------------------------------------------------

# window closed without executing
if(params == False):
    sys.exit()

# update parameters
directory = params[0] + "/"
lower_hsv1 = params[1]
upper_hsv1 = params[2]
lower_hsv2 = params[3]
upper_hsv2 = params[4]
img_border_upper = params[5]
img_border_lower = params[6]
blob_size_lower = params[7]
blob_size_upper = params[8]
roi_coordinates = params[9]
template_path = params[10]
clip_limit = params[11]
tile_size = tuple(params[12])
template_intersections = params[14]
template_tensor = params[15]

# flag to run program in debug mode
debug = params[13]

# other parameters
median_kernal_size = 5
dilate_kernel = (5, 5)
angle_thresh = -45
bar_width_low = 15
bar_width_high = 300

# ---------------------------------------------------------------------------------
# Create Directories
# ---------------------------------------------------------------------------------

# dictionary of paths
paths_dict = dict()

# create directories
if(not os.path.isdir("measure-depth")):
    os.mkdir("./measure-depth")

# add folder for run
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "-").replace(":", "-")
path = "./measure-depth/" + date
os.mkdir(path)

# add optional directories
paths_dict["equalized"] = path + "/equalized/"
paths_dict["equalized-template"] = path + "/equalized-template/"
paths_dict["registered"] = path + "/registered/"
paths_dict["matches"] = path + "/matches/"
paths_dict["template-overlay"] = path + "/template-overlay/"
paths_dict["stake-check"] = path + "/stake-check/"
paths_dict["intersection"] = path + "/intersection/"
paths_dict["testing"] = path + "/testing/"

if(debug):
    os.mkdir(paths_dict["equalized"])
    os.mkdir(paths_dict["equalized-template"])
    os.mkdir(paths_dict["registered"])
    os.mkdir(paths_dict["matches"])
    os.mkdir(paths_dict["template-overlay"])
    os.mkdir(paths_dict["stake-check"])
    os.mkdir(paths_dict["intersection"])
    os.mkdir(paths_dict["testing"])

# ---------------------------------------------------------------------------------
# Filter Out Night Images
# ---------------------------------------------------------------------------------

# get images
images = [file_name for file_name in os.listdir(directory)]
num_images = len(images)

# list for filtered images
images_filtered = list()
filtered_names = list()

print("\nFiltering Night Images")

for count, img_name in enumerate(images):
    # update progress bar
    progress(count + 1, num_images, status=img_name)

    # read in image
    img = cv2.imread(directory + img_name)

    # filter out night images
    if(isDay(img, [lower_hsv1, upper_hsv1, lower_hsv2, upper_hsv2],
        blob_size_lower, blob_size_upper)):
        # add to lists
        images_filtered.append(img)
        filtered_names.append(img_name)

# ---------------------------------------------------------------------------------
# Equalize Images
# ---------------------------------------------------------------------------------

# get number of filtered images
num_images = len(images_filtered)

# list to hold equalized images
images_equalized = list()

print("\n\nEqualizing Images")

# iterate through images
for count, img in enumerate(images_filtered):
    # update progress bar
    progress(count + 1, num_images, status=filtered_names[count])

    # get height and width
    h, w = img.shape[:2]

    # crop image
    img = img[img_border_upper:(h - img_border_lower), :, :]

    # equalize image according to specified parameters
    img_eq = equalize_hist(img, clip_limit, tile_size)

    # add to list
    images_equalized.append(img_eq)

    # if debugging write to directory
    if(debug):
        cv2.imwrite((paths_dict["equalized"] + filtered_names[count]), img_eq)

# equalize and crop template
template = cv2.imread(template_path)
h_temp, w_temp = template.shape[:2]
template = template[img_border_upper:(h - img_border_lower), :, :]
template_eq = equalize_hist(template, clip_limit, tile_size)

# if debugging write to directory
if(debug):
    cv2.imwrite((paths_dict["equalized-template"]+ os.path.split(template_path)[1]), template_eq)

# ---------------------------------------------------------------------------------
# Register Images to Template
# ---------------------------------------------------------------------------------

# list to hold registered images
images_registered = list()

print("\n\nRegistering Images")

# iterate through equalized images
for count, img in enumerate(images_equalized):
    # update progress bar
    progress(count + 1, num_images, status=filtered_names[count])

    # align images
    imgReg, h, imgMatch = alignImages(img, template_eq)

    # add to list
    images_registered.append(imgReg)

    # if debugging write to directory
    if(debug):
        cv2.imwrite((paths_dict["registered"] + filtered_names[count]), imgReg)
        cv2.imwrite((paths_dict["matches"] + filtered_names[count]), imgMatch)

# ---------------------------------------------------------------------------------
# Overlay ROI from template onto images
# ---------------------------------------------------------------------------------

# only run if in debugging mode
if(debug):
    print("\n\nOverlaying ROI")

    # iterate through images
    for count, img in enumerate(images_registered):
        # create copy
        img_write = img.copy()

        # update progress bar
        progress(count + 1, num_images, status=filtered_names[count])

        # iterate through stakes
        for j, stake in enumerate(roi_coordinates):
            # overaly template intersection point
            cv2.circle(img_write, (int(template_intersections[j][0]),
                int(template_intersections[j][1] - img_border_upper)), 5, (0,255,0), 3)

            # iterate through roi in each stake
            for i, rectangle in enumerate(stake):
                # stake itself
                if(i == 0):
                    cv2.rectangle(img_write, (rectangle[0][0], rectangle[0][1]-img_border_upper),
                        (rectangle[1][0], rectangle[1][1]-img_border_upper), (0, 0, 255), 3)
                # blobs
                else:
                    cv2.rectangle(img_write, (rectangle[0][0], rectangle[0][1]-img_border_upper),
                        (rectangle[1][0], rectangle[1][1]-img_border_upper), (0, 255, 0), 3)

        cv2.imwrite(paths_dict["template-overlay"] + filtered_names[count], img_write)

# ---------------------------------------------------------------------------------
# Get Valid Stakes
# ---------------------------------------------------------------------------------

print("\n\nValidating Stakes")

# check stakes in image
stake_validity, blob_coords = getValidStakes(images_registered, roi_coordinates, [lower_hsv1, upper_hsv1, lower_hsv2, upper_hsv2],
    blob_size_lower, blob_size_upper, img_border_upper, debug, filtered_names,
    paths_dict["stake-check"])

# ---------------------------------------------------------------------------------
# Determine Snow Intersection Point
# ---------------------------------------------------------------------------------

print("\n\nDetermining Intersection Points")

# get intersection points
intersection_coords = getIntersections(images_registered, blob_coords, stake_validity, roi_coordinates, 130, filtered_names, debug, paths_dict["intersection"])

# test output
for i, img_name in enumerate(filtered_names):
    img_write2 = images_registered[i]
    print "\n"
    print img_name
    coords_stake = intersection_coords[img_name]
    for j,stake in enumerate(coords_stake):
        if stake_validity[img_name][j]:
            cv2.circle(img_write2, (int(template_intersections[j][0]), int(template_intersections[j][1])-img_border_upper), 5, (255,0,0), 3)
            cv2.circle(img_write2, (int(stake['average'][0]), int(stake['average'][1])), 5, (0,255,0), 2)
            print "stake %s : %s mm" % (j, (((template_intersections[j][1] - img_border_upper)-stake['average'][1])*template_tensor[j]))

    cv2.imwrite(paths_dict["testing"] + img_name, img_write2)

# display run time
print("\n\nRun Time: %.2f s" % (time.time() - start))

sys.exit()



































# process all images in sub-directory
img_dir = params[0] + '/'

# contains output data
coordinate_output = {}

# add directory to output
coordinate_output["directory"] = img_dir

# get images
images = [file_name for file_name in os.listdir(img_dir)]
num_images = len(images)

# output to command line
print("\nGenerating Coordinates")

# iterate through images
for count, img_name in enumerate(images):
    # update progress bar
    progress(count + 1, num_images, status=img_name)

    # read in image
    img = cv2.imread(img_dir + img_name)

    # contains data that will be written to file
    output_string = "["

    # get height and width
    h, w = img.shape[:2]

    # crop image to remove metadata
    img = img[img_border_upper:(h - img_border_lower), :, :]

    # replace original image with cropped image
    output_path = img_dir + img_name
    cv2.imwrite(output_path, img)

    # reduce noise in image by local smoothing
    img = cv2.medianBlur(img, median_kernal_size)

    # identify coloured areas in image based on two hardcoded ranges of HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    mask = cv2.bitwise_or(mask1, mask2)

    # erosion followed by dilation to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dilate_kernel)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # dilate contours
    mask_open = cv2.dilate(mask_open, dilate_kernel, iterations=5)

    # find final coloured polygon regions
    mask_filtered = np.zeros((h, w), dtype=np.uint8)
    contours = cv2.findContours(mask_open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    number_contours = len(contours)

    for i, cnt in enumerate(contours):

        # filter by area
        contour_area = cv2.contourArea(cnt)
        if min_contour_area <= contour_area <= max_contour_area:
            rect = cv2.minAreaRect(cnt)
            _, size, angle = rect
            width, height = size

            # filter by aspect ratio
            if (angle >= angle_thresh and width <= height and bar_width_low <= width <= bar_width_high) or \
                    (angle <= angle_thresh and width >= height and bar_width_low <= height <= bar_width_high):
                # draw contour
                cv2.drawContours(mask_filtered, [cnt], 0, 255, -1)

                # get rectangle points
                coords = cv2.boxPoints(rect)
                coords = coords.tolist()
                coords2 = coords.sort()

               	# get top left coordinate
               	topLeft = (10000,100000)
               	for point in coords:
               		if(point[0] < (topLeft[0]+10) and point[1] < (topLeft[1] + 10)):
               			topLeft = point

                # get top left coordinate
                #topLeft = min(coords)

                # increase the size of the bounding boxes
                horizontal_increase = 5
                vertical_increase = 5

                # update output string
                output_string += ("%s %s %s %s" % (topLeft[0] - horizontal_increase, (topLeft[1]) - vertical_increase, \
                                                   width + (horizontal_increase*2), height + vertical_increase))
                if i != number_contours - 1:
                    output_string += ";"

    # update output string
    output_string += "]"

    # add data to output
    coordinate_output[img_name] = {
        'coordinates': output_string
    }

file = open('./coordinates.json', 'w')
json.dump(coordinate_output, file, sort_keys=True, indent=4, separators=(',', ': '))
