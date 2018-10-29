# add to python path
import sys

sys.path.append('./includes')
sys.path.append('./includes/GUI')

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
from main import GUI
from get_intersection import getIntersections
from calculate_depth import getDepths
from equalize import equalize_hist_colour
from overlay_roi import overlay
import Tkinter as tk
import datetime
import time
from colour_balance import balanceColour
from update_dataset import createDataset
from update_dataset import createDatasetTensor

# create GUI window
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
roi_coordinates = params[7]
template_path = params[8]
clip_limit = params[9]
tile_size = tuple(params[10])
template_intersections = params[12]
template_tensor = params[13]
template_blob_sizes = params[14]
template_data_set = params[15]
template_name = params[16]
tensor_data_set = params[17]
blob_distances_template = params[18]
STD_DEV_REG, STD_DEV_TENSOR, ROTATION, TRANSLATION, SCALE = params[19]

# determine if the dataset for the template is established
# must have registered at least 50 images to the template
dataset_enabled = True if template_data_set[0][2] != 0 else False

# output to user the status of the dataset
print "\nStatus:"
print "Registration Dataset is %s" % ("ENABLED" if dataset_enabled else "DISABLED")
if(not dataset_enabled):
    print "Number of images required: %d\n" % (50-len(template_data_set[1]))

# determine if tensor dataset for the template is established
# must have calculated at least 50 tensors
dataset_tensor_enabled = list()
for x in tensor_data_set:
    if(x[0][2] != 0): dataset_tensor_enabled.append(True)
    else: dataset_tensor_enabled.append(False)

# output to user the status of the tensor dataset
for k, stake in enumerate(tensor_data_set):
    print "Stake %d Dataset is %s" % (k, "ENABLED" if dataset_tensor_enabled[k] else "DISABLED")

    if(not dataset_tensor_enabled[k]):
        print "Number of images required: %d" % (50-len(stake[1]))

# flag to run program in debug mode
debug = params[11]

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
paths_dict["snow-depth"] = path + "/snow-depth/"

# results directory
os.mkdir(paths_dict["snow-depth"])

# debug directories
if(debug):
    os.mkdir(paths_dict["equalized"])
    os.mkdir(paths_dict["equalized-template"])
    os.mkdir(paths_dict["registered"])
    os.mkdir(paths_dict["matches"])
    os.mkdir(paths_dict["template-overlay"])
    os.mkdir(paths_dict["stake-check"])
    os.mkdir(paths_dict["intersection"])

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
    if(isDay(img.copy(), [lower_hsv1, upper_hsv1, lower_hsv2, upper_hsv2],
        template_blob_sizes)):
        # add to lists
        images_filtered.append(balanceColour(img, 1))
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
    img_eq_gray = equalize_hist(img, clip_limit, tile_size)
    img_eq = equalize_hist_colour(img, clip_limit, tile_size)

    # replace image in list with equalized image
    images_filtered[count] = img_eq

    # add to list
    images_equalized.append(img_eq_gray)

    # if debugging write to directory
    if(debug):
        cv2.imwrite((paths_dict["equalized"] + filtered_names[count]), img_eq)

# equalize and crop template
template = cv2.imread(template_path)
h_temp = template.shape[:2][0]
template = template[img_border_upper:(h_temp - img_border_lower), :, :]
template_eq = equalize_hist(template, clip_limit, tile_size)

# if debugging write to directory
if(debug):
    cv2.imwrite((paths_dict["equalized-template"]+ os.path.split(template_path)[1]), template_eq)

# ---------------------------------------------------------------------------------
# Register Images to Template
# ---------------------------------------------------------------------------------

print("\n\nRegistering Images")

# get registered images
images_registered, template_data_set, filtered_names_reg = alignImages(images_equalized, template_eq, filtered_names,
    images_filtered, paths_dict["registered"], paths_dict["matches"], debug, template_data_set, dataset_enabled,
    ROTATION, TRANSLATION, SCALE, STD_DEV_REG)

# update registration dataset
createDataset(template_name, template_data_set, dataset_enabled)

# ---------------------------------------------------------------------------------
# Overlay ROI from template onto images
# ---------------------------------------------------------------------------------

# only run if in debugging mode
if(debug):
    print("\n\nOverlaying ROI")

    overlay(images_registered, template_intersections, roi_coordinates, img_border_upper,
        filtered_names_reg, paths_dict["template-overlay"])

# ---------------------------------------------------------------------------------
# Get Valid Stakes
# ---------------------------------------------------------------------------------

print("\n\nValidating Stakes")

# check stakes in image
stake_validity, blob_coords, tensor_data_set = getValidStakes(images_registered, roi_coordinates, [lower_hsv1,
    upper_hsv1, lower_hsv2, upper_hsv2], template_blob_sizes, img_border_upper, debug, filtered_names_reg, paths_dict["stake-check"],
    tensor_data_set, dataset_tensor_enabled, STD_DEV_TENSOR)

# update tensor dataset
createDatasetTensor(template_name, tensor_data_set, dataset_tensor_enabled)

# ---------------------------------------------------------------------------------
# Determine Snow Intersection Point
# ---------------------------------------------------------------------------------

print("\n\nDetermining Intersection Points")

# get intersection points
intersection_coords, intersection_dist = getIntersections(images_registered, blob_coords, stake_validity, roi_coordinates,
    filtered_names_reg, debug, paths_dict["intersection"])

# ---------------------------------------------------------------------------------
# Calculate Change in Snow Depth
# ---------------------------------------------------------------------------------

print("\n\nCalculating Change in Snow Depth")

# get snow depths
depths = getDepths(images_registered, filtered_names_reg, intersection_coords, stake_validity,
    template_intersections, img_border_upper, template_tensor, intersection_dist,
    blob_distances_template, debug, paths_dict["snow-depth"])

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
