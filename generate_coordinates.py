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
from GUI import GUI

# open gui
gui = GUI()

# get parameters
params = gui.getValues()

if(params == False):
    sys.exit()

# update parameters
img_border_upper = params[5]
img_border_lower = params[6]
lower_hsv1 = params[1]
upper_hsv1 = params[2]
lower_hsv2 = params[3]
upper_hsv2 = params[4]
median_kernal_size = 5
dilate_kernel = (5, 5)
min_contour_area = 1e2
max_contour_area = 1e5
angle_thresh = -45
bar_width_low = 15
bar_width_high = 300

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

                # get top left coordinate
                topLeft = min(coords)

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