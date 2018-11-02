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
from check_stakes import getValidStakes
from main import GUI
from get_intersection import getIntersections
from calculate_depth import getDepths
from overlay_roi import overlay
import Tkinter as tk
import datetime
import time
from colour_balance import balanceColour
from update_dataset import createDataset
from update_dataset import createDatasetTensor

if __name__ == '__main__':
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
    # Setup paralell pool
    # ---------------------------------------------------------------------------------

    # number of images
    num_imgs = len([file_name for file_name in os.listdir(directory)])

    # only use parallel pool if there are more than 5 images
    if(num_imgs > 5):
        from multiprocessing import Pool
        from multiprocessing import Manager
        from multiprocessing import cpu_count
        from multiprocessing import Queue

        print("\nCreating Parallel Pool...")

        # create pool with 75% as many processes as there are cores
        num_cores = float(cpu_count()) * 0.75
        manager = Manager()
        pool = Pool(int(num_cores))

    # ---------------------------------------------------------------------------------
    # Filter Out Night Images
    # ---------------------------------------------------------------------------------

    print("\nFiltering Night Images")

    # get filtered images and image names
    if(num_imgs > 50):
        from filter_night import filterNightParallel
        images_filtered, filtered_names = filterNightParallel(pool, manager, directory,
            img_border_upper, img_border_lower)
    else:
        from filter_night import filterNight
        images_filtered, filtered_names = filterNight(directory, img_border_upper, img_border_lower)

    # ---------------------------------------------------------------------------------
    # Equalize Images
    # ---------------------------------------------------------------------------------

    print("\n\nEqualizing Images")

    if(num_imgs > 50):
        from equalize import equalizeImagesParallel
        images_equalized, images_filtered, template_eq = equalizeImagesParallel(pool, manager,
            images_filtered, filtered_names, template_path, img_border_upper, img_border_lower,
            clip_limit, tile_size, debug, paths_dict["equalized"], paths_dict["equalized-template"])
    else:
        from equalize import equalizeImages
        images_equalized, images_filtered, template_eq = equalizeImages(images_filtered, filtered_names,
            template_path, img_border_upper, img_border_lower, clip_limit, tile_size, debug,
            paths_dict["equalized"], paths_dict["equalized-template"])

    # ---------------------------------------------------------------------------------
    # Register Images to Template
    # ---------------------------------------------------------------------------------

    # update number of images
    num_imgs = len(images_equalized)

    print("\n\nRegistering Images")

    if(num_imgs > 5):
        from register import alignImagesParallel
        images_registered, template_data_set, filtered_names_reg = alignImagesParallel(pool, manager, images_equalized,
            template_eq, filtered_names, images_filtered, paths_dict["registered"], paths_dict["matches"], debug,
            template_data_set, dataset_enabled, ROTATION, TRANSLATION, SCALE, STD_DEV_REG)
    else:
        from register import alignImages
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
