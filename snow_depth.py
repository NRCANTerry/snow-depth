# add to python path
import sys

sys.path.append('./Include')
sys.path.append('./Include/GUI')
sys.path.append('./Include/DL')

# import necessary modules
import cv2
import os
import numpy as np
import tkinter as tk
from main import GUI
from calculate_depth import getDepths
from overlay_roi import overlay
from datetime import datetime
from time import time
from update_dataset import createDataset
from update_dataset import createDatasetTensor
from tqdm import tqdm
from pathlib import Path
from generate_report import generate

# class to allow dot functionality with dict
class Map(dict):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

if __name__ == '__main__':
    # create GUI window
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()

    # get parameters
    params = gui.getValues()

    # start timer
    start = time()

    # ---------------------------------------------------------------------------------
    # Get parameters from GUI
    # ---------------------------------------------------------------------------------

    # dictionary to hold program output data
    summary = Map()

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
    date_range = params[20]
    reg_params = params[21]
    int_params = params[22]
    misc_params = params[23]

    # update summary
    summary.start = datetime.now()
    summary.HSVRange = [lower_hsv1, upper_hsv1, lower_hsv2, upper_hsv2]
    summary["Borders (Upper, Lower)"] = [img_border_upper, img_border_lower]
    summary.Debug = params[11]
    summary["Image Directory"] = os.path.basename(os.path.normpath(directory))
    summary.Template = template_name

    # determine if the dataset for the template is established
    # must have registered at least 50 images to the template
    dataset_enabled = True if template_data_set[0][2] != 0 else False

    # output to user the status of the dataset
    print("\nStatus:")
    print("Registration Dataset is %s" % ("ENABLED" if dataset_enabled else "DISABLED"))
    if(not dataset_enabled):
        print("Number of images required: %d\n" % (50-len(template_data_set[1])))

    # determine if tensor dataset for the template is established
    # must have calculated at least 50 tensors
    dataset_tensor_enabled = list()
    for x in tensor_data_set:
        if(x[0][2] != 0): dataset_tensor_enabled.append(True)
        else: dataset_tensor_enabled.append(False)

    # output to user the status of the tensor dataset
    for k, stake in enumerate(tensor_data_set):
        print("Stake %d Dataset is %s" % (k, "ENABLED" if dataset_tensor_enabled[k] else "DISABLED"))

        if(not dataset_tensor_enabled[k]):
            print("Number of images required: %d" % (50-len(stake[1])))

    # flag to run program in debug mode
    debug = params[11]

    # get path to training directories and model file
    filename = os.path.splitext(os.path.split(template_path)[1])[0]
    training_path = str(Path(template_path).parents[1]) + "\\training\\" + filename + "\\"
    model_path = str(Path(template_path).parents[1]) + "\\models\\" + filename + ".model"

    # output status of deep learning model
    if(os.path.isfile(model_path) and misc_params[0]):
        print("Deep Learning Blob Model is ENABLED")
    elif not misc_params[0]:
        print("Deep Learning Blob Model is DISABLED")
    else:
        print("Deep Learning Blob Model is DISABLED")
        validTrainingDir = [int(os.path.splitext(x)[0]) for x in os.listdir(training_path + "blob\\")]
        currentIndex = max(validTrainingDir) + 1 if len(validTrainingDir) > 0 else 0
        print("Number of training samples required: %d" % (1000-currentIndex))

    # update summary
    summary["Registration Dataset Enabled"] = dataset_enabled
    summary["Tensor Datasets Enabled"] = dataset_tensor_enabled
    summary["Training Path"] = os.path.basename(os.path.normpath(training_path))
    summary["Model Path"] = os.path.basename(os.path.normpath(model_path))

    # ---------------------------------------------------------------------------------
    # Create Directories
    # ---------------------------------------------------------------------------------

    # dictionary of paths
    paths_dict = dict()

    # create directories
    if(not os.path.isdir("Results")):
        os.mkdir("./Results")

    # add folder for run
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "-").replace(":", "-")
    path = "./Results/" + date
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
    # Setup parallel pool
    # ---------------------------------------------------------------------------------

    # number of images
    num_imgs = len([file_name for file_name in os.listdir(directory)])
    summary["Number of Images"] = num_imgs

    # only use parallel pool if there are more than 5 images
    if(num_imgs > 5):
        from multiprocessing import Pool
        from multiprocessing import cpu_count
        from multiprocessing import Queue

        print("\nCreating Parallel Pool...")

        # create pool with 75% as many processes as there are cores
        num_cores = float(cpu_count()) * 0.75
        pool = Pool(int(num_cores))
        print("Parallel Pool Created (%d Workers)" % int(num_cores))

        # update summary
        summary["Parallel Pool"] = True
        summary["Number of Cores"] = num_cores

    # update summary
    else: summary["Parallel Pool"] = False

    # ---------------------------------------------------------------------------------
    # Filter Out Night Images
    # ---------------------------------------------------------------------------------

    print("\nSelecting Valid Images")
    intervalTime = time()
    numInitial = len([file_name for file_name in os.listdir(directory)]) # get intial image numbers

    # get filtered images and image names
    if(num_imgs > 50 and not date_range[3]):
        from filter_night import filterNightParallel
        images_filtered, filtered_names = filterNightParallel(pool, directory,
            img_border_upper, img_border_lower)
    else:
        from filter_night import filterNight
        images_filtered, filtered_names = filterNight(directory, img_border_upper,
            img_border_lower, date_range)

    # output results of filtering
    if(date_range[3]): print("Number of Valid Images: %d" % len(images_filtered))

    # update summary
    filteringTime = time() - intervalTime
    summary["Filtering Time"] = "%0.2fs" % filteringTime
    summary["Per Image Filtering Time"] = "%0.2f" % (filteringTime / float(num_imgs))
    summary["Number of Input Images"] = numInitial
    summary["Number of Night Images"] = numInitial - len(images_filtered)

    # update number of images
    num_imgs = len(images_filtered)

    # ---------------------------------------------------------------------------------
    # Equalize Images
    # ---------------------------------------------------------------------------------

    print("\n\nEqualizing Images")
    intervalTime = time()

    if(num_imgs > 50):
        from equalize import equalizeImageSetParallel
        images_equalized, images_filtered, template_eq, template = equalizeImageSetParallel(pool,
            images_filtered, filtered_names, template_path, img_border_upper, img_border_lower,
            clip_limit, tile_size, debug, paths_dict["equalized"], paths_dict["equalized-template"])
    else:
        from equalize import equalizeImageSet
        images_equalized, images_filtered, template_eq, template = equalizeImageSet(images_filtered,
            filtered_names, template_path, img_border_upper, img_border_lower, clip_limit, tile_size, debug,
            paths_dict["equalized"], paths_dict["equalized-template"])

    # update summary
    equalizationTime = time() - intervalTime
    summary["Equalization Time"] = "%0.2fs" % equalizationTime
    eqIntervalTime = equalizationTime / float(num_imgs) if num_imgs > 0 else 0
    summary["Per Image Equalization Time"] = "%0.2fs" % (eqIntervalTime)
    summary["Clip Limit"] = clip_limit
    summary["Tile Size"] = tile_size

    # ---------------------------------------------------------------------------------
    # Register Images to Template
    # ---------------------------------------------------------------------------------

    # update number of images
    num_imgs = len(images_equalized)

    print("\n\nRegistering Images")
    intervalTime = time()

    if(num_imgs > 5):
        from register import alignImagesParallel
        images_registered, template_data_set, filtered_names_reg, stats = alignImagesParallel(pool, images_equalized,
            template_eq, template, filtered_names, images_filtered, paths_dict["registered"], paths_dict["matches"], debug,
            template_data_set, dataset_enabled, ROTATION, TRANSLATION, SCALE, STD_DEV_REG, reg_params)
    else:
        from register import alignImages
        images_registered, template_data_set, filtered_names_reg, stats = alignImages(images_equalized, template_eq, template,
            filtered_names, images_filtered, paths_dict["registered"], paths_dict["matches"], debug, template_data_set,
            dataset_enabled, ROTATION, TRANSLATION, SCALE, STD_DEV_REG, reg_params)

    # update summary
    registrationTime = time() - intervalTime
    summary["Registration Time"] = "%0.2fs" % registrationTime
    regIntervalTime = registrationTime / float(num_imgs) if num_imgs > 0 else 0
    summary["Per Image Registration Time"] = "%0.2fs" % (regIntervalTime)
    summary.regRestrictions = [ROTATION, TRANSLATION, SCALE]
    summary["ORB Registration"] = "%d/%d" % (stats[0], num_imgs)
    summary["ECC Registration"] = "%d/%d" % (stats[1], num_imgs)
    summary["Failed Registration"] = "%d/%d" % (num_imgs - len(images_registered), num_imgs)
    summary["Average Mean Squared Error"] = "%0.2f" % stats[2]

    # update registration dataset
    createDataset(template_name, template_data_set, dataset_enabled)

    # ---------------------------------------------------------------------------------
    # Get Date and Time of Images from EXIF data
    # ---------------------------------------------------------------------------------

    print("\n\nExtracting EXIF Data")

    from PIL import Image

    image_dates = list() # list for image EXIF data
    for img in tqdm(filtered_names_reg):
        pil_im = Image.open(directory+img)
        exif = pil_im._getexif()
        if exif is not None: # if exif data exists
            image_dates.append(datetime.strptime(exif[36867], '%Y:%m:%d %H:%M:%S'))
        else:
            image_dates.append(img)

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
    intervalTime = time()
    #from check_stakes import getValidStakes
    from validate_stakes import getValidStakes

    # check stakes in image
    stake_validity, blob_coords, tensor_data_set, actual_tensors = getValidStakes(images_registered, roi_coordinates, [lower_hsv1,
        upper_hsv1, lower_hsv2, upper_hsv2], template_blob_sizes, img_border_upper, debug, filtered_names_reg, paths_dict["stake-check"],
        tensor_data_set, dataset_tensor_enabled, STD_DEV_TENSOR, training_path, model_path, misc_params[0])

    # update tensor dataset
    createDatasetTensor(template_name, tensor_data_set, dataset_tensor_enabled)

    # update summary
    checkTime = time() - intervalTime
    summary["Check Time"] = "%0.2fs" % checkTime
    checkIntervalTime = checkTime / float(num_imgs) if num_imgs > 0 else 0
    summary["Per Image Check Time"] = "%0.2fs" % (checkIntervalTime)

    # ---------------------------------------------------------------------------------
    # Determine Snow Intersection Point
    # ---------------------------------------------------------------------------------

    print("\n\nDetermining Intersection Points")
    intervalTime = time()

    # get intersection points
    if(num_imgs > 5):
        from intersect import getIntersectionsParallel
        intersection_coords, intersection_dist = getIntersectionsParallel(pool, images_registered, blob_coords, stake_validity,
            roi_coordinates, filtered_names_reg, debug, paths_dict["intersection"], int_params, actual_tensors,
            img_border_upper)
    else:
        from intersect import getIntersections
        intersection_coords, intersection_dist = getIntersections(images_registered, blob_coords, stake_validity, roi_coordinates,
            filtered_names_reg, debug, paths_dict["intersection"], int_params, actual_tensors, img_border_upper)

    # update summary
    intersectionTime = time() - intervalTime
    summary["Intersection Time"] = "%0.2fs" % intersectionTime
    intIntervalTime = intersectionTime / float(num_imgs) if num_imgs > 0 else 0
    summary["Per Image Intersection Time"] = "%0.2fs" % (intIntervalTime)

    # ---------------------------------------------------------------------------------
    # Calculate Change in Snow Depth
    # ---------------------------------------------------------------------------------

    print("\n\nCalculating Change in Snow Depth")
    intervalTime = time()

    # get snow depths
    depths = getDepths(images_registered, filtered_names_reg, intersection_coords, stake_validity,
        template_intersections, img_border_upper, template_tensor, actual_tensors, intersection_dist,
        blob_distances_template, debug, paths_dict["snow-depth"], image_dates)

    # update summary
    calcTime = time() - intervalTime
    summary["Calculation Time"] = "%0.2fs" % calcTime
    calcIntervalTime = calcTime / float(num_imgs) if num_imgs > 0 else 0
    summary["Per Image Calculation Time"] = "%0.2fs" % (calcIntervalTime)

    # display run time
    runtime = time() - start
    perImgTime = runtime / float(num_imgs) if num_imgs > 0 else runtime
    print("\n\nRun Time: %.2f s (%.2f s/img)" % (runtime, perImgTime))

    # ---------------------------------------------------------------------------------
    # Generate Report
    # ---------------------------------------------------------------------------------

    print("\n\nGenerating Report...")
    generate(summary, paths_dict["snow-depth"])
