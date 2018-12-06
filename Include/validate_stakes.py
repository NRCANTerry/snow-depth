# import necessary packages
import cv2
import numpy as np
import json
from order_points import orderPoints
import os
from get_tensor import getTensor
import statistics
import tqdm
import random
from lenet import LeNet
from pathlib import Path
from classify import classify
from keras.models import load_model

# parameters
median_kernel_size = 5
dilate_kernel = (5, 5)

def getActualCoords(dilatedCoords, upper_border):
    '''
    Function to get undilated coordinates from template roi
    '''
    top_left = (dilatedCoords[0][0], dilatedCoords[0][1]-upper_border)
    bottom_right = (dilatedCoords[1][0], dilatedCoords[1][1]-upper_border)
    blob_width = abs(top_left[0] - bottom_right[0]) / 1.6666
    dilate_px = int(float(blob_width) * 0.33)

    return (
        [top_left[0]+dilate_px, top_left[1]+dilate_px],
        [bottom_right[0]-dilate_px, top_left[1]+dilate_px],
        [bottom_right[0]-dilate_px, bottom_right[1]-dilate_px],
        [top_left[0]+dilate_px, bottom_right[1]-dilate_px]
    )

def roiValid(coordinates, blobs):
    '''
    Function to determine if a randomly sampled roi intersects with any blobs
    @param coordinates the coordinates of the randomly sampled roi
    @param blobs list of blob coordinates
    @type coordinates list([x0, y0], [x1, y1])
    @type blobs list(blob1[[x0, y0], [x1, y1]], ...)
    @return whether the randomly sampled roi is valid
    @rtype bool
    '''

    # iterate through blobs
    for blob in blobs:
        # if one rectangle is to the side of the other
        if(coordinates[0][0] > blob[1][0] or blob[0][0] > coordinates[1][0]):
            continue # check next

        # if one rectangle is above the other
        if(coordinates[0][1] > blob[1][1] or blob[0][1] > coordinates[1][1]):
            continue # check next

        # if neither condition is met the rectangles overlap
        return False

    # return True if passes all tests
    return True

def imageValid(img_, coordinates, hsvRanges, blobSizes, upper_border, debug, name,
    debug_directory, dataset, dataset_enabled, NUM_STD_DEV, validPath, invalidPath,
    model, modelInitialized, validIndex, invalidIndex, flattened_list, DLActive):
    '''
    Function to determine which stakes in an image are valid
    '''

    # duplicate image
    img = img_.copy()
    img_low_blob = img_.copy()

    # get image size
    h_img, w_img = img_.shape[:2]

    # determine whether single or double HSV range
    numRanges = len(hsvRanges)

    # create bool list for stakes
    validStakes = list()

    # create list for blob coordinates on stakes
    blobCoordsStake = list()

    # create list for actual blob coordinates
    actualCoordsStake = list()

    # create list for stake tensors
    actualTensorsStake = list()

    # create list for valid blobs
    validBlobsImage = list()

    # reduce noise in image by local smoothing
    img_blur = cv2.medianBlur(img, median_kernel_size)

    # identify coloured regions in image
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, hsvRanges[0], hsvRanges[1])

    # apply second mask if required
    if(numRanges == 4):
        mask_hsv2 = cv2.inRange(hsv, hsvRanges[2], hsvRanges[3])
        mask_hsv = cv2.bitwise_or(mask_hsv, mask_hsv2)

    # erosion followed by dilation to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dilate_kernel)
    mask_open = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)

    # iterate through stakes
    for j, stake in enumerate(coordinates):
        # create bool list for blobs for each stake
        validBlobs = list()

        # get blob size range for stake
        blob_size_range = blobSizes[j]

        # list to store actual coordinates of blobs
        actualCoords = list()

        # iterate through roi in each stake
        for i, rectangle in enumerate(stake):
            # skip stakes
            if i == 0:
                continue

            # blob counter
            num_blobs = 0

            # create a zero image
            mask = np.zeros(mask_open.shape, np.uint8)

            # get points
            top_left = (rectangle[0][0], rectangle[0][1]-upper_border)
            bottom_right = (rectangle[1][0], rectangle[1][1]-upper_border)

            # copy ROI to zero image
            mask[int(top_left[1]):int(bottom_right[1]),int(top_left[0]):int(bottom_right[0])] = \
                mask_open[int(top_left[1]):int(bottom_right[1]),int(top_left[0]):int(bottom_right[0])]

            # find final coloured polygon regions
            contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            contour_index = 0

            # iterate through contours
            for k, cnt in enumerate(contours):
                # filter by area
                if(blob_size_range[0] <= cv2.contourArea(cnt) <= blob_size_range[1]):
                    # increment blob counter
                    num_blobs += 1
                    contour_index = k

            # add to valid blob list if one valid blob found
            if(num_blobs == 1):
                validBlobs.append(True)

                # update lowest/highest blob variables
                rect = cv2.minAreaRect(contours[contour_index])
                coords = cv2.boxPoints(rect)
                box = np.array(coords, dtype = "int")

                # add to list of points for stake
                actualCoords.append(orderPoints(box, False))

                # if in debugging mode draw green (valid) rectangle
                if(debug):
                    cv2.rectangle(img, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
                        (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 255, 0), 3)

                # Generate Training Images for Deep Learning Model
                if(not modelInitialized and DLActive):
                    # write to training folder
                    train_img = img_[int(rectangle[0][1])-upper_border:int(rectangle[1][1])-upper_border,
                        int(rectangle[0][0]):int(rectangle[1][0])]
                    train_name = "%d.JPG" % validIndex
                    cv2.imwrite(validPath + train_name, train_img)
                    validIndex += 1

                    # roi size
                    roi_width = abs(rectangle[0][0] - rectangle[1][0])
                    roi_height = abs(rectangle[0][1] - rectangle[1][1])

                    # sample an invalid blob (random roi from image) and write to invalid training folder
                    # generate random x and y coordinates
                    x_rand = random.randint(0, int(w_img - roi_width))
                    y_rand = random.randint(0, int(h_img - roi_height))

                    # iterate until valid roi
                    while(not(roiValid([[x_rand, y_rand], [x_rand+roi_width, y_rand+roi_height]],
                        flattened_list))):
                        x_rand = random.randint(0, int(w_img - roi_width))
                        y_rand = random.randint(0, int(h_img - roi_height))

                    # write random roi to training folder
                    train_img_invalid = img_[int(y_rand):int(y_rand+roi_height),
                        int(x_rand):int(x_rand+roi_width)]
                    train_name_invalid = "%d.JPG" % invalidIndex
                    cv2.imwrite(invalidPath + train_name_invalid, train_img_invalid)
                    invalidIndex += 1

                    # add yellow rectangle for roi taken
                    cv2.rectangle(img, (int(x_rand), int(y_rand)),
                        (int(x_rand+roi_width), int(y_rand+roi_height)), (0, 255, 255), 3)

            # else add invalid blob
            else:
                # verify using deep learning network
                DLValid = False
                if(modelInitialized and DLActive):
                    subset = img_[int(rectangle[0][1])-upper_border:int(rectangle[1][1])-upper_border,
                        int(rectangle[0][0]):int(rectangle[1][0])]
                    output = classify(subset, model)
                    if(output[0] and output[1] > 0.9):
                        DLValid = True

                # if identified as blob by deep learning network
                if(DLValid):
                    validBlobs.append(True)

                    # use template blob coordinates
                    blobWidth = abs(top_left[0] - bottom_right[0]) / 1.6666
                    dilate_px = int(float(blobWidth) * 0.33)
                    actualCoords.append((
                        [top_left[0]+dilate_px, top_left[1]+dilate_px],
                        [bottom_right[0]-dilate_px, top_left[1]+dilate_px],
                        [bottom_right[0]-dilate_px, bottom_right[1]-dilate_px],
                        [top_left[0]+dilate_px, bottom_right[1]-dilate_px]
                    ))

                # else it is not a blob
                else:
                    validBlobs.append(False)
                    actualCoords.append(False)

                # if in debugging mode draw red (invalid) rectangle
                if(debug):
                    if(modelInitialized and DLValid):
                        cv2.rectangle(img, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
                            (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 165, 255), 3)
                    else:
                        cv2.rectangle(img, (int(rectangle[0][0]), int(rectangle[0][1])-upper_border),
                            (int(rectangle[1][0]), int(rectangle[1][1])-upper_border), (0, 0, 255), 3)

        # determine number of valid blobs on stake
        validBlobsOnStake = validBlobs.count(True)

        # determine tensor for stake
        tensors_low = list()
        tensors_high = list()

        # mean tensor
        mean_tensor = 0

        # get top and bottom tensor
        # if more than 2 blobs on bottom part of stake
        if(validBlobs[0:3].count(True) >= 2):
            for x in range(0, 4):
                for y in range(x+1, 4):
                    # if valid blob, calculate tensor
                    if(validBlobs[x] and validBlobs[y]):
                        tensors_low.append(getTensor(actualCoords[x][1], actualCoords[y][1],
                            ((y-x) * (80+56))))

            # get median
            if(len(tensors_low) > 0):
                median_tensor_low = statistics.median(tensors_low)
                mean_tensor += median_tensor_low

        if(len(validBlobs) >= 6 and validBlobs[4:7].count(True) >= 2):
            # determine number of blobs on stake
            num_blobs_on_stake = len(validBlobs)
            if(len(validBlobs) > 8): num_blobs_on_stake = 8

            for x in range(4, num_blobs_on_stake):
                for y in range(x+1, num_blobs_on_stake):
                    # if valid blob, calculate tensor
                    if(validBlobs[x] and validBlobs[y]):
                        tensors_high.append(getTensor(actualCoords[x][1], actualCoords[y][1],
                            ((y-x) * (80+56))))

            # get median
            if(len(tensors_high) > 0):
                median_tensor_high = statistics.median(tensors_high)

                if(mean_tensor != 0):
                    mean_tensor = (mean_tensor + median_tensor_high) / 2.0
                else:
                    mean_tensor += median_tensor_high

        # flag to indicate whether stake is valid based on tensor comparison
        tensorValid = True

        # check tensor against dataset
        if(not dataset_enabled[j]):
            # add tensor to list
            if(validBlobsOnStake > 4):
                actualTensorsStake.append(mean_tensor)
            else:
                actualTensorsStake.append(True) # use template tensor

        # if dataset is enabled, compare tensor to mean
        else:
            # get mean and standard deviation from dataset
            mean = dataset[j][0][0]
            std_dev = dataset[j][0][1]

            # if tensor measurement is within defined range or within 5%
            if(mean_tensor != 0 and (((mean-(std_dev*NUM_STD_DEV)) <= mean_tensor and
                mean_tensor <= (mean+(std_dev*NUM_STD_DEV))) or abs(mean-mean_tensor)/mean_tensor < 0.05)):
                # add tensor to list
                if(validBlobsOnStake > 4):
                    actualTensorsStake.append(mean_tensor)
                else:
                    actualTensorsStake.append(True) # use template tensor

            # update flag to indicate bad tensor match
            else:
                tensorValid = False

                # add False to list
                actualTensorsStake.append(False)

        # determine if stake is valid
        # need at minimum 2 blobs for stake to be valid
        validStake = (validBlobsOnStake >= 2 and tensorValid)

        # if in debugging mode draw appropriate rectangle around stake
        if(validStake and debug):
            # green rectangle
            cv2.rectangle(img, (int(stake[0][0][0]), int(stake[0][0][1])-upper_border),
                (int(stake[0][1][0]), int(stake[0][1][1])-upper_border), (0, 255, 0), 3)
        elif(debug):
            # red rectangle
            cv2.rectangle(img, (int(stake[0][0][0]), int(stake[0][0][1])-upper_border),
                (int(stake[0][1][0]), int(stake[0][1][1])-upper_border), (0, 0, 255), 3)

        # if more than 2 valid blobs list stake as valid
        validStakes.append(validStake)
        validBlobsImage.append(validBlobs)

        # add lowest blob to list
        if validStake:
            # order coordinates and append to list
            validCoordinates = [t for t in actualCoords if t != False]

            # if there are more than 2 valid blobs on the stake
            if len(validCoordinates) > 2:
                ordered_coordinates_low = validCoordinates[0]
                ordered_coordinates_high = validCoordinates[len(validCoordinates)-1]
                blobCoordsStake.append(list(ordered_coordinates_low + ordered_coordinates_high))

            # if there are only 2 valid blobs on the stake
            else:
                # take lowest and highest possible blobs on stake to create intersection lines
                ordered_coordinates_low = getActualCoords(coordinates[j][1], upper_border)
                ordered_coordinates_high = getActualCoords(coordinates[j][len(coordinates[j])-1], upper_border)

                # replace valid blob coordinates with template coordinates in actualCoords
                for blob_index in range(0, len(actualCoords)):
                    if actualCoords[blob_index] != False: # valid blob but unreliable coordinates
                        actualCoords[blob_index] = getActualCoords(coordinates[j][blob_index], upper_border)

                # add to upper and lower list
                blobCoordsStake.append(list(ordered_coordinates_low + ordered_coordinates_high))

            # add actual coordinates to list for stake
            actualCoordsStake.append(actualCoords)

            # write labelled image if in debugging mode
            if(debug):
                # draw rectangles
                cv2.rectangle(img_low_blob, tuple(map(int, ordered_coordinates_low[0])),
                    tuple(map(int, ordered_coordinates_low[2])), (0,255,0), 3)
                cv2.rectangle(img_low_blob, tuple(map(int, ordered_coordinates_high[0])),
                    tuple(map(int, ordered_coordinates_high[2])), (0,255,0), 3)
        else:
            # if stake is invalid add zero box
            blobCoordsStake.append([0,0,0,0,0,0,0,0])
            actualCoordsStake.append(False)

    # if in debugging mode
    if(debug):
        # write images to debug directory
        filename, file_extension = os.path.splitext(name)
        cv2.imwrite(debug_directory + name, img)
        cv2.imwrite(debug_directory + filename + '-boxes' + file_extension, img_low_blob)

    # create temporary dictionary
    stake_dict = dict()
    stake_dict_coords_low = dict()
    stake_dict_coords_high = dict()
    stake_dict_tensor = dict()

    # add data to output
    for x in range(0, len(coordinates)):
        stake_dict['stake' + str(x)] = validStakes[x]
        stake_dict_coords_low['stake' + str(x)] = blobCoordsStake[x][0:4]
        stake_dict_coords_high['stake' + str(x)] = blobCoordsStake[x][4:8]
        stake_dict_tensor['stake' + str(x)] = actualTensorsStake[x]

    return (validStakes, actualCoordsStake, actualTensorsStake, stake_dict,
        stake_dict_coords_low, stake_dict_coords_high, validIndex, invalidIndex,
        name, validBlobsImage, stake_dict_tensor)

def getModelData(validPath, invalidPath, coordinates):
    '''
    Function to get image indices and flattened list of coordinates
    '''

    # get indexes from image names
    validFiles = [int(os.path.splitext(x)[0]) for x in os.listdir(validPath)]
    invalidFiles = [int(os.path.splitext(x)[0]) for x in os.listdir(invalidPath)]
    validIndex = max(validFiles) + 1 if len(validFiles) > 0 else 0
    invalidIndex = max(invalidFiles) + 1 if len(invalidFiles) > 0 else 0

    # flatten roi coordinates (get list of only blob roi)
    # this is used to check whether randomly sample roi for deep learning
    # algorithm contain any blob
    flattened_list = list()
    for stake in coordinates:
        for a, roi in enumerate(stake):
            if a == 0: continue
            flattened_list.append(roi)

    # return indices and flattened list
    return validIndex, invalidIndex, flattened_list

def updateDatset(dataset, tensor_vals, dataset_enabled):
    '''
    Update dataset given the tensors for a set of images
    '''

    # run for all stakes
    for i, x in enumerate(dataset):
        # if dataset enabled
        if dataset_enabled[i]:
            # get mean and standard deviation from dataset
            mean = dataset[i][0][0]
            std_dev = dataset[i][0][1]

            # iterate through tensor values
            for y in tensor_vals.values():
                if y[i] != True and y[i] != False:
                    mean_tensor = y[i]
                    num_vals_dataset = dataset[i][0][2]
                    new_vals_dataset = num_vals_dataset + 1
                    new_mean = ((mean * num_vals_dataset) + mean_tensor) / new_vals_dataset
                    new_std_dev = np.sqrt(pow(std_dev, 2) + ((((mean_tensor - mean) * (mean_tensor - new_mean)) - \
                                pow(std_dev, 2)) / new_vals_dataset))
                    dataset[i] = np.array([[new_mean, new_std_dev, new_vals_dataset], []])

        # else add values to dataset
        else:
            for y in tensor_vals.values():
                if y[i] != True and y != False:
                    dataset[i][1].append(y[i])

    # return updated dataset
    return dataset

# function to determine which stakes are valid
# verify that blobs are still within reference windows
# need at least two blobs to have a valid stake
# returns a dictionary indicating which stakes in each image are valid
def getValidStakes(imgs, coordinates, hsvRanges, blobSizes, upper_border, debug,
    img_names, debug_directory, dataset, dataset_enabled, NUM_STD_DEV,
    training_path, model_path, DLActive, imageSummary):

    # determine paths for training images
    validPath = training_path + "blob\\"
    invalidPath = training_path + "non-blob\\"

    # flag whethermodel if initialized
    modelInitialized = False
    if os.path.isfile(model_path):
        modelInitialized = True

    if not modelInitialized:
        # get data about model training
        validIndex, invalidIndex, flattened_list = getModelData(validPath, invalidPath,
            coordinates)
        model = None
    else:
        # load model
        model = load_model(model_path)
        validIndex = None
        invalidIndex = None
        flattened_list = None

    # contains output data
    stake_output = {}

    # create bool dictionary for images
    validImages = dict()

    # dictionary for blob coordinates
    blobCoords = dict()

    # dictionary for stkae tensor
    actualTensors = dict()

    # image iterator
    iterator = 0

    # iterate through images
    for img_ in tqdm.tqdm(imgs):
        # get valid stakes from image
        validStakes, actualCoordsStake, actualTensorsStake, stake_dict, stake_dict_coords_low, stake_dict_coords_high, \
            validIndex, invalidIndex, name, validImgBlobs, stake_dict_tensor = imageValid(img_,
            coordinates, hsvRanges, blobSizes, upper_border, debug, img_names[iterator],
            debug_directory, dataset, dataset_enabled, NUM_STD_DEV, validPath,
            invalidPath, model, modelInitialized, validIndex, invalidIndex,
            flattened_list, DLActive)

        # add to output dict if debugging
        if debug:
            stake_output[img_names[iterator]] = {
                "validity": stake_dict,
                "lower blob": stake_dict_coords_low,
                "upper blob": stake_dict_coords_high,
                "tensors": stake_dict_tensor
            }

        # add to return dictionaries
        validImages[img_names[iterator]] = validStakes
        blobCoords[img_names[iterator]] = actualCoordsStake
        actualTensors[img_names[iterator]] = actualTensorsStake

        # update image summary
        imageSummary[name][""] = ""
        imageSummary[name]["Stake (Blob Validity)"] = "Validity      Blob Validity      Tensor (mm/px)"
        for e, stake in enumerate(validImgBlobs):
            num_valid = stake.count(True)
            imageSummary[name]["   %d" % (e+1)] = "%s                %d/%d                       %0.2f        " \
                % (validStakes[e], num_valid, len(stake), actualTensorsStake[e])

        # increment iterator
        iterator += 1

    # update dataset
    dataset = updateDatset(dataset, actualTensors, dataset_enabled)

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory + 'stakes.json', 'w')
        json.dump(stake_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # determine whether model should be initialized
    if not modelInitialized and validIndex > 1000 and invalidIndex > 1000 and DLActive:
        # create neural network
        print("\nInitializing Deep Learning Model")
        LeNet(model_path, validPath, invalidPath)

        # delete training images
        import shutil
        shutil.rmtree(str(Path(validPath).parents[0]))

    # return list of valid stakes
    return validImages, blobCoords, dataset, actualTensors, imageSummary


def unpackArgs(args):
    '''
    Function to unpack arguments explicitly
    @param args function arguments
    @type args arguments
    @return output of imageValid function
    @rtype list
    '''
    return imageValid(*args)

# parallel function to determine which stakes are valid
# verify that blobs are still within reference windows
# need at least two blobs to have a valid stake
# returns a dictionary indicating which stakes in each image are valid
def getValidStakesParallel(pool, imgs, coordinates, hsvRanges, blobSizes, upper_border,
    debug, img_names, debug_directory, dataset, dataset_enabled, NUM_STD_DEV,
    training_path, model_path, DLActive, imageSummary):

    # determine paths for training images
    validPath = training_path + "blob\\"
    invalidPath = training_path + "non-blob\\"

    # flag whethermodel if initialized
    modelInitialized = False
    if os.path.isfile(model_path):
        modelInitialized = True

    if not modelInitialized:
        # get data about model training
        validIndex, invalidIndex, flattened_list = getModelData(validPath, invalidPath,
            coordinates)
        model = None
    else:
        # load model
        model = load_model(model_path)
        validIndex = None
        invalidIndex = None
        flattened_list = None

    # contains output data
    stake_output = {}

    # create bool dictionary for images
    validImages = dict()

    # dictionary for blob coordinates
    blobCoords = dict()

    # dictionary for stkae tensor
    actualTensors = dict()

    # get number of blobs in image
    num_blobs = len(flattened_list)

    # create task list for pool
    tasks = list()
    for i, img_ in enumerate(imgs):
        # determine start valid and invalid index
        # assume all possible blobs on stake are valid
        startIndexValid = validIndex + (i * num_blobs)
        startIndexInvalid = invalidIndex + (i * num_blobs)

        tasks.append((img_, coordinates, hsvRanges, blobSizes, upper_border, debug,
            img_names[i], debug_directory, dataset, dataset_enabled, NUM_STD_DEV,
            validPath, invalidPath, model, modelInitialized, startIndexValid,
            startIndexInvalid, flattened_list, DLActive))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total=len(tasks)):
        # unpack return
        validStakes, actualCoordsStake, actualTensorsStake, stake_dict, stake_dict_coords_low, \
            stake_dict_coords_high, validIndex, invalidIndex, name, validImgBlobs, stake_dict_tensor = i

        # add to output dict if debugging
        if debug:
            stake_output[name] = {
                "validity": stake_dict,
                "lower blob": stake_dict_coords_low,
                "upper blob": stake_dict_coords_high,
                "tensors": stake_dict_tensor
            }

        # add to return dictionaries
        validImages[name] = validStakes
        blobCoords[name] = actualCoordsStake
        actualTensors[name] = actualTensorsStake

        # update image summary
        imageSummary[name][""] = ""
        imageSummary[name]["Stake (Blob Validity)"] = "Validity      Blob Validity      Tensor (mm/px)"
        for e, stake in enumerate(validImgBlobs):
            num_valid = stake.count(True)
            imageSummary[name]["   %d" % (e+1)] = "%s                %d/%d                       %0.2f        " \
                % (validStakes[e], num_valid, len(stake), actualTensorsStake[e])

    # update dataset
    dataset = updateDatset(dataset, actualTensors, dataset_enabled)

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory + 'stakes.json', 'w')
        json.dump(stake_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # determine whether model should be initialized
    if not modelInitialized and validIndex > 1000 and invalidIndex > 1000 and DLActive:
        # create neural network
        print("\nInitializing Deep Learning Model")
        LeNet(model_path, validPath, invalidPath)

        # delete training images
        import shutil
        shutil.rmtree(str(Path(validPath).parents[0]))

    # return list of valid stakes
    return validImages, blobCoords, dataset, actualTensors, imageSummary
