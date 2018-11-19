# import necessary packages
import cv2
import numpy as np
import json
import tqdm
from matplotlib import pyplot as plt
import timeit

# constants
MAX_FEATURES = 262144

def register(img, name, template, template_reduced_noise, img_apply, debug,
    debug_directory_registered, debug_directory_matches, dataset, dataset_enabled,
    NUM_STD_DEV, max_mean_squared_error):
    '''
    Function to align image to template
    @param img image to be aligned
    @param name name of image to be aligned
    @param template the template image
    @param img_apply the image that registration will be applied to
    @param debug flag indicating whether to write debug images
    @param debug_directory_registered path where registered images are saved
    @param debug_directory_matches path where match images are saved
    @param dataset list containing registration mean and std dev
    @param dataset_enabled flag indicating whether the dataset is in use
    @param NUM_STD_DEV number of standard deviations away from the mean the
        MSE of the affine matrix can be
    @param max_mean_squared_error maximum magnitude of registration
    @type img cv2.image
    @type name string
    @type template cv2.image
    @type img_apply cv2.image
    @type debug bool
    @type debug_directory_registered string
    @type debug_directory_matches string
    @type dataset list(list(mean, std_dev, number), list(data))
    @type dataset_enabled bool
    @type NUM_STD_DEV int
    @type max_mean_squared_error float
    @return imgECCAligned the aligned image
    @return name the name of the image
    @return mean_squared_error the MSE for ORB feature alignment
    @return ORB_aligned_flag flag indicating whether image was ORB aligned
    @return affine_matrix the affine transformation matrix for ORB alignment
    @return ECC_aligned_flag flag indicating whether iamge was ECC aligned
    @return warp_matrix the affine transformation matrix for ECC alignment
    @rtype imgECCAligned cv2.image
    @rtype name string
    @rtype mean_squared_error float
    @rtype ORB_aligned_flag bool
    @rtype affine_matrix np.array
    @rtype ECC_aligned_flag bool
    @rtype warp_matrix np.array
    '''

    # flags for whether image was aligned
    ORB_aligned_flag = False
    ECC_aligned_flag = False

    # detect ORB features and compute descriptors
    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    kp1, desc1 = orb.detectAndCompute(img, None)
    kp2, desc2 = orb.detectAndCompute(template_reduced_noise, None)

    # create brute-force matcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # sort matches by score and remove poor matches
    # matches with a score greater than 30 are removed
    matches = [x for x in matches if x.distance <= 30]
    matches = sorted(matches, key=lambda x: x.distance)
    if(len(matches) > 100):
        matches = matches[:100]

    # draw top matches
    imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None, flags=2)

    # extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype = np.float32)
    points2 = np.zeros((len(matches), 2), dtype = np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # determine affine 2D transformation using RANSAC robust method
    affine_matrix = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC)[0]
    height, width = template.shape

    # set registered images to original images
    # will be warped if affine matrix is within spec
    imgReg = img_apply
    imgRegGray = img

    # get mean squared error between affine matrix and zero matrix
    zero_matrix = np.zeros((2,3), dtype=np.float32)
    mean_squared_error = np.sum(np.square(abs(affine_matrix) - zero_matrix))

    # update dataset
    # if dataset isn't enabled, append mean_squared_error to dataset
    if(not dataset_enabled and mean_squared_error <= max_mean_squared_error):
        # apply registration
        imgReg = cv2.warpAffine(img_apply, affine_matrix, (width, height))
        imgRegGray = cv2.cvtColor(imgReg, cv2.COLOR_BGR2GRAY)
        ORB_aligned_flag = True

    # if dataset is enabled, compare matrix to mean
    elif mean_squared_error <= max_mean_squared_error:
        # get mean and standard deviation from dataset
        mean = dataset[0][0]
        std_dev = dataset[0][1]

        # if mean mean_squared_error is within defined range
        if (mean_squared_error <= (mean+(std_dev*NUM_STD_DEV))):
            # apply registration
            imgReg = cv2.warpAffine(img_apply, affine_matrix, (width, height))
            imgRegGray = cv2.cvtColor(imgReg, cv2.COLOR_BGR2GRAY)
            ORB_aligned_flag = True

    # write matches to debug directory
    if(debug):
        cv2.imwrite(debug_directory_matches + name, imgMatches)

    # define ECC motion model
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # specify the number of iterations and threshold
    number_iterations = 750
    termination_thresh = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

    # run ECC algorithm (results are stored in warp matrix)
    warp_matrix = cv2.findTransformECC(template, imgRegGray, warp_matrix, warp_mode, criteria)[1]

    # compare warp matrix to data set
    mean_squared_error_ecc = np.sum(np.square(abs(warp_matrix) - zero_matrix))

    # only check if dataset enabled
    if(dataset_enabled):
        # get mean and standard deviation from dataset
        mean = dataset[0][0]
        std_dev = dataset[0][1]

        # align image if warp is within spec
        if (mean_squared_error_ecc <= (mean+(std_dev*NUM_STD_DEV))):
            # align image
            imgECCAligned = cv2.warpAffine(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            ECC_aligned_flag = True

        # else use ORB registered image
        else:
            imgECCAligned = imgReg

    # align image if dataset not enabled
    elif mean_squared_error_ecc <= max_mean_squared_error:
        # align image
        imgECCAligned = cv2.warpAffine(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        ECC_aligned_flag = True

    # doesn't meet criteria
    else: imgECCAligned = imgReg

    # only if image was aligned (is not the same as input image)
    if(ORB_aligned_flag or ECC_aligned_flag):
        # write registered image to debug directory
        cv2.imwrite(debug_directory_registered + name, imgECCAligned)

        # return registered image and parameters
        return (imgECCAligned, name, mean_squared_error, ORB_aligned_flag,
            affine_matrix.tolist(), ECC_aligned_flag, warp_matrix.tolist())

    # return None if not aligned
    else: return (None, name, mean_squared_error, ORB_aligned_flag,
        affine_matrix.tolist(), ECC_aligned_flag, warp_matrix.tolist())

def getMaxError(MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING):
    '''
    Determine maximum mean squared error for non-initialized dataset
    @param MAX_ROTATION maximum allowed rotation
    @param MAX_TRANSLATION maximum allowed translation
    @param MAX_SCALING maximum allowed scaling
    @type MAX_ROTATION float
    @type MAX_TRANSLATION float
    @type MAX_SCALING float
    @return max_error the maximum mean squared error for a vaild registration
    @rtype float
    '''

    # adjust scaling from % to absolute
    MAX_SCALING /= 100.0

    # create zero matrix
    zero_matrix = np.zeros((2, 3), dtype=np.float32)

    # create affine matrix according to specified rotation, translation and scale
    alpha = MAX_SCALING * np.cos(np.deg2rad(MAX_ROTATION))
    beta = MAX_SCALING * np.sin(np.deg2rad(MAX_ROTATION))
    affine_transform_matrix = np.array([
        [alpha, beta, MAX_TRANSLATION],
        [-beta, alpha, MAX_TRANSLATION]])

    # get maximum mean squared error
    return np.sum(np.square(abs(affine_transform_matrix) - zero_matrix))

def updateDataset(dataset, MSE_vals, dataset_enabled):
    '''
    Update dataset given the mean squared error values for a set of images
    @param dataset the dataset to be updated
    @param MSE_vals the mean squared error values
    @param dataset_enabled whether the dataset is enabled or not
    @type dataset list(list(mean, std_dev, number), list(data))
    @type MSE_vals  list(float)
    @type dataset_enabled bool
    @return dataset
    @rtype dataset list(list(mean, std_dev, number), list(data))
    '''

    # if dataset is enabled
    if dataset_enabled:
        # get mean and standard deviation from dataset
        mean = dataset[0][0]
        std_dev = dataset[0][1]

        # iterate through mean squared error values
        for x in MSE_vals:
            num_vals_dataset = dataset[0][2]
            new_vals_dataset = num_vals_dataset + 1
            new_mean = ((mean * num_vals_dataset) + x) / new_vals_dataset
            new_std_dev = np.sqrt(pow(std_dev, 2) + ((((x - mean) * (x - new_mean)) - \
                pow(std_dev, 2)) / new_vals_dataset))
            dataset = np.array([[new_mean, new_std_dev, new_vals_dataset], []])

    # else add values to dataset
    else:
        for x in MSE_vals:
            dataset[1].append(x)

    # return updated dataset
    return dataset

def alignImages(imgs, template, template_reduced_noise, img_names, imgs_apply,
    debug_directory_registered, debug_directory_matches, debug, dataset,
    dataset_enabled, MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING, NUM_STD_DEV):
    '''
    Align a set of images to the provided template
    @param imgs the images to be aligned
    @param template the template image to which the images are aligned
    @param img_names the corresponding names of the images
    @param imgs_apply the colour images that the transformation matrices are
        applied to
    @param debug_directory_registered the directory where registered images
        are written if debugging
    @param debug_directory_matches the directory where matches images
        are written in debugging
    @param debug flag indicating whether running in debugging mode
    @param dataset registration dataset
    @param dataset_enabled flag indicating whether registration dataset is
        enabled
    @param MAX_ROTATION maximum allowed rotation
    @param MAX_TRANSLATION maximum allowed translation
    @param MAX_SCALING maximum allowed scaling
    @param NUM_STD_DEV number of standard deviations away from the mean the
        MSE of the affine matrix can be
    @type imgs list(cv2.image)
    @type template cv2.image
    @type img_names list(string)
    @type img_apply list(cv2.image)
    @type debug_directory_registered string
    @type debug_directory_matches string
    @type debug bool
    @type dataset list(list(mean, std_dev, number), list(data))
    @type dataset_enabled bool
    @type MAX_ROTATION float
    @type MAX_TRANSLATION float
    @type MAX_SCALING float
    @type NUM_STD_DEV int
    @return registeredImages list of registered images
    @return dataset updated dataset
    @return images_names_registered names of registered images
    @rtype registeredImages list(cv2.image)
    @rtype dataset list(list(mean, std_dev, number), list(data))
    @rtype images_names_registered list(string)
    '''

    # get maximum mean squared error
    max_mean_squared_error = getMaxError(MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING)

    # create output list for images
    registeredImages = list()
    images_names_registered = list()

    # contains output data for JSON file
    registration_output = dict()

    # list to hold new MSE values
    MSE_vals = list()

    # iterator
    count = 0

    # iterate through images
    for img in tqdm.tqdm(imgs):
        # get application image
        img_apply = imgs_apply[count]

        # align image
        output = register(img, img_names[count], template, template_reduced_noise,
            img_apply, debug, debug_directory_registered, debug_directory_matches,
            dataset, dataset_enabled, NUM_STD_DEV, max_mean_squared_error)

        # if image was aligned
        if output[0] is not None:
            # unpack return
            (imgAligned, name, MSE, ORBFlag, ORBMatrix, ECCFlag, ECCMatrix) = output

            # add data to JSON output
            registration_output[name] = {
                "ORB Aligned": ORBFlag,
                "ORB Matrix": ORBMatrix,
                "ECC Aligned": ECCFlag,
                "ECC Matrix": ECCMatrix
            }

            # add images to list
            registeredImages.append(imgAligned)
            images_names_registered.append(name)

            # add MSE to list
            MSE_vals.append(MSE)

        # image wasn't registered
        else:
            # unpack return
            (_, name, MSE, ORBFlag, ORBMatrix, ECCFlag, ECCMatrix) = output

            # add data to JSON output
            registration_output[name] = {
                "ORB Aligned": ORBFlag,
                "ORB Matrix": ORBMatrix,
                "ECC Aligned": ECCFlag,
                "ECC Matrix": ECCMatrix
            }

        # increment iterator
        count += 1

    # update dataset
    print "Updating Dataset..."
    dataset = updateDataset(dataset, MSE_vals, dataset_enabled)

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory_registered + 'registered.json', 'w')
        json.dump(registration_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # return list of registered images
    return registeredImages, dataset, images_names_registered

def unpackArgs(args):
    '''
    Function to unpack arguments explicitly
    @param args function arguments
    @type args arguments
    @return output of register function
    @rtype (imgAligned, name, MSE, ORBFlag, ORBMatrix, ECCFlag, ECCMatrix)
    '''
    return register(*args)

def alignImagesParallel(pool, imgs, template, template_reduced_noise, img_names,
     imgs_apply, debug_directory_registered, debug_directory_matches, debug, dataset,
    dataset_enabled, MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING, NUM_STD_DEV):
    '''
    Align a set of images to the provided template using a parallel pool to
        improve efficiency when working with large image sets
    @param pool the parallel pool used for computing
    @param imgs the images to be aligned
    @param template the template image to which the images are aligned
    @param img_names the corresponding names of the images
    @param imgs_apply the colour images that the transformation matrices are
        applied to
    @param debug_directory_registered the directory where registered images
        are written if debugging
    @param debug_directory_matches the directory where matches images
        are written in debugging
    @param debug flag indicating whether running in debugging mode
    @param dataset registration dataset
    @param dataset_enabled flag indicating whether registration dataset is
        enabled
    @param MAX_ROTATION maximum allowed rotation
    @param MAX_TRANSLATION maximum allowed translation
    @param MAX_SCALING maximum allowed scaling
    @param NUM_STD_DEV number of standard deviations away from the mean the
        MSE of the affine matrix can be
    @type pool multiprocessing pool
    @type imgs list(cv2.image)
    @type template cv2.image
    @type img_names list(string)
    @type img_apply list(cv2.image)
    @type debug_directory_registered string
    @type debug_directory_matches string
    @type debug bool
    @type dataset list(list(mean, std_dev, number), list(data))
    @type dataset_enabled bool
    @type MAX_ROTATION float
    @type MAX_TRANSLATION float
    @type MAX_SCALING float
    @type NUM_STD_DEV int
    @return registeredImages list of registered images
    @return dataset updated dataset
    @return images_names_registered names of registered images
    @rtype registeredImages list(cv2.image)
    @rtype dataset list(list(mean, std_dev, number), list(data))
    @rtype images_names_registered list(string)
    '''

    # get maximum mean squared error
    max_mean_squared_error = getMaxError(MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING)

    # create output list for images
    registeredImages = list()
    images_names_registered = list()

    # contains output data for JSON file
    registration_output = dict()

    # list to hold new MSE values
    MSE_vals = list()

    # create task list for pool
    tasks = list()
    for i, img in enumerate(imgs):
        tasks.append((img, img_names[i], template, template_reduced_noise,
            imgs_apply[i], debug, debug_directory_registered, debug_directory_matches,
            dataset, dataset_enabled, NUM_STD_DEV, max_mean_squared_error))

    # run tasks using pool
    for i in tqdm.tqdm(pool.imap(unpackArgs, tasks), total=len(tasks)):
        # if image was aligned
        if i[0] is not None:
            # unpack return
            (imgAligned, name, MSE, ORBFlag, ORBMatrix, ECCFlag, ECCMatrix) = i

            # add data to JSON output
            registration_output[name] = {
                "ORB Aligned": ORBFlag,
                "ORB Matrix": ORBMatrix,
                "ECC Aligned": ECCFlag,
                "ECC Matrix": ECCMatrix
            }

            # add images to list
            registeredImages.append(imgAligned)
            images_names_registered.append(name)

            # add MSE to list
            MSE_vals.append(MSE)

        # image wasn't registered
        else:
            # unpack return
            (_, name, MSE, ORBFlag, ORBMatrix, ECCFlag, ECCMatrix) = i

            # add data to JSON output
            registration_output[name] = {
                "ORB Aligned": ORBFlag,
                "ORB Matrix": ORBMatrix,
                "ECC Aligned": ECCFlag,
                "ECC Matrix": ECCMatrix
            }

    # update dataset
    print "Updating Dataset..."
    dataset = updateDataset(dataset, MSE_vals, dataset_enabled)

    # if in debugging mode
    if(debug):
        # output JSON file
        file = open(debug_directory_registered + 'registered.json', 'w')
        json.dump(registration_output, file, sort_keys=True, indent=4, separators=(',', ': '))

    # return list of registered images
    return registeredImages, dataset, images_names_registered
