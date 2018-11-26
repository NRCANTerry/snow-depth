# import necessary packages
import cv2
import numpy as np
import time
import os

MAX_ROTATION = 5
MAX_SCALING = 110
MAX_TRANSLATION = 200

def validTransform(MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING, matrix):
    '''
    Determine whether a transformation is valid based on parameters
    @param MAX_ROTATION maximum allowed rotation
    @param MAX_TRANSLATION maximum allowed translation
    @param MAX_SCALING maximum allowed scaling
    @param matrix the generated transformation matrix
    @type MAX_ROTATION float
    @type MAX_TRANSLATION float
    @type MAX_SCALING float
    @type matrix np.array()
    @return whether the transformation is valid (won't destroy image)
    @rtype bool
    '''

    # calculate range for alpha
    scale = abs(100-MAX_SCALING) / 100.0
    alpha_low = (1-scale) * np.cos(np.deg2rad(MAX_ROTATION))
    alpha_high = (1+scale) * np.cos(np.deg2rad(0))

    # verify affine matrix elements are within range
    if(alpha_low <= abs(matrix[0][0]) and abs(matrix[0][0]) <= alpha_high and
        alpha_low <= abs(matrix[1][1]) and abs(matrix[1][1]) <= alpha_high and
        abs(matrix[0][2]) < MAX_TRANSLATION and abs(matrix[1][2]) < MAX_TRANSLATION):
        return True
    else:
        return False

MAX_FEATURES = 262144
import timeit
# path to images
#img_path = "C:\\Users\\tbaricia\\Documents\\GitHub\\snow-depth\\measure-depth\\2018-11-21-14-13-48\\equalized\\"
img_path = "C:\\Users\\tbaricia\\Documents\\GitHub\\snow-depth\\measure-depth\\2018-11-22-13-35-28\\equalized\\"

# path to template
#temp_path = "C:\\Users\\tbaricia\\Documents\\GitHub\\snow-depth\\measure-depth\\2018-11-21-14-13-48\\equalized-template\\IMG_2684-dorset-no-snow.JPG"
temp_path = "C:\\Users\\tbaricia\\Documents\\GitHub\\snow-depth\\registration-testing\\MFDC4130.JPG"

# output path
output_path = "C:\\Users\\tbaricia\\Documents\\Test\\"

# get images
imgs = [x for x in os.listdir(img_path)]

# variables for each registration type
ORB_time = 0
SIFT_time = 0
ORB_valid = 0
SIFT_valid = 0

# total number of images
total_ims = len(imgs)

# import template
template = cv2.imread(temp_path, 0)
template = cv2.imread(temp_path)
height, width = template.shape[:2]
h=height
w=width

red_low1 = np.array([0,50,50])
red_high1 = np.array([10,255,255])
red_low2 = np.array([160,50,50])
red_high2 = np.array([180,255,255])

template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
tmask1 = cv2.inRange(template, red_low1, red_high1)
tmask2 = cv2.inRange(template, red_low2, red_high2)
tmask = cv2.bitwise_or(tmask1, tmask2)
#template = cv2.bilateralFilter(template, 9, 75, 75)

# iterate through images
for img_ in imgs:
    # output image name
    print(img_)
    filename, ext = os.path.splitext(img_)

    # read in image
    img = cv2.imread(img_path + img_, 0)
    img = cv2.imread(img_path + img_)

    #img = cv2.bilateralFilter(img, 9, 75, 75)

    gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0)
    gaussian_3t = cv2.GaussianBlur(template, (9,9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    unsharpt = cv2.addWeighted(template, 1.5, gaussian_3t, -0.5, 0, template)

    ''' ORB '''
    # start timer
    startORB = time.time()

    '''
    # define ECC motion model
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # specify the number of iterations and threshold
    number_iterations = 2000
    termination_thresh = 1e-8
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

    warp_matrix = cv2.findTransformECC(unsharpt, unsharp, warp_matrix, warp_mode, criteria)[1]
    imgECCAligned = cv2.warpAffine(img, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # write image to directory
    cv2.imwrite(output_path + filename + "-ECC" + ext, imgECCAligned)
    ORB_time += time.time() - startORB
    print("ORB %0.2f" % (time.time() - startORB))
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    immask1 = cv2.inRange(img.copy(), red_low1, red_high1)
    immask2 = cv2.inRange(img.copy(), red_low2, red_high2)
    immask = cv2.bitwise_or(immask1, immask2)

    # detect ORB features and compute descriptors
    orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
    #kp1, desc1 = orb.detectAndCompute(img.copy(), None)
    #kp2, desc2 = orb.detectAndCompute(template, None)
    kp1, desc1 = orb.detectAndCompute(immask, None)
    kp2, desc2 = orb.detectAndCompute(tmask, None)

    # create brute-force matcher object and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # sort matches by score and remove poor matches
    # matches with a score greater than 30 are removed
    #matches = [x for x in matches if x.distance <= 30]
    matches = sorted(matches, key=lambda x: x.distance)
    #if(len(matches) > 100):
    #   matches = matches[:100]

    good = []
    thresholdDist = 0.01 * np.sqrt(float(h*h + w*w))
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        dist = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        if(dist < thresholdDist):
            good.append(m)
        if(len(good) >= 1000): break
            #print(pt1, pt2)

    print(len(good))
    #if(len(good) > 2000):
    #    good = sorted(good, key=lambda x: x.distance)
    #    good = good[:1999]

    # extract location of good matches
    points1 = np.zeros((len(good), 2), dtype = np.float32)
    points2 = np.zeros((len(good), 2), dtype = np.float32)

    for i, match in enumerate(good):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # determine affine 2D transformation using RANSAC robust method
    affine_matrix = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC)[0]

    print(affine_matrix)

    # get mean squared error between affine matrix and zero matrix
    zero_matrix = np.zeros((2,3), dtype=np.float32)
    mean_squared_error = np.sum(np.square(abs(affine_matrix) - zero_matrix))

    # calculate maximum alpha
    alpha = 1.10 * np.cos(np.deg2rad(25))
    diff = abs(alpha - affine_matrix[0][0]) / alpha

    if validTransform(MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING, affine_matrix):
        ORB_valid += 1
        print("Good")

    # stop timer
    ORB_time += time.time() - startORB
    print("ORB %0.2f" % (time.time() - startORB))

    # write image to directory
    imgReg = cv2.warpAffine(img.copy(), affine_matrix, (width, height))
    cv2.imwrite(output_path + filename + "-ORB" + ext, imgReg)
    imgMatches = cv2.drawMatches(img, kp1, template, kp2, good, None, flags=2)
    cv2.imwrite(output_path + filename + "-ORB-Matches" + ext, imgMatches)

    ''' SIFT '''
    # start timer
    '''
    startSIFT = time.time()

    #fast = cv2.FastFeatureDetector_create()
    #kp1 = fast.detect(img, None)
    #kp2 = fast.detect(template, None)

    # detect SIFT features and compute descriptors
    sift = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)
    kp1, desc1 = sift.detectAndCompute(img, None)
    kp2, desc2 = sift.detectAndCompute(template, None)
    #kp1, desc1 = sift.compute(img, kp1)
    #kp2, desc2 = sift.compute(template, kp2)

    # create brute-force matcher and match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dist = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            if(dist < thresholdDist):
                good.append([m])
                #print(pt1, pt2)

    # extract location of good matches
    points1 = np.zeros((len(good), 2), dtype = np.float32)
    points2 = np.zeros((len(good), 2), dtype = np.float32)

    for i, match in enumerate(good):
        points1[i, :] = kp1[match[0].queryIdx].pt
        points2[i, :] = kp2[match[0].trainIdx].pt

    # determine affine 2D transformation using RANSAC robust method
    affine_matrix = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC)[0]

    # get mean squared error between affine matrix and zero matrix
    zero_matrix = np.zeros((2,3), dtype=np.float32)
    mean_squared_error = np.sum(np.square(abs(affine_matrix) - zero_matrix))

    # calculate maximum alpha
    alpha = 1.10 * np.cos(np.deg2rad(25))
    diff = abs(alpha - affine_matrix[0][0]) / alpha

    if validTransform(MAX_ROTATION, MAX_TRANSLATION, MAX_SCALING, affine_matrix):
        SIFT_valid += 1
        print("Good")

    # stop timer
    SIFT_time += time.time() - startSIFT
    print("SIFT %0.2f" % (time.time() - startSIFT))

    # write image to directory
    imgReg = cv2.warpAffine(img.copy(), affine_matrix, (width, height))
    cv2.imwrite(output_path + filename + "-SIFT" + ext, imgReg)
    '''
# output results
print("Results")
print("Total Images %d" % total_ims)
print("ORB Valid %d" % ORB_valid)
print("Average ORB Time %0.2f" % (ORB_time / float(ORB_valid)))
#print("SIFT Valid %d" % SIFT_valid)
#print("Average SIFT Time %0.2f" % (SIFT_time / float(SIFT_valid)))
