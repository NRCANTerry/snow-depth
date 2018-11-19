# import necessary packages
import cv2
import numpy as np
import json
import tqdm
from matplotlib import pyplot as plt
import timeit

# constants
MAX_FEATURES = 262144

img = cv2.imread("MFDC3488.JPG", 0)
template = cv2.imread("MFDC3367.JPG", 0)
img_apply = cv2.imread("MFDC3488.JPG")

height, width = template.shape
maxHeight = 1080.0
maxWidth = 1920.0
resizeFactor = min(maxWidth/float(width), maxHeight/float(height))
print resizeFactor
img_small = cv2.resize(img, None, fx=resizeFactor, fy=resizeFactor)
template_small = cv2.resize(template, None, fx=resizeFactor, fy=resizeFactor)
img_apply_small = cv2.resize(img_apply, None, fx=resizeFactor, fy=resizeFactor)

# detect ORB features and compute descriptors
orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
kp1, desc1 = orb.detectAndCompute(img_small, None)
kp2, desc2 = orb.detectAndCompute(template_small, None)

# create brute-force matcher object and match descriptors
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#matches = bf.match(desc1, desc2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
knnMatches = bf.knnMatch(desc1, desc2, k=2)

# sort matches by score and remove poor matches
# matches with a score greater than 30 are removed
#matches = [x for x in matches if x.distance <= 30]
#matches = sorted(matches, key=lambda x: x.distance)
#if(len(matches) > 100):
#    matches = matches[:100]

matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        matches.append(m)

# draw top matches
imgMatches = cv2.drawMatches(img_small, kp1, template_small, kp2, matches, None, flags=2)
cv2.imwrite("matches.jpg", imgMatches)

# extract location of good matches
points1 = np.zeros((len(matches), 2), dtype = np.float32)
points2 = np.zeros((len(matches), 2), dtype = np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# determine affine 2D transformation using RANSAC robust method
affine_matrix = cv2.estimateAffine2D(points1, points2, method = cv2.RANSAC)[0]
#affine_matrix *= np.array([[1, 1, 1/resizeFactor], [1, 1, 1/resizeFactor]], dtype=np.float32)
height, width = template.shape

imgReg = cv2.warpAffine(img_apply_small, affine_matrix, (width, height))
affine_matrix *= np.array([[1, 1, 1/resizeFactor], [1, 1, 1/resizeFactor]], dtype=np.float32)
imgRegBig = cv2.warpAffine(img_apply, affine_matrix, (width, height))
imgRegGray = cv2.cvtColor(imgReg, cv2.COLOR_BGR2GRAY)
imgRegBigGray = cv2.cvtColor(imgRegBig, cv2.COLOR_BGR2GRAY)
print('ORB time:', timeit.default_timer() - pyr_start_time)

# define ECC motion model
warp_mode = cv2.MOTION_AFFINE
warp_matrix = np.eye(2, 3, dtype=np.float32)

# specify the number of iterations and threshold
number_iterations = 2000#1000
termination_thresh = 1e-7#1e-7
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)
warp_matrix = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)

# run ECC algorithm (results are stored in warp matrix)
warp_matrix = cv2.findTransformECC(template_small, imgRegGray, warp_matrix, warp_mode, criteria)[1]
warp_matrix *= np.array([[1, 1, 1/resizeFactor], [1, 1, 1/resizeFactor]], dtype=np.float32)
#imgECCAligned = cv2.warpAffine(imgRegBig, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#warp *= np.array([[1, 1, 1/resizeFactor], [1, 1, 1/resizeFactor]], dtype=np.float32)
height, width = img.shape
imgECCAligned = cv2.warpAffine(img, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

cv2.imwrite("IMG_22522.jpg", imgECCAligned)

print('Pyramid time:', timeit.default_timer() - pyr_start_time)
