# import necessary packages
import cv2
import numpy as np
import Tkinter as tk
import tkFileDialog
from matplotlib import pyplot as plt
import statistics
from register import alignImages3

# global variables
MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15

# function to increase the brightness of an image
def increase_brightness(img, val):
	# convert image to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	# increase value of pixels
	lim = 255 - val
	v[v > lim] = 255
	v[v <= lim] += val

	# merge channels
	hsv_merge = cv2.merge((h,s,v))

	# convert image to BGR and return
	img = cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)
	return img

# GUI class
class GUI:
	def __init__(self):
		# open window
		self.root = tk.Tk()
		self.root.configure(background='#ffffff')
		self.root.title("Select Images")

		# variables
		self.templateDir = ""
		self.inputDir = ""

		# label
		self.label = tk.Label(
			self.root,
			text = "Register Image to Template",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri Light", 18))

		# buttons
		self.image1Button = tk.Button(
			self.root,
			text = "Select Template",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.selectFile("1"),
			width = 17,
			font = ("Calibri Light", 14))

		self.image2Button = tk.Button(
			self.root,
			text = "Select Image",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.selectFile("2"),
			width = 17,
			font = ("Calibri Light", 14))

		self.execute = tk.Button(
			self.root,
			text = "Register",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.root.destroy(),
			width = 17,
			font = ("Calibri Light", 14))

		# packing
		self.label.pack(pady = 20, padx = 50)
		self.image1Button.pack(pady = (20,5), padx = 50)
		self.image2Button.pack(pady = (5,20), padx = 50)
		self.execute.pack(pady = (5,20), padx = 50)

		self.root.mainloop()

	def selectFile(self, method):
		# open file selector
		if(method == "1"):
			self.templateDir = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		elif(method == "2"):
			self.inputDir = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

	def getDirs(self):
		return self.templateDir, self.inputDir

# function to align image to template
def alignImages(img1_, img2_):

	# apply median blur to highlight foreground features
	img1 = cv2.medianBlur(img1_, 5)
	img2 = cv2.medianBlur(img2_, 5)

	# convert images to grayscale
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# denoise grayscale image
	img1 = cv2.fastNlMeansDenoising(img1,None,5,10,7)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(nfeatures = 5000)
	kp1, desc1 = orb.detectAndCompute(img1, None)
	kp2, desc2 = orb.detectAndCompute(img2, None)

	# create brute-force matcher object and match descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(desc1, desc2)

	# sort matches by score and remove poor matches
	# matches with score of lower than 30 are removed
	matches = sorted(matches, key = lambda x: x.distance)
	matches = [x for x in matches if x.distance <= 30]

	# draw top matches
	imgMatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
	cv2.imwrite("./matches.jpg", imgMatches)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	h2, mask_new = cv2.findHomography(points1, points2, cv2.RANSAC)
	h2, mask2 = cv2.findHomography(points1, points2, cv2.LMEDS, mask = mask_new)

	# use homography
	height, width, channels = img2_.shape
	img1Reg = cv2.warpPerspective(img1_, h2, (width, height))

	# return registered images and homography
	return img1Reg, h2

# function to align image to template
def alignImages2(img, template):
	# apply median blur to highlight foreground featurse
	img_blur = cv2.medianBlur(img, 5)
	template_blur = cv2.medianBlur(template, 5)

	# convert images to grayscale
	img1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# denoise grayscale image
	img1Gray = cv2.fastNlMeansDenoising(img1Gray,None,5,10,7)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(nfeatures = MAX_FEATURES)
	kp1, desc1 = orb.detectAndCompute(img_blur, None)
	kp2, desc2 = orb.detectAndCompute(template_blur, None)

	# create brute-force matcher object and match descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	matches = bf.match(desc1, desc2)

	# sort matches by score and remove poor matches
	# matches with a score greater than 30 are removed
	matches = sorted(matches, key = lambda x: x.distance)
	matches = [x for x in matches if x.distance <= 30]

	# draw top matches
	imgMatches = cv2.drawMatches(img, kp1, template, kp2, matches, None)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# determine homography
	# apply RANSAC-based robust method first then Least-Median robust method
	RANSAC_h, RANSAC_mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	LMEDS_h, LMEDS_mask = cv2.findHomography(points1, points2, cv2.LMEDS, mask = RANSAC_mask)

	# use homography
	height, width, channels = template.shape
	imgReg = cv2.warpPerspective(img, LMEDS_h, (width, height))

	# convert registered image to grayscale
	imgRegGray = cv2.cvtColor(imgReg, cv2.COLOR_BGR2GRAY)

	# define ECC motion model
	warp_mode = cv2.MOTION_HOMOGRAPHY

	# define 3x3 matrix
	warp_matrix = np.eye(3, 3, dtype=np.float32)

	# specify the number of iterations and threshold
	number_iterations = 250
	termination_thresh = 1e-3

	# define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_iterations,  termination_thresh)

	# run ECC algorithm (results are stored in warp matrix)
	warp_matrix = cv2.findTransformECC(img2Gray, imgRegGray, warp_matrix, warp_mode, criteria)[1]

	# align image
	imgECCAligned = cv2.warpPerspective(imgReg, warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	# return aligned image and matches
	return imgECCAligned, imgMatches, warp_matrix


# open GUI
gui = GUI()

template, image = gui.getDirs()

# read reference image
print("Reading reference image :", template)
imReference = cv2.imread(template)

# read image to be aligned
print("Reading image to be aligned :", image)
img = cv2.imread(image)

print("Aligning images ...")

# Registered image stored in imReg
# Estimated homography stored in h
imReg, _, warp_matrix = alignImages3(img.copy(), imReference.copy())

'''
# Convert images to grayscale
im1_gray = cv2.cvtColor(imReference,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(imReg,cv2.COLOR_BGR2GRAY)

# Find size of image1
sz = imReference.shape

# Define the motion model
warp_mode = cv2.MOTION_HOMOGRAPHY

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 250

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-2

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective (imReg, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(imReg, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
'''
# write aligned image to disk
outputFile = "aligned.jpg"
#outputFile = "MFD.jpg"
print("Saving aligned image :", outputFile)
cv2.imwrite(outputFile, imReg)
#cv2.imwrite(outputFile, im2_aligned)

# print estimated homography
print "Estimated homography: \n", warp_matrix
