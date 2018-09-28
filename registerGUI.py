# import necessary packages
import cv2
import numpy as np
import Tkinter as tk
import tkFileDialog

# global variables
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

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
def alignImages(img1, img2):

	# convert images to grayscale
	img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# detect ORB features and compute descriptors
	orb = cv2.ORB_create(MAX_FEATURES)
	kp1, desc1 = orb.detectAndCompute(img1Gray, None)
	kp2, desc2 = orb.detectAndCompute(img2Gray, None)

	# match features
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(desc1, desc2, None)

	# sort matches by score
	matches.sort(key = lambda x: x.distance, reverse = False)

	# remove poor matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	# draw top matches
	imgMatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
	cv2.imwrite("matches.jpg", imgMatches)

	# extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype = np.float32)
	points2 = np.zeros((len(matches), 2), dtype = np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	# find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# use homography
	height, width, channels = img2.shape
	img1Reg = cv2.warpPerspective(img1, h, (width, height))

	return img1Reg, h

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
imReg, h = alignImages(img, imReference)

# write aligned image to disk
outputFile = "aligned.jpg"
print("Saving aligned image :", outputFile)
cv2.imwrite(outputFile, imReg)

# print estimated homography
print("Estimated homography: \n", h)