# import necessary packages
import cv2
import numpy as np
import Tkinter as tk
import tkFileDialog
from matplotlib import pyplot as plt
import statistics
import time
import math
import sys

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

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        print "Lowval: ", low_val
        print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

# open GUI
gui = GUI()

template, image = gui.getDirs()

# read reference image
print("Reading reference image :", template)
imReference = cv2.imread(template)

imReference2 = simplest_cb(imReference, 5)

cv2.imshow("Before", cv2.resize(imReference, None, fx =0.25,fy= 0.25))
cv2.imshow("Balanced", cv2.resize(imReference2, None, fx=0.25, fy=0.25))
cv2.waitKey()
