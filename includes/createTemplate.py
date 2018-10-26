# import necessary modules
import Tkinter as tk
from tkFont import Font
import tkFileDialog
import tkMessageBox
import tkSimpleDialog
import ttk
import numpy as np
import ConfigParser
import ast
import cv2
import sys
import os
from PIL import ImageTk, Image
from order_points import orderPoints
from get_intersection import lineIntersections
from scipy import ndimage
import statistics
from get_tensor import getTensor
from equalize import equalize_hist_colour
import matplotlib
from matplotlib import pyplot as plt
from progress_bar import progress
from scipy.signal import find_peaks
from scipy import signal
from scipy import ndimage
import math

# create template window class
class createTemplate:
    def __init__(self, master):
        # setup window
        self.root = master
        self.root.configure(bg='#243447')
        self.root.title("Generate Snow Depth Template")
        self.root.iconbitmap(default="transparent.ico")

        #-----------------------------------------------------------------------
        # Setup
        #-----------------------------------------------------------------------

        self.titleFont = Font(family = "Calibri Light", size = 30)
        self.largeFont = Font(family  = "Calibri Light", size = 24)
        self.mediumFont = Font(family = "Calibri Light", size = 18)
        self.smallFont = Font(family = "Calibri Light", size = 16)
        self.entryFont = Font(family = "Calibri Light", size = 14)
        self.boldFont = Font(family = "Calibri", size = 19)
        self.gray = "#243447"
        self.white = '#ffffff'

        #-----------------------------------------------------------------------
        # System parameters
        #-----------------------------------------------------------------------

        self.systemParameters = {
            "Rotation": 0.0,
            "Translation": 0.0,
            "Scale": 0.0,
            "Tensor_STD_DEV": 0.0,
            "Register_STD_DEV": 0.0,
            "Name": ""
        }

        #-----------------------------------------------------------------------
        # Variables
        #-----------------------------------------------------------------------

        # preferences lists
        self.templateCoordinates = list()
        self.rawTemplateCoordinates = list()
        self.templateIntersections = list()
        self.templateDistances = list()
        self.templateTensors = list()
        self.blobSizeRanges = list()

        self.templatePath = ""
        self.rectList = list()
        self.lastCoord = list()
        self.firstCoord = True
        self.numRect = 0
        self.stakeNum = 0
        self.blobNum = 0
        self.cropRatio = 0.0
        self.blobIndex = -1
        self.windowClosed = False
        self.templateSaved = False

        #-----------------------------------------------------------------------
        # Frames
        #-----------------------------------------------------------------------

        self.leftFrame = tk.Frame(self.root, bg = self.gray)
        self.tensorFrame = tk.Frame(self.leftFrame, bg = self.gray)
        self.registerFrame = tk.Frame(self.leftFrame, bg = self.gray)

        self.rightFrame = tk.Frame(self.root, bg = self.gray)
        self.rotationFrame = tk.Frame(self.rightFrame, bg = self.gray)
        self.translationFrame = tk.Frame(self.rightFrame, bg = self.gray)
        self.scaleFrame = tk.Frame(self.rightFrame, bg = self.gray)

        #-----------------------------------------------------------------------
        # Labels
        #-----------------------------------------------------------------------

        # left frame
        self.titleLabel = tk.Label(self.root, text = "Generate Template", bg = self.gray, fg = self.white, font = self.titleFont)
        self.directoryLabel = tk.Label(self.leftFrame, text = "Select Marked Template", bg = self.gray, fg = self.white, font = self.boldFont)
        self.pathLabel = tk.Label(self.leftFrame, text = "No Template Selected", bg = self.gray, fg = self.white, font = self.smallFont)
        self.parametersLabel = tk.Label(self.leftFrame, text = "Parameters", bg = self.gray, fg = self.white, font = self.boldFont)
        self.tensorLabel = tk.Label(self.tensorFrame, text = "Tensor Std Dev", bg = self.gray, fg = self.white, font = self.smallFont)
        self.registerLabel = tk.Label(self.registerFrame, text = "Register Std Dev", bg = self.gray, fg = self.white, font = self.smallFont)

        # right frame
        self.instructionsLabel = tk.Label(self.rightFrame, text = "Maximum Transformation", bg = self.gray, fg = self.white, font = self.boldFont)
        self.rotationLabel = tk.Label(self.rotationFrame, text = "Rotation", bg = self.gray, fg = self.white, font = self.smallFont)
        self.rotationUnitsLabel = tk.Label(self.rotationFrame, text = "degrees", bg = self.gray, fg = self.white, font = self.smallFont)
        self.translationLabel = tk.Label(self.translationFrame, text = "Translation", bg = self.gray, fg = self.white, font = self.smallFont)
        self.translationUnitsLabel = tk.Label(self.translationFrame, text = "pixels", bg = self.gray, fg = self.white, font = self.smallFont)
        self.scaleLabel = tk.Label(self.scaleFrame, text = "Scale", bg = self.gray, fg = self.white, font = self.smallFont)
        self.scaleUnitsLabel = tk.Label(self.scaleFrame, text = "%", bg = self.gray, fg = self.white, font = self.smallFont)

        #-----------------------------------------------------------------------
        # Entries
        #-----------------------------------------------------------------------

        # create validate command
        validateCommand = self.root.register(self.validate)

        # left frame
        self.tensorEntry = tk.Entry(self.tensorFrame, font = self.entryFont,
            validate = "key", validatecommand = (validateCommand, '%P', 'Tensor_STD_DEV'), width = 10)
        self.registerEntry = tk.Entry(self.registerFrame, font = self.entryFont,
            validate = "key", validatecommand = (validateCommand, '%P', 'Register_STD_DEV'), width = 10)

        # right frame
        self.rotationEntry = tk.Entry(self.rotationFrame, font = self.entryFont,
            validate = "key", validatecommand = (validateCommand, '%P', 'Rotation'), width = 10)
        self.translationEntry = tk.Entry(self.translationFrame, font = self.entryFont,
            validate = "key", validatecommand = (validateCommand, '%P', 'Translation'), width = 10)
        self.scaleEntry = tk.Entry(self.scaleFrame, font = self.entryFont,
            validate = "key", validatecommand = (validateCommand, '%P', 'Scale'), width = 10)

        # list of entries
        self.entriesCheck = [self.rotationEntry, self.translationEntry, self.scaleEntry, self.tensorEntry,
            self.registerEntry]

        #-----------------------------------------------------------------------
        # Buttons
        #-----------------------------------------------------------------------

        self.continueButton = tk.Button(self.rightFrame, text = "Continue", bg = self.gray, fg = self.white, font = self.smallFont,
            command = lambda: self.generateTemplate(), width = 25)
        self.directoryButton = tk.Button(self.leftFrame, text = "Select Template", bg = self.gray, fg = self.white, font = self.smallFont,
            command = lambda: self.getImage(), width = 25)

        #-----------------------------------------------------------------------
        # Packing
        #-----------------------------------------------------------------------

        self.titleLabel.pack(pady = (30,20))
        self.leftFrame.pack(side = tk.LEFT, padx = 50, pady = (0,10))
        self.directoryLabel.pack(pady = 10)
        self.pathLabel.pack(pady = 10)
        self.directoryButton.pack(pady = 10)

        self.parametersLabel.pack(pady = (20,10))
        self.tensorFrame.pack(pady = 13)
        self.tensorLabel.pack(side = tk.LEFT, padx = (20,5))
        self.tensorEntry.pack(side = tk.LEFT, padx = (10,20))

        self.registerFrame.pack(pady = (13, 46))
        self.registerLabel.pack(side = tk.LEFT, padx = (17,5))
        self.registerEntry.pack(side = tk.LEFT, padx = (5,20))

        self.rightFrame.pack(side = tk.RIGHT, padx = 50, pady = (0,20))
        self.instructionsLabel.pack(pady = 10)
        self.rotationFrame.pack(pady = 20)
        self.rotationLabel.pack(side = tk.LEFT, padx = (41,5))
        self.rotationEntry.pack(side = tk.LEFT, padx = 5)
        self.rotationUnitsLabel.pack(side = tk.LEFT, padx = (5,20))

        self.translationFrame.pack(pady = 20)
        self.translationLabel.pack(side = tk.LEFT, padx = (0,5))
        self.translationEntry.pack(side = tk.LEFT, padx = 5)
        self.translationUnitsLabel.pack(side = tk.LEFT, padx = (5,20))

        self.scaleFrame.pack(pady = 20)
        self.scaleLabel.pack(side = tk.LEFT, padx = (20,5))
        self.scaleEntry.pack(side = tk.LEFT, padx = 5)
        self.scaleUnitsLabel.pack(side = tk.LEFT, padx = (5,20))

        self.continueButton.pack(pady = (25,35))

    #-----------------------------------------------------------------------
    # Function to get template paths
    #-----------------------------------------------------------------------

    def getImage(self):
        # open file selector
        filename = tkFileDialog.askopenfilename(initialdir = "/",title = "Select " + str(type) + \
            " template",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        shortName = os.path.split(filename)[1]

        # if valid filename
        if(filename != ""):
            self.templatePath = filename
            self.pathLabel.config(text=shortName)

    #-----------------------------------------------------------------------
    # Function to validate text entry
    #-----------------------------------------------------------------------

    def validate(self, new_text, entry_field):
        # the field is being cleared
        if not new_text:
            self.systemParameters[str(entry_field)] = 0.0

        # update system parameters with new text
        try:
            if(new_text == ""):
                self.systemParameters[str(entry_field)] = 0.0
            else:
                self.systemParameters[str(entry_field)] = float(new_text)
            return True

        except ValueError:
            return False

    #-----------------------------------------------------------------------
    # Function to check if all fields are filled
    #-----------------------------------------------------------------------

    def fieldsFilled(self):
        if(all(v.get() != "" for v in self.entriesCheck)):
            return True
        else:
            return False

    #-----------------------------------------------------------------------
    # Function to generate template
    #-----------------------------------------------------------------------

    def generateTemplate(self):
        # check that all required fields are filled in
        if(not self.fieldsFilled() or self.templatePath == ""):
            tkMessageBox.showinfo("Error", "Not All Fields Populated")
            return

        # hide window
        self.root.withdraw()

        # create top level window for template creation
        self.templateWindow = tk.Toplevel(self.root)
        self.templateWindow.configure(bg = self.gray)
        self.templateWindow.protocol("WM_DELETE_WINDOW", self.closeTemplateWindow)

        #-----------------------------------------------------------------------
        # Reset variables
        #-----------------------------------------------------------------------

        self.templateCoordinates = list()
        self.rectList = list()
        self.lastCoord = list()
        self.firstCoord = True
        self.numRect = 0
        self.stakeNum = 0
        self.blobNum = 0
        self.cropRatio = 0.0
        self.blobIndex = -1
        self.windowClosed = False

        #-----------------------------------------------------------------------
        # Frames
        #-----------------------------------------------------------------------

        templateFrame = tk.Frame(self.templateWindow, bg = self.gray)
        leftFrame = tk.Frame(self.templateWindow, bg = self.gray)

        #-----------------------------------------------------------------------
        # Labels
        #-----------------------------------------------------------------------

        titleLabel = tk.Label(leftFrame, text = "Generate Template", bg = self.gray, fg = self.white, font = self.titleFont)
        self.instructionsLabelTemplate = tk.Label(leftFrame, text = "Select All Stakes", bg = self.gray, fg = self.white, font = self.largeFont)
        self.stakesLabel = tk.Label(leftFrame, text = "No Stakes Selected", bg = self.gray, fg = self.white, font = self.boldFont)
        coordinateTitleLabel = tk.Label(leftFrame, text = "Last Coordinate", bg = self.gray, fg = self.white, font = self.boldFont)
        self.coordinateLabel = tk.Label(leftFrame, text = "None", bg = self.gray, fg = self.white, font = self.mediumFont)

        #-----------------------------------------------------------------------
        # Buttons
        #-----------------------------------------------------------------------

        self.buttonVar = tk.IntVar()

        undoButton = tk.Button(leftFrame, text = "Undo", bg = self.gray, fg = self.white, font = self.smallFont,
            width = 25, command = lambda: self.undo())
        self.nextButton = tk.Button(leftFrame, text = "Next", bg = self.gray, fg = self.white, font = self.smallFont,
            width = 25, command = lambda: self.next())

        #-----------------------------------------------------------------------
        # Setup canvas
        #-----------------------------------------------------------------------

        # get screen dimensions
        screen_width = float(self.root.winfo_screenwidth())
        screen_height = float(self.root.winfo_screenheight())

        # import template image
        self.cv2_img = cv2.imread(self.templatePath)
        self.equalized_img = equalize_hist_colour(self.cv2_img.copy(), 5.0, (8,8))
        self.equalized_img = cv2.cvtColor(self.equalized_img, cv2.COLOR_BGR2RGB)
        self.img_orig = Image.fromarray(self.equalized_img)
        width, height = self.img_orig.size

        # determine height and width for canvas
        self.ratio = min(screen_width/float(width), screen_height/float(height)) * 0.9
        self.imgWidth = int(float(width) * self.ratio)
        self.imgHeight = int(float(height) * self.ratio)

        # resize image and place in canvas
        img_resize = self.img_orig.resize((self.imgWidth, self.imgHeight))
        self.img = ImageTk.PhotoImage(img_resize.copy())

        # create canvas to display image
        self.canvas = tk.Canvas(templateFrame, width = self.imgWidth*0.8, height = self.imgHeight,
            bg = self.gray, scrollregion = (0, 0, self.imgWidth, self.imgHeight))

        #-----------------------------------------------------------------------
        # Packing
        #-----------------------------------------------------------------------

        # pack frames
        leftFrame.pack(side = tk.LEFT, padx = 25)
        templateFrame.pack(side = tk.RIGHT)

        # setup scrollbars to allow user to pan in image
        h_bar = tk.Scrollbar(templateFrame, orient = tk.HORIZONTAL, command = self.canvas.xview)
        h_bar.pack(side = tk.BOTTOM, fill = tk.X)
        v_bar = tk.Scrollbar(templateFrame, orient = tk.VERTICAL, command = self.canvas.yview)
        v_bar.pack(side = tk.RIGHT, fill = tk.Y)

        # display canvas
        self.canvas.config(xscrollcommand = h_bar.set, yscrollcommand = v_bar.set, bg = self.gray)
        self.canvas.pack(side = tk.LEFT, expand = tk.YES, fill = tk.BOTH)
        self.canvas.create_image(0, 0, image = self.img, anchor = tk.NW, tag = 'image')

        # bind for mouse click
        self.canvas.tag_bind('image', '<Button-1>', self.windowClick)

        # bind for mouse scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # pack left frame widgets
        titleLabel.pack(pady = 10)
        self.instructionsLabelTemplate.pack(pady = (0,10))
        self.stakesLabel.pack(pady = 20)
        coordinateTitleLabel.pack(pady = (10,0))
        self.coordinateLabel.pack(pady = (5,20))
        undoButton.pack(pady = 30)
        self.nextButton.pack(pady = 30)

        # run window
        self.root.wait_variable(self.buttonVar)

        # if window was closed early
        if(self.windowClosed):
            tkMessageBox.showinfo("Error", "Template Window Was Closed")
            return

        #-----------------------------------------------------------------------
        # Update widgets
        #-----------------------------------------------------------------------

        self.stakesLabel.config(text = "No Blobs Selected")
        self.instructionsLabelTemplate.config(text = "Select All Blobs on Stake 0")
        self.coordinateLabel.config(text = "None")
        self.nextButton.config(command = lambda: self.next(self.blobIndex))

        self.lastCoord = [0,0]
        self.firstCoord = True
        self.numRect = 0

        # run window
        self.root.wait_window(self.templateWindow)

        # if window was closed early
        if(self.windowClosed):
            tkMessageBox.showinfo("Error", "Template Window Was Closed")
            return

        # calculate tensors
        self.calculateTensors()

        # get intersection points
        self.calculateIntersections()

        # create overaly
        self.createOverlay()

    #-----------------------------------------------------------------------
    # Functions to enable mouse scrolling
    #-----------------------------------------------------------------------

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1*(event.delta/120), "units")

    def _on_mousewheel_preview(self, event):
        self.previewCanvas.yview_scroll(-1*(event.delta/120), "units")

    #-----------------------------------------------------------------------
    # Function to handle click events
    #-----------------------------------------------------------------------

    def windowClick(self, event):
        # determine coordinates of click
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # update coordinate label
        coordinateString = "(%0.2f, %0.2f)" % (x,y)
        self.coordinateLabel.config(text = coordinateString)

        # if this is the second coordinate of a stake bounding box
        if(not self.firstCoord):
            # update stakes label
            self.numRect += 1
            if(self.blobIndex == -1): self.stakesLabel.config(text = ("%d Stakes Selected" % self.numRect))
            else: self.stakesLabel.config(text = ("%d Blobs Selected" % self.numRect))

            # get last coordinate
            x0, y0 = self.lastCoord

            # draw rectangle on canvas
            if(self.blobIndex == -1): rect = self.canvas.create_rectangle(x0, y0, x, y, outline = 'red', width = 3)
            else: rect = self.canvas.create_rectangle(x0, y0, x, y, outline = 'green', width = 3)

            # add rectangle to list
            self.rectList.append(rect)

            # update bool flag
            self.firstCoord = True

        # else this is the first coordinate
        # update the bool flag
        else:
            self.firstCoord = False

        # update last coordinate variable
        self.lastCoord = [x, y]

    #-----------------------------------------------------------------------
    # Function to handle undo events
    #-----------------------------------------------------------------------

    def undo(self):
        # if rectangle was just created, delete it
        if(self.firstCoord == True and len(self.rectList) > 0):
            self.canvas.delete(self.rectList[len(self.rectList) - 1])
            self.rectList.pop()

            # reset label
            self.numRect -= 1
            if self.blobIndex == -1:
                self.stakesLabel.config(text = ("%d Stakes Selected" % self.numRect))
            else:
                self.stakesLabel.config(text = ("%d Blobs Selected" % self.numRect))

        # else if point was just created, remove that point
        else:
            self.lastCoord = [0, 0]
            self.firstCoord = True

            # reset label
            self.coordinateLabel.config(text = "(0, 0)")

    #-----------------------------------------------------------------------
    # Function that is executed after all rectangles are selected
    #-----------------------------------------------------------------------

    def next(self, index = -1):
        # stake rectangles
        if(self.numRect > 0 and index == -1):
            # sort stakes from left to right
            self.rectList.sort(key = lambda x: self.canvas.coords(x)[0])

            # update template coordinates list with stake coordinates
            for rect in self.rectList:
                # append stake coordinates to list
                coords = self.canvas.coords(rect)
                [x0, y0, x1, y1] = [float(x) / self.ratio for x in coords]
                self.rawTemplateCoordinates.append([[[x0, y0], [x1, y1]]])
                self.templateCoordinates.append([[[x0, y0], [x1, y1]]])

                # add blank list to intersection distance list
                self.templateDistances.append(list())

            # update wait variable
            self.buttonVar.set(1)

            # update stake counter
            self.stakeNum = len(self.rectList)

            # load first stake image into canvas
            [[x0, y0], [x1, y1]] = self.templateCoordinates[0][0]
            self.canvas.delete("all")
            img = self.img_orig.crop((x0, y0, x1, y1))
            width, height = img.size

            # determine new width and height to fill canvas
            self.cropRatio = max(float(self.imgWidth)/float(width), float(self.imgHeight)/float(height))
            new_width = int(float(width) * self.cropRatio)
            new_height = int(float(height) * self.cropRatio)

            # resize image and place in canvas
            img_resize = img.resize((new_width, new_height))
            self.img = ImageTk.PhotoImage(img_resize)
            self.canvas.config(scrollregion = (0, 0, new_width, new_height))
            self.canvas.create_image(0, 0, image = self.img, anchor = tk.NW, tag = 'image')

            # update blob index
            self.blobIndex = 0

        # if no stakes selected show error message
        elif(self.numRect <= 0 and index == -1):
            tkMessageBox.showinfo("Error", "No Stakes Selected")

        # blob rectangles
        elif(self.numRect > 1):
            # track average blob size
            blobArea = 0.0
            blobNum = 0.0

            # sort blobs from bottom to top
            self.rectList.sort(key = lambda x: self.canvas.coords(x)[1], reverse = True)

            # update template coordinates list with blob coordinates
            for rect in self.rectList:
                # append stake coordinates to list
                [[sx0, sy0], [sx1, sy1]] = self.rawTemplateCoordinates[self.blobIndex][0]
                coords = self.canvas.coords(rect)
                [x0, y0, x1, y1] = [float(x) / self.cropRatio for x in coords]
                [x0, y0, x1, y1] = [x0 + sx0, y0 + sy0, x1 + sx0, y1 + sy0]
                self.rawTemplateCoordinates[self.blobIndex].append([[x0, y0], [x1, y1]])

                # determine dilated blob sizes
                dilate_px = int(abs(float(x1-x0) * 0.33))
                self.templateCoordinates[self.blobIndex].append([[x0-dilate_px, y0-dilate_px], [x1+dilate_px, y1+dilate_px]])

                # append centroid coordinates to intersection list
                centroid = (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)
                self.templateDistances[self.blobIndex].append(centroid)

                # increment blob area and counter
                blobArea += abs((x1-x0) * (y1-y0))
                blobNum += 1.0
                self.blobNum += 1.0

            # determine blob ranges for stake
            avgSize = float(blobArea) / float(blobNum)
            self.blobSizeRanges.append([avgSize * 0.7, avgSize * 1.50])

            # update blob index
            self.blobIndex += 1
            if(self.blobIndex >= self.stakeNum):
                self.windowClosed = False
                self.templateWindow.destroy()
                return
            elif(self.blobIndex == (self.stakeNum - 1)):
                self.continueButton.config(text = "Generate")

            # load next stake image into canvas
            [[x0, y0], [x1, y1]] = self.templateCoordinates[self.blobIndex][0]
            self.canvas.delete("all")
            img = self.img_orig.crop((x0, y0, x1, y1))
            width, height = img.size

            # determine new width and height to fill canvas
            self.cropRatio = max(float(self.imgWidth)/float(width), float(self.imgHeight)/float(height))
            new_width = int(float(width) * self.cropRatio)
            new_height = int(float(height) * self.cropRatio)

            # resize image and place in canvas
            img_resize = img.resize((new_width, new_height))
            self.img = ImageTk.PhotoImage(img_resize)
            self.canvas.config(scrollregion = (0, 0, new_width, new_height))
            self.canvas.create_image(0, 0, image = self.img, anchor = tk.NW, tag = 'image')

            # reset labels
            self.stakesLabel.config(text = "No Blobs Selected")
            self.coordinateLabel.config(text = "None")
            self.instructionsLabelTemplate.config(text = ("Select All Blobs on Stake %d" % self.blobIndex))

            self.lastCoord = [0,0]
            self.firstCoord = True
            self.numRect = 0

        # if no blobs are selected show error message
        else:
            tkMessageBox.showinfo("Error", "Not Enough Blobs Selected (Minimum 2)")

        # clear list of rectangles
        self.rectList = list()

    #-----------------------------------------------------------------------
    # Function to calculate tensor for each stake
    #-----------------------------------------------------------------------

    def calculateTensors(self):
        # iterate through stakes
        for i, stake in enumerate(self.rawTemplateCoordinates):
            # get list with only blobs (remove coordinates of stake)
            blobsFiltered = stake[1:]

            # lists to hold tensors
            tensorsLow = list()
            tensorsHigh = list()

            # mean tensor
            meanTensor = 0

            # get bottom tensor
            for x in range(0, 4):
                for y in range(x+1, 4):
                    # calculate tensor
                    tensorsLow.append(getTensor(blobsFiltered[x][1], blobsFiltered[y][1],
                        ((y-x) * (80+56))))

            # get median
            medianTensorLow = statistics.median(tensorsLow)
            meanTensor += medianTensorLow

            # get upper tensor
            if(len(blobsFiltered) >= 6):
                # determine number of blobs on stake
                numBlobs = len(blobsFiltered)
                if(numBlobs > 8): numBlobs = 8

                for x in range(4, numBlobs):
                    for y in range(x+1, numBlobs):
                        # calculate tensor
                        tensorsHigh.append(getTensor(blobsFiltered[x][1], blobsFiltered[y][1],
                            ((y-x) * (80+56))))

                # get median
                medianTensorHigh = statistics.median(tensorsHigh)
                meanTensor = (meanTensor + medianTensorHigh) / 2.0

            # append mean tensor to list
            self.templateTensors.append(meanTensor)

    #-----------------------------------------------------------------------
    # Function to calculate intersections for each stake
    #-----------------------------------------------------------------------

    def adjustCoords(self, x0, x1, degree, status):
        if(status == 1):
            return x0+5, x1+5
        elif(status == 2):
            return x0-5, x1-5
        else:
            return x0, x1

    def calculateIntersections(self):
        # convert image to grayscale
        imgGray = cv2.cvtColor(equalize_hist_colour(self.cv2_img.copy(), 5.0, (8,8)), cv2.COLOR_BGR2GRAY)

        # iterate through stakes
        for i, stake in enumerate(self.rawTemplateCoordinates):
            # get coordinates of top and bottom blobs on stake
            bottomBlob = [[stake[1][0][0],stake[1][0][1]],
                [stake[1][1][0],stake[1][1][1]]]
            topIndex = len(stake) - 1
            topBlob = [[stake[topIndex][0][0],stake[topIndex][0][1]],
                [stake[topIndex][1][0],stake[topIndex][1][1]]]

            # generate combinations
            coordinateCombinations = list()

            # determine middle of box
            middleBottom = ((bottomBlob[0][0] + bottomBlob[1][0]) / 2.0,
                (bottomBlob[0][1] + bottomBlob[1][1]) / 2.0)
            middleTop = ((topBlob[0][0] + topBlob[1][0]) / 2.0,
                (topBlob[0][1] + topBlob[1][1]) / 2.0)

            coordinateCombinations.append((middleTop, middleBottom)) # middle
            coordinateCombinations.append((topBlob[0], bottomBlob[0])) # Left
            coordinateCombinations.append((topBlob[1], bottomBlob[1])) # right

            # list for combination results
            combinationResults = list()

            # iterate through combinations
            for j, points in enumerate(coordinateCombinations):
                # get points
                x0, x1 = self.adjustCoords(points[0][0], points[1][0], 3, j)
                y0, y1 = self.adjustCoords(points[0][1], points[1][1], 3, j)

                # calculate line length
                num = 1000 + ((stake[1][1][1]-y1) * 4)

                # get endpoint for line
                # intersection of line between points on blob with line defining bottom of
                x1, y1 = (lineIntersections((x0,y0), (x1,y1), (stake[0][0][0],
                    stake[0][1][1]), tuple(stake[0][1])))
                y0 = points[1][1]
                x0 = points[1][0]

                # draw line on output image
                cv2.line(self.cv2_img, (int(x0),int(y0)), (int(x1), int(y1)), (255,0,0), 5)

                # make a line with "num" points
                x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

                # extract values along the line
                lineVals = ndimage.map_coordinates(np.transpose(imgGray), np.vstack((x,y)))

                # apply gaussian filter to smooth line
                lineVals_smooth = ndimage.filters.gaussian_filter1d(lineVals, 10)

                # append zero to signal to create peak
                lineVals_smooth = np.append(lineVals_smooth, 0)

                # determine peaks and properties
                peaks, properties = find_peaks(lineVals_smooth, height=100, prominence=1, width=10)

                # get sorted indexes (decreasing distance down the line)
                sorted_index = np.argsort(peaks)
                sorted_index = sorted_index[::-1]

                # index of selected peak in sorted list of peaks
                selected_peak = -1

                # iterate through peaks from bottom to top
                for index in sorted_index:
                    # only check if there is more than 1 peak remaining
                    if(index > 0):
                        # check that peak is isolated (doesn't have peak immediately next to it
                        # of similar size)
                        current_width = properties["right_ips"][index] - properties["left_ips"][index]
                        next_width = properties["right_ips"][index-1] - properties["left_ips"][index-1]

                        if(properties["left_ips"][index] - properties["right_ips"][index-1] > 50
                            or properties["peak_heights"][index-1] < properties["peak_heights"][index-1] * 0.5
                            or (current_width > (next_width*3) and index-1 == 0)):
                            selected_peak = index
                            break
                    # else select the only peak remaining
                    else:
                        # determine if this is a no snow case
                        # must see mostly snow after peak (50% coverage)
                        # snow threshold is 75% of peak
                        peak_index = peaks[index]
                        peak_intensity = lineVals[peak_index]
                        peak_range = lineVals[peak_index:]
                        snow_cover = float(len(np.where(peak_range > peak_intensity * 0.75)[0])) / float(len(peak_range)) if \
                            peak_intensity * 0.75 < 140 else float(len(np.where(peak_range > 140)[0])) / float(len(peak_range))

                        if(snow_cover > 0.5 or float(len(peak_range)) / float(len(lineVals)) < 0.15):
                            selected_peak = 0
                        else:
                            selected_peak = -1
                        break

                # if a snow peak was found
                if(selected_peak != -1):
                    # determine peak index in lineVals array
                    peak_index_line = np.uint32(peaks[selected_peak])

                    # determine threshold for finding stake
                    # average of intensity at left edge of peak and intensity at base of peak
                    left_edge_index = properties["left_ips"][selected_peak]
                    left_edge_intensity = lineVals[int(left_edge_index)]
                    left_base_index = properties["left_bases"][selected_peak]
                    left_base_intensity = lineVals[int(left_base_index)]
                    stake_threshold = (float(left_edge_intensity) - float(left_base_intensity)) / 2.0 + \
                                        float(left_base_intensity)

                    # restrict stake threshold
                    stake_threshold = 65 if stake_threshold < 65 else stake_threshold
                    stake_threshold = 115 if stake_threshold > 115 else stake_threshold

                    # determine index of intersection point
                    intersection_index = 0

                    # calculate gradients
                    line_gradients = np.gradient(lineVals.astype(np.float32))[0:peak_index_line][::-1]

                    # iterate through points prior to peak
                    for t, intensity in enumerate(reversed(lineVals[:peak_index_line])):
                        # if below threshold or large drop
                        if(intensity < stake_threshold or (line_gradients[t] > 25 and \
                            lineVals[peak_index_line-t-10] < stake_threshold+10)):
                            intersection_index = peak_index_line-t
                            break

                # add coordinates to list
                if(selected_peak != -1 and intersection_index != 0):
                    combinationResults.append((x[int(intersection_index)], y[int(intersection_index)]))

            # calculate median
            if(len(combinationResults) > 0):
                y_vals = [x[1] for x in combinationResults]
                x_vals = [x[0] for x in combinationResults]
                median_y = statistics.median(y_vals)
                median_x = statistics.median(x_vals)
                self.templateIntersections.append([median_x, median_y])

                # calculate offset for overlay
                num_blobs = len(stake)
                offset = abs(float(bottomBlob[0][0] - bottomBlob[1][0])) / num_blobs

                # update intersection distance measurements for that stake
                for g, coordinate_set in enumerate(self.templateDistances[i]):
                    distance_blob = math.hypot(median_x - coordinate_set[0], \
                        median_y - coordinate_set[1])

                    # overlay on output image
                    cv2.circle(self.cv2_img, (int(coordinate_set[0]), int(coordinate_set[1])), 5,
                        (0,255,255), 3)
                    cv2.line(self.cv2_img, (int(coordinate_set[0] + (g-(num_blobs/2.0)) * offset), int(coordinate_set[1])), \
						(int(median_x + (g-(num_blobs/2.0)) * offset), int(median_y)), (0,255,255), 2)

                    # update list with distance
                    self.templateDistances[i][g] = distance_blob

            # if no data stop template generation
            else:
                # reopen other window
                self.root.deiconify()

                # output error
                tkMessageBox.showinfo("Error", "No Intersection Point Found")
                return

    #-----------------------------------------------------------------------
    # Function to create image showing results of template generation
    #-----------------------------------------------------------------------

    def createOverlay(self):
        # overlay stake and blob boxes
        for i, stake in enumerate(self.templateCoordinates):
            for j, blob in enumerate(stake):
                if(j == 0):
                    # draw stake
                    cv2.rectangle(self.cv2_img, (int(blob[0][0]), int(blob[0][1])), (int(blob[1][0]), \
                        int(blob[1][1])),(0,0,255), 2)

                    # add tensor
                    tensor_text = str(i) + ": " + str(format(self.templateTensors[i], '.2f')) + "mm/px"
                    cv2.putText(self.cv2_img, tensor_text, (int(blob[0][0]), int(blob[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

                else:
                    # draw blob
                    cv2.rectangle(self.cv2_img, (int(blob[0][0]), int(blob[0][1])), (int(blob[1][0]), \
                        int(blob[1][1])),(0,255,0), 2)

                # output intersection points
                for point in self.templateIntersections:
                    cv2.circle(self.cv2_img, (int(point[0]), int(point[1])), 5, (0,255,0), 2)

        # create toplevel window
        self.previewWindow = tk.Toplevel(self.root)
        self.previewWindow.configure(bg = self.gray)
        self.previewWindow.protocol("WM_DELETE_WINDOW", self.closePreviewWindow)

        #-----------------------------------------------------------------------
        # Widgets
        #-----------------------------------------------------------------------

        imageFrame = tk.Frame(self.previewWindow, bg = self.gray)
        leftFrame = tk.Frame(self.previewWindow, bg = self.gray)

        titleLabel = tk.Label(leftFrame, text = "Template Generated", bg = self.gray, fg = self.white, font = self.titleFont)
        stakesLabel = tk.Label(leftFrame, text = "Stakes: %d     Blobs: %d" % (self.stakeNum, self.blobNum), bg = self.gray, fg = self.white, font = self.largeFont)
        nameLabel = tk.Label(leftFrame, text = "Template Name", bg = self.gray, fg = self.white, font = self.titleFont)

        saveButton = tk.Button(leftFrame, text = "Save", bg = self.gray, fg = self.white, font = self.smallFont,
            width = 25, command = lambda: self.saveTemplate())

        name = tk.StringVar()
        self.nameEntry = tk.Entry(leftFrame, font = self.entryFont, textvariable = name, width = 25)

        #-----------------------------------------------------------------------
        # Setup canvas
        #-----------------------------------------------------------------------

        # get screen dimensions
        screen_width = float(self.root.winfo_screenwidth())
        screen_height = float(self.root.winfo_screenheight())

        # get marked image
        marked_im = cv2.cvtColor(self.cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(marked_im)
        width, height = img.size

        ratio = min(screen_width/float(width), screen_height/float(height)) * 0.9
        imgWidth = int(float(width) * ratio)
        imgHeight = int(float(height) * ratio)
        img_resize = img.resize((imgWidth, imgHeight))
        img = ImageTk.PhotoImage(img_resize.copy())

        # create canvas to display image
        self.previewCanvas = tk.Canvas(imageFrame, width = imgWidth*0.75, height = imgHeight,
            bg = self.gray, scrollregion = (0, 0, imgWidth, imgWidth))

        self.previewCanvas.bind_all("<MouseWheel>", self._on_mousewheel_preview)

        #-----------------------------------------------------------------------
        # Packing
        #-----------------------------------------------------------------------

        # pack frames
        leftFrame.pack(padx = 50, side = tk.LEFT)
        imageFrame.pack(padx = 50, side = tk.RIGHT)

        # setup scrollbars to allow user to pan in image
        h_bar = tk.Scrollbar(imageFrame, orient = tk.HORIZONTAL, command = self.previewCanvas.xview)
        h_bar.pack(side = tk.BOTTOM, fill = tk.X)
        v_bar = tk.Scrollbar(imageFrame, orient = tk.VERTICAL, command = self.previewCanvas.yview)
        v_bar.pack(side = tk.RIGHT, fill = tk.Y)

        # display canvas
        self.previewCanvas.config(xscrollcommand = h_bar.set, yscrollcommand = v_bar.set, bg = self.gray)
        self.previewCanvas.pack(side = tk.LEFT, expand = tk.YES, fill = tk.BOTH)
        self.previewCanvas.create_image(0, 0, image = img, anchor = tk.NW)

        # pack left frame widgets
        titleLabel.pack(pady = (10,0))
        stakesLabel.pack(pady = (5,75))
        nameLabel.pack(pady = (20,10))
        self.nameEntry.pack(pady = 10)
        saveButton.pack(pady = (15,200))

        # run window
        self.root.wait_window(self.previewWindow)

    #-----------------------------------------------------------------------
    # Function to save template
    #-----------------------------------------------------------------------

    def saveTemplate(self):
        # get name
        name = self.nameEntry.get()

        # if invalid name
        if(name == ""): return

        # update system parameters
        self.systemParameters["Name"] = name

        # close preview window
        self.previewWindow.destroy()

        # close template window
        self.templateWindow.destroy()

        # update flag
        self.templateSaved = True

        # close main window
        self.root.destroy()

    #-----------------------------------------------------------------------
    # Accessor function allowing user to get template
    #-----------------------------------------------------------------------

    def getTemplate(self):
        # if a valid template was saved
        if(self.templateSaved):
            # return template coordinates, intersections, distances, tensors, sizes
            # and system parameters to user
            return [self.templateCoordinates, self.templateIntersections,
                self.templateDistances, self.templateTensors, self.blobSizeRanges,
                self.systemParameters]
        else:
            return False

    #-----------------------------------------------------------------------
    # Function to handle template window closing
    #-----------------------------------------------------------------------

    def closeTemplateWindow(self):
        # close template window and root
        self.buttonVar.set(1)
        self.windowClosed = True
        self.templateWindow.destroy()
        self.root.deiconify()

    def closePreviewWindow(self):
        self.root.deiconify()
        self.previewWindow.destroy()

root = tk.Tk()
templateWindow = createTemplate(root)
root.mainloop()
data = templateWindow.getTemplate()
print "Template Coordinates"
print data[0]
print "\n\nTemplate Intersections"
print data[1]
print "\n\nTemplate Distances"
print data[2]
print "\n\nTemplate Tensors"
print data[3]
print "\n\nBlob Ranges"
print data[4]
print "\n\n"
print data[5]
