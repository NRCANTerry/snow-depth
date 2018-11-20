# add to python path
import sys

sys.path.append('...')

# import necessary modules
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import configparser
import ast
import cv2
import os
from PIL import ImageTk, Image
from order_points import orderPoints
from get_intersection import lineIntersections
from scipy import ndimage
import statistics
from get_tensor import getTensor
import matplotlib
from matplotlib import pyplot as plt
from progress_bar import progress
from scipy.signal import find_peaks
from scipy import signal
from scipy import ndimage
import math
import template
import time
import datetime

class GUI:
    def __init__(self, master):

        # ---------------------------------------------------------------------------------
        # Create window
        # ---------------------------------------------------------------------------------

        self.root = master
        self.root.configure(background='#243447')
        self.root.title("Measure Snow Depth")
        self.root.iconbitmap(default="include/GUI/transparent.ico")

        # ---------------------------------------------------------------------------------
        # Variables
        # ---------------------------------------------------------------------------------

        # dictionary with options for program
        self.systemParameters = {
            "Directory": "",
            "Lower_HSV_1": np.array([0,0,0]),
            "Upper_HSV_1": np.array([0,0,0]),
            "Lower_HSV_2": np.array([0,0,0]),
            "Upper_HSV_2": np.array([0,0,0]),
            "Upper_Border": 0,
            "Lower_Border": 0,
            "Clip_Limit": 0,
            "Tile_Size": [0,0],
            "Saved_Colours": dict(),
            "Saved_Profiles": dict(),
            "Colour_Options": list(),
            "Profile_Options": list(),
            "Templates": dict(),
            "Templates_Options": list(),
            "Template_Paths": dict(),
            "Template_Intersections": dict(),
            "Template_Tensors": dict(),
            "Template_Blob_Sizes": dict(),
            "Template_Datasets": dict(),
            "Tensor_Datasets": dict(),
            "Blob_Distances": dict(),
            "Template_Settings": dict(),
            "Current_Template_Name": "",
            "Current_Template_Coords": list(),
            "Current_Template_Path": "",
            "Current_Template_Intersections": list(),
            "Current_Template_Tensor": list(),
            "Current_Template_Blob_Sizes": list(),
            "Current_Template_Dataset": list(),
            "Current_Tensor_Dataset": list(),
            "Current_Blob_Distances": list(),
            "Current_Template_Settings": list(),
            "Window_Closed": False,
        }

        # ConfigParser object
        self.config = configparser.ConfigParser()

        # open preferences file
        updated_parameters = self.getPreferences()
        self.systemParameters["Saved_Colours"] = updated_parameters[0]
        self.systemParameters["Saved_Profiles"] = updated_parameters[1]
        self.systemParameters["Templates"] = updated_parameters[2]
        self.systemParameters["Template_Paths"] = updated_parameters[3]
        self.systemParameters["Template_Intersections"] = updated_parameters[4]
        self.systemParameters["Template_Tensors"] = updated_parameters[5]
        self.systemParameters["Template_Blob_Sizes"] = updated_parameters[6]
        self.systemParameters["Template_Datasets"] = updated_parameters[7]
        self.systemParameters["Tensor_Datasets"] = updated_parameters[8]
        self.systemParameters["Blob_Distances"] = updated_parameters[9]
        self.systemParameters["Template_Settings"] = updated_parameters[10]
        self.systemParameters["Colour_Options"] = list(self.systemParameters["Saved_Colours"].keys())
        self.systemParameters["Profile_Options"] = list(self.systemParameters["Saved_Profiles"].keys())
        self.systemParameters["Templates_Options"] = list(self.systemParameters["Templates"].keys())

        # window closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---------------------------------------------------------------------------------
        # Labels
        # ---------------------------------------------------------------------------------

        self.gray = "#243447"
        self.white = "#ffffff"

        # main frame
        self.rootFrame = tk.Frame(self.root, bg = self.gray)

        # Step 1
        self.step1Label = tk.Label(self.rootFrame, text = "Image Folder")
        self.pathLabel = tk.Label(self.rootFrame, text = "No Directory Selected")

        # Step 2
        self.step2Label = tk.Label(self.rootFrame, text = "HSV Range")

        # H, S, V for Range 1
        self.range1Frame = tk.Frame(self.rootFrame, bg = self.gray)
        self.lowerH1 = tk.Label(self.range1Frame, text = "H")
        self.lowerS1 = tk.Label(self.range1Frame, text = "S")
        self.lowerV1 = tk.Label(self.range1Frame, text = "V")
        self.arrow1 = tk.Label(self.range1Frame, text = "-->")
        self.upperH1 = tk.Label(self.range1Frame, text = "H")
        self.upperS1 = tk.Label(self.range1Frame, text = "S")
        self.upperV1 = tk.Label(self.range1Frame, text = "V")

        # H, S, V for Range 2
        self.range2Frame = tk.Frame(self.rootFrame, bg = self.gray)
        self.lowerH2 = tk.Label(self.range2Frame, text = "H")
        self.lowerS2 = tk.Label(self.range2Frame, text = "S")
        self.lowerV2 = tk.Label(self.range2Frame, text = "V")
        self.arrow2 = tk.Label(self.range2Frame, text = "-->")
        self.upperH2 = tk.Label(self.range2Frame, text = "H")
        self.upperS2 = tk.Label(self.range2Frame, text = "S")
        self.upperV2 = tk.Label(self.range2Frame, text = "V")

        # Step 3
        self.step3Label = tk.Label(self.rootFrame, text = "Settings")

        # lists containing labels
        self.titleLabels = [self.step1Label, self.step2Label, self.step3Label]
        self.otherLabels = [self.pathLabel, self.lowerH1, self.lowerS1, self.lowerV1, self.arrow1, self.upperH1, self.upperS1, self.upperV1]
        self.grayLabels = [self.lowerH2, self.lowerS2, self.lowerV2, self.arrow2, self.upperH2, self.upperS2, self.upperV2]

        # configure title labels
        for label in self.titleLabels:
            label.config(bg = self.gray, fg = self.white, font=("Calibri Light", 28))

        # configure other labels
        for label in self.otherLabels:
            label.config(bg = self.gray, fg = self.white, font=("Calibri Light", 16))

        # configure gray labels
        for label in self.grayLabels:
            label.config(bg = self.gray, fg = "#787d84", font=("Calibri Light", 16))

        # ---------------------------------------------------------------------------------
        # Buttons
        # ---------------------------------------------------------------------------------

        # choose directory button
        self.directoryButton = tk.Button(self.rootFrame, text = "Select", command = lambda: self.selectDirectory())

        # execute button
        self.runButton = tk.Button(self.rootFrame, text = "Run", command = lambda: self.saveValues())

        # list containing buttons
        self.buttons = [self.directoryButton, self.runButton]

        # configure buttons
        for button in self.buttons:
            button.config(bg = self.gray, fg = self.white, font=("Calibri Light", 14), width = 17)

        # ---------------------------------------------------------------------------------
        # Entries
        # ---------------------------------------------------------------------------------

        validateCommand = self.root.register(self.validate)

        # Range 1
        self.entryLowerH1 = tk.Entry(self.range1Frame, validatecommand =((validateCommand, '%P', "Lower_HSV_1", 0)))
        self.entryLowerS1 = tk.Entry(self.range1Frame, validatecommand =((validateCommand, '%P', "Lower_HSV_1", 1)))
        self.entryLowerV1 = tk.Entry(self.range1Frame, validatecommand =((validateCommand, '%P', "Lower_HSV_1", 2)))
        self.entryUpperH1 = tk.Entry(self.range1Frame, validatecommand =((validateCommand, '%P', "Upper_HSV_1", 0)))
        self.entryUpperS1 = tk.Entry(self.range1Frame, validatecommand =((validateCommand, '%P', "Upper_HSV_1", 1)))
        self.entryUpperV1 = tk.Entry(self.range1Frame, validatecommand =((validateCommand, '%P', "Upper_HSV_1", 2)))

        # Range 2
        self.entryLowerH2 = tk.Entry(self.range2Frame, validatecommand =((validateCommand, '%P', "Lower_HSV_2", 0)))
        self.entryLowerS2 = tk.Entry(self.range2Frame, validatecommand =((validateCommand, '%P', "Lower_HSV_2", 1)))
        self.entryLowerV2 = tk.Entry(self.range2Frame, validatecommand =((validateCommand, '%P', "Lower_HSV_2", 2)))
        self.entryUpperH2 = tk.Entry(self.range2Frame, validatecommand =((validateCommand, '%P', "Upper_HSV_2", 0)))
        self.entryUpperS2 = tk.Entry(self.range2Frame, validatecommand =((validateCommand, '%P', "Upper_HSV_2", 1)))
        self.entryUpperV2 = tk.Entry(self.range2Frame, validatecommand =((validateCommand, '%P', "Upper_HSV_2", 2)))

        # lists of entries
        self.entries1 = [self.entryLowerH1, self.entryLowerS1, self.entryLowerV1, self.entryUpperH1, self.entryUpperS1, self.entryUpperV1]
        self.entries2 = [self.entryLowerH2, self.entryLowerS2, self.entryLowerV2, self.entryUpperH2, self.entryUpperS2, self.entryUpperV2]

        # configure entries
        for entry in self.entries1:
            entry.config(validate = "key", font=("Calibri Light", 14), width = 4)
        for entry in self.entries2:
            entry.config(validate = "key", font=("Calibri Light", 14), width = 4, state = "disabled")

        # ---------------------------------------------------------------------------------
        # Checkbox
        # ---------------------------------------------------------------------------------

        self.secondHSVFlag = tk.IntVar()
        self.checkBox = tk.Checkbutton(self.rootFrame, text="Second HSV Range", bg = self.gray, fg = self.white, selectcolor = self.gray,
            activebackground = self.gray, activeforeground = self.white, variable = self.secondHSVFlag, command = lambda:self.updateSelections(), font=("Calibri Light", 14))

        self.debug = tk.IntVar()
        self.debugCheckBox = tk.Checkbutton(self.rootFrame, text="Debug", bg = self.gray, fg = self.white, selectcolor = self.gray, activebackground = self.gray,
            activeforeground = self.white, variable = self.debug, font=("Calibri Light", 14))

        # ---------------------------------------------------------------------------------
        # Drop Down Menus
        # ---------------------------------------------------------------------------------

        self.colourMenuVar = tk.StringVar(self.root)
        self.colourMenuVar.set('Select HSV Range')
        self.colourMenu = tk.OptionMenu(self.rootFrame, self.colourMenuVar, 'Select HSV Range', *self.systemParameters["Colour_Options"])
        self.colourMenu.config(font=("Calibri Light", 14), width = 15, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.colourMenu["menu"].config(bg=self.gray, fg = self.white)
        self.colourMenuVar.trace('w', self.change_HSV_dropdown)

        self.profileMenuVar = tk.StringVar(self.root)
        self.profileMenuVar.set('Select Profile')
        self.profileMenu = tk.OptionMenu(self.rootFrame, self.profileMenuVar, 'Select Profile', *self.systemParameters["Profile_Options"])
        self.profileMenu.config(font=("Calibri Light", 14), width = 15, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.profileMenu["menu"].config(bg=self.gray, fg = self.white)
        self.profileMenuVar.trace('w', self.change_Preferences_dropdown)

        # ---------------------------------------------------------------------------------
        # Top Menu
        # ---------------------------------------------------------------------------------

        # create menu bar
        self.menubar = tk.Menu(self.root, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.filemenu = tk.Menu(self.menubar, tearoff=0, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.HSVmenu = tk.Menu(self.menubar, tearoff=0, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.prefMenu = tk.Menu(self.menubar, tearoff=0, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)

        # add commands
        self.filemenu.add_command(label = "Load Preview Tool", command = lambda: self.runPreview())
        self.filemenu.add_separator()
        self.filemenu.add_command(label = "Restart", command = lambda: self.restart())
        self.filemenu.add_command(label = "Exit", command = lambda: self.on_closing())
        self.menubar.add_cascade(label = "File", menu = self.filemenu)

        # HSV menu
        self.HSVmenu.add_command(label = "Save HSV Range", command = lambda: self.saveRanges())
        self.HSVmenu.add_command(label = "Remove HSV Range", command = lambda: self.removeRanges())
        self.menubar.add_cascade(label = "HSV Options", menu = self.HSVmenu)

        # Preferences menu
        self.prefMenu.add_command(label = "Create Profile", command = lambda: self.createProfile())
        self.prefMenu.add_command(label = "Remove Profile", command = lambda: self.removeProfile())
        self.prefMenu.add_command(label = "Preview Profile", command = lambda: self.previewProfile())
        self.menubar.add_cascade(label = "Preferences", menu = self.prefMenu)

        # configure menu bar
        self.root.config(menu = self.menubar)

        # ---------------------------------------------------------------------------------
        # Advanced Menu Widgets
        # ---------------------------------------------------------------------------------

        self.advancedFrame = tk.Frame(self.root, bg=self.gray)
        self.advancedFrameWidgets =tk.Frame(self.advancedFrame, bg=self.gray)
        self.advancedFrameOpen = False # flag indicating if advanced options are selected

        # Button to open frame
        def on_enter(e):
            self.advancedButton['bg'] = 'white'
            self.advancedButton['fg'] = self.gray
        def on_leave(e):
            time.sleep(0.10)
            self.advancedButton['bg'] = self.gray
            self.advancedButton['fg'] = 'white'

        # advanced options button
        self.advancedButton = tk.Button(self.advancedFrame, text=">", bg=self.gray, fg=self.white, borderwidth=0,
            font=("Calibri Light", 20), command=lambda: self.openAdvancedOptions(0))
        self.advancedButton.bind("<Enter>", on_enter)
        self.advancedButton.bind("<Leave>", on_leave)

        # Frames
        self.startFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)
        self.endFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)
        self.timeFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)

        # Labels
        self.dateLabel = tk.Label(self.advancedFrameWidgets, text="Date Range", bg=self.gray, fg=self.white,
            font=("Calibri Light", 28))
        self.timeLabel = tk.Label(self.timeFrame, text="Time", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.startLabel = tk.Label(self.startFrame, text="Start", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.endLabel = tk.Label(self.endFrame, text="End", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.splitLabel = tk.Label(self.timeFrame, text=":", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))

        # Time list
        self.selectedTime = [None, None]

        # Entries
        from datePicker import Datepicker
        advancedValidateCommand = self.root.register(self.validateAdvanced)
        self.startDate = Datepicker(self.startFrame, entrywidth=12)
        self.endDate = Datepicker(self.endFrame, entrywidth=12)
        self.hourTime = tk.Entry(self.timeFrame, font=("Calibri Light", 14), width=5, validate="key",
            validatecommand =((advancedValidateCommand, '%P', 0)))
        self.minuteTime = tk.Entry(self.timeFrame, font=("Calibri Light", 14), width=5, validate="key",
            validatecommand =((advancedValidateCommand, '%P', 1)))

        # Advanced menu packing
        self.dateLabel.pack(pady=10)
        self.startFrame.pack(pady=(10,5))
        self.startLabel.pack(side=tk.LEFT, padx=(15,5))
        self.startDate.pack(side=tk.LEFT, padx=5)
        self.endFrame.pack(pady=(10,5))
        self.endLabel.pack(side=tk.LEFT, padx=(20,5))
        self.endDate.pack(side=tk.LEFT, padx=(5,0))

        self.timeFrame.pack(pady=5)
        self.timeLabel.pack(side=tk.LEFT, padx=(15,5))
        self.hourTime.pack(side=tk.LEFT, padx=(5,3))
        self.splitLabel.pack(side=tk.LEFT)
        self.minuteTime.pack(side=tk.LEFT, padx=(3,5))

        # ---------------------------------------------------------------------------------
        # Packing
        # ---------------------------------------------------------------------------------

        self.advancedFrame.pack(side=tk.RIGHT, fill=tk.Y)
        self.advancedButton.pack(side=tk.RIGHT, fill=tk.Y)
        self.rootFrame.pack(side = tk.RIGHT)
        self.step1Label.pack(pady = (30,5))
        self.pathLabel.pack()
        self.directoryButton.pack(pady = 10)
        self.step2Label.pack(pady = (20,5))

        # Range 1
        self.range1Frame.pack(pady = 5)
        self.lowerH1.pack(side = tk.LEFT, padx = (20,5))
        self.entryLowerH1.pack(side = tk.LEFT, padx = 5)
        self.lowerS1.pack(side = tk.LEFT, padx = 5)
        self.entryLowerS1.pack(side = tk.LEFT, padx = 5)
        self.lowerV1.pack(side = tk.LEFT, padx = 5)
        self.entryLowerV1.pack(side = tk.LEFT, padx = (5,0))
        self.arrow1.pack(side = tk.LEFT, padx = 20)
        self.upperH1.pack(side = tk.LEFT, padx = (0,5))
        self.entryUpperH1.pack(side = tk.LEFT, padx = 5)
        self.upperS1.pack(side = tk.LEFT, padx = 5)
        self.entryUpperS1.pack(side = tk.LEFT, padx = 5)
        self.upperV1.pack(side = tk.LEFT, padx = 5)
        self.entryUpperV1.pack(side = tk.LEFT, padx = (5,20))
        self.checkBox.pack(pady = 10)

        # Range 2
        self.range2Frame.pack(pady = 5)
        self.lowerH2.pack(side = tk.LEFT, padx = (20,5))
        self.entryLowerH2.pack(side = tk.LEFT, padx = 5)
        self.lowerS2.pack(side = tk.LEFT, padx = 5)
        self.entryLowerS2.pack(side = tk.LEFT, padx = 5)
        self.lowerV2.pack(side = tk.LEFT, padx = 5)
        self.entryLowerV2.pack(side = tk.LEFT, padx = (5,0))
        self.arrow2.pack(side = tk.LEFT, padx = 20)
        self.upperH2.pack(side = tk.LEFT, padx = (0,5))
        self.entryUpperH2.pack(side = tk.LEFT, padx = 5)
        self.upperS2.pack(side = tk.LEFT, padx = 5)
        self.entryUpperS2.pack(side = tk.LEFT, padx = 5)
        self.upperV2.pack(side = tk.LEFT, padx = 5)
        self.entryUpperV2.pack(side = tk.LEFT, padx = (5,20))

        # drop down menu
        self.colourMenu.pack(pady = (25,10))

        # border frame packing
        self.step3Label.pack(pady = (20,5))
        self.profileMenu.pack(pady = 10)

        # button packing
        self.runButton.pack(pady = (20,10))
        self.debugCheckBox.pack(pady = (10,20))

    # ---------------------------------------------------------------------------------
    # Functions
    # ---------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # Validate method for text entry
    # ---------------------------------------------------------------------------------

    def validate(self, new_text, entry_field, index):
        if(index != "-1"):
            # the field is being cleared
            if not new_text:
                self.systemParameters[str(entry_field)][int(index)] = 0

            try:
                if(new_text == ""):
                    self.systemParameters[str(entry_field)][int(index)] = 0
                else:
                    self.systemParameters[str(entry_field)][int(index)] = int(new_text)
                return True

            except ValueError:
                return False
        else:
            # the field is being cleared
            if not new_text:
                self.systemParameters[str(entry_field)] = 0

            try:
                if(new_text == ""):
                    self.systemParameters[str(entry_field)] = 0
                else:
                    self.systemParameters[str(entry_field)] = int(new_text)
                return True

            except ValueError:
                return False

    # ---------------------------------------------------------------------------------
    # Validate method for time entry
    # ---------------------------------------------------------------------------------

    def validateAdvanced(self, new_text, index):
        # the field is being cleared
        if not new_text:
            self.selectedTime[int(index)] = None
        try:
            if(new_text == ""):
                self.selectedTime[int(index)] = None
            elif (int(index) == 0 and int(new_text) in range(0, 25)) \
                or (int(index) == 1 and int(new_text) in range(0,60)):
                self.selectedTime[int(index)] = int(new_text)
            else:
                return False
            return True

        except ValueError:
            return False

    # ---------------------------------------------------------------------------------
    # Function to confirm that required fields are filled in
    # ---------------------------------------------------------------------------------

    def fieldsFilled(self, directory = True):
        # flag indicating if date fields are properly filled in
        dateValid = (self.startDate.current_date is None and self.endDate.current_date is None) or \
            (self.startDate.current_date is not None and self.endDate.current_date is not None)
        dateFilled = (self.startDate.current_date is not None and self.endDate.current_date is not None)
        timeValid = (self.selectedTime[0] is not None and self.selectedTime[1] is not None)
        timeEmpty = (self.selectedTime[0] is None and self.selectedTime[1] is None)

        if(directory):
            return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
                and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
                and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
                and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
                and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1) and self.systemParameters["Directory"] != "" \
                and self.profileMenuVar.get() != "Select Profile" and
                (not self.advancedFrameOpen or (self.advancedFrameOpen and ((timeValid and dateValid) or (dateFilled and timeEmpty)))))
        else:
            return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
                and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
                and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
                and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
                and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1) and self.profileMenuVar.get() != "Select Profile"
                and (not self.advancedFrameOpen or (self.advancedFrameOpen and ((timeValid and dateValid) or (dateFilled and timeEmpty)))))

    # ---------------------------------------------------------------------------------
    # Function to confirm that required HSV fields are filled in
    # ---------------------------------------------------------------------------------

    def fieldsFilledHSV(self):
        return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
            and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
            and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
            and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
            and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1))

    # ---------------------------------------------------------------------------------
    # Function open advanced options (when arrow clicked)
    # ---------------------------------------------------------------------------------

    def openAdvancedOptions(self, status):
        if(status==0):
            # change button appearance
            self.advancedButton.config(text="<", command=lambda: self.openAdvancedOptions(1))

            # pack advanced widgets
            self.advancedFrameWidgets.pack(side=tk.RIGHT, padx=40)
            self.advancedFrameOpen = True # update flag
        else:
            # change button appearance
            self.advancedButton.config(text=">", command=lambda: self.openAdvancedOptions(0))

            # clear entries
            self.selectedTime = [None, None]
            self.hourTime.delete(0, tk.END)
            self.minuteTime.delete(0, tk.END)
            self.startDate.erase()
            self.endDate.erase()

            # unpack
            self.advancedFrameWidgets.pack_forget()
            self.advancedFrameOpen = False # update flag

    # ---------------------------------------------------------------------------------
    # Function to allow selection of directory/file where images are stored
    # ---------------------------------------------------------------------------------

    # function to check whether a file is an image
    def checkImage(self, path):
        try:
            Image.open(path)
        except IOError:
            return False
        return True

    def selectDirectory(self):
        # open directory selector
        dirname = filedialog.askdirectory(parent=self.root, initialdir="/", title='Select Directory')

        # if new directory selected
        if(len(dirname) > 0):
            # check that all files in directory are images
            files = [file for file in os.listdir(dirname)]
            valid = all(self.checkImage((dirname + "/" + str(y))) for y in files)

            # if valid update label
            if(valid):
                self.pathLabel.config(text=dirname)
                self.systemParameters["Directory"] = str(dirname)
            # else warn user
            else:
                messagebox.showinfo("Error", "Not all files in the selected directory are images")

    # ---------------------------------------------------------------------------------
    # Function to save inputted values and close window
    # ---------------------------------------------------------------------------------

    def saveValues(self):
        # if second HSV range is not selected
        if (self.secondHSVFlag.get() != 1):
            # make both ranges equal
            self.systemParameters["Lower_HSV_2"] = self.systemParameters["Lower_HSV_1"]
            self.systemParameters["Upper_HSV_2"] = self.systemParameters["Lower_HSV_2"]

        # if required fields are filled in
        if(self.fieldsFilled()):
            # close window and return to other program
            self.systemParameters["Window_Closed"] = True

            # write preferences to file
            with open('./AppData/preferences.cfg', 'w') as configfile:
                self.config.write(configfile)

            # close window
            self.root.destroy()

        # else show error
        else:
            messagebox.showinfo("Error", "Not All Fields Populated")

    # ---------------------------------------------------------------------------------
    # Accessor function to return parameters to main file
    # ---------------------------------------------------------------------------------

    def getValues(self):
        # increment end date by one
        if self.endDate.current_date is not None:
            self.endDate.current_date += datetime.timedelta(days=1)

        # return values in tuple format
        if(self.systemParameters["Window_Closed"]):
            return self.systemParameters["Directory"], self.systemParameters["Lower_HSV_1"], self.systemParameters["Upper_HSV_1"], \
                    self.systemParameters["Lower_HSV_2"], self.systemParameters["Upper_HSV_2"], self.systemParameters["Upper_Border"], \
                    self.systemParameters["Lower_Border"], self.systemParameters["Current_Template_Coords"], self.systemParameters["Current_Template_Path"], \
                    self.systemParameters["Clip_Limit"], tuple(self.systemParameters["Tile_Size"]), (self.debug.get() == 1), \
                    self.systemParameters["Current_Template_Intersections"], self.systemParameters["Current_Template_Tensor"], \
                    self.systemParameters["Current_Template_Blob_Sizes"], self.systemParameters["Current_Template_Dataset"], \
                    self.systemParameters["Current_Template_Name"], self.systemParameters["Current_Tensor_Dataset"], \
                    self.systemParameters["Current_Blob_Distances"], self.systemParameters["Current_Template_Settings"], \
                    [self.startDate.current_date, self.endDate.current_date, self.selectedTime, self.advancedFrameOpen]

        # return False if run button wasn't pressed
        else:
            return False

    # ---------------------------------------------------------------------------------
    # Function to update appearance of GUI based on status of checkbox
    # ---------------------------------------------------------------------------------

    def updateSelections(self):
        if (self.secondHSVFlag.get() == 1):
            # update labels
            for label in self.grayLabels:
                label.config(fg = self.white)
            # update fields
            for field in self.entries2:
                field.config(state = "normal")
        else:
            # update labels
            for label in self.grayLabels:
                label.config(fg ='#787d84')
            # update fields
            for field in self.entries2:
                field.delete(0, tk.END)
                field.config(state = "disabled")

    # ---------------------------------------------------------------------------------
    # Function to fetch preferences from preferences.cfg file
    # ---------------------------------------------------------------------------------

    def getPreferences(self):
        # if no preferences file present, create one
        if(str(self.config.read('./AppData/preferences.cfg')) == "[]"):
            self.config.add_section('HSV Ranges')
            self.config.add_section('Profiles')
            self.config.add_section('Template Coordinates')
            self.config.add_section('Template Intersections')
            self.config.add_section('Template Images')
            self.config.add_section('Template Tensor')
            self.config.add_section('Template Blob Sizes')
            self.config.add_section('Template Registration Dataset')
            self.config.add_section('Tensor Dataset')
            self.config.add_section('Template Blob Distances')
            self.config.add_section('Template Settings')

        # else read in existing file
        else:
            self.config.read('./AppData/preferences.cfg')

        # load in preferences
        return [dict(self.config.items('HSV Ranges')), dict(self.config.items('Profiles')),
            dict(self.config.items('Template Coordinates')), dict(self.config.items('Template Images')),
            dict(self.config.items('Template Intersections')), dict(self.config.items('Template Tensor')),
            dict(self.config.items('Template Blob Sizes')), dict(self.config.items('Template Registration Dataset')),
            dict(self.config.items('Tensor Dataset')), dict(self.config.items('Template Blob Distances')),
            dict(self.config.items('Template Settings'))]

    # ---------------------------------------------------------------------------------
    # Function that is run when user closes window
    # ---------------------------------------------------------------------------------

    def on_closing(self):
        # write preferences to file
        with open('./AppData/preferences.cfg', 'w') as configfile:
            self.config.write(configfile)

        # close window
        self.root.destroy()

    # ---------------------------------------------------------------------------------
    # Function to update HSV values on menu selection
    # ---------------------------------------------------------------------------------

    def change_HSV_dropdown(self, *args):
        # if menu selection has changed
        if(self.colourMenuVar.get() != 'Select HSV Range'):
            # create list from value stored in preferences
            hsvList = ast.literal_eval(self.systemParameters["Saved_Colours"][str(self.colourMenuVar.get())])

            # determine whether stored value is 1 range or 2
            length = len(hsvList)

            # update entries
            for field in self.entries1:
                field.delete(0, tk.END)
            for field in self.entries2:
                field.delete(0, tk.END)

            # single HSV range
            self.systemParameters["Lower_HSV_1"] = np.asarray(hsvList[0])
            self.systemParameters["Upper_HSV_1"] = np.asarray(hsvList[1])

            # update entries
            for count, field in enumerate(self.entries1):
                if(count < 3):
                    field.insert(0, self.systemParameters["Lower_HSV_1"][count])
                else:
                    field.insert(0, self.systemParameters["Upper_HSV_1"][count-3])

            # double HSV ranges
            if(length == 4):
                self.systemParameters["Lower_HSV_2"] = np.asarray(hsvList[2])
                self.systemParameters["Upper_HSV_2"] = np.asarray(hsvList[3])

                # enable second range
                if (self.secondHSVFlag.get() != 1):
                    self.secondHSVFlag.set(1)
                    self.updateSelections()

                # update entries
                for count, field in enumerate(self.entries2):
                    if(count < 3):
                        field.insert(0, self.systemParameters["Lower_HSV_2"][count])
                    else:
                        field.insert(0, self.systemParameters["Upper_HSV_2"][count-3])

            # single HSV range
            else:
                # disable second range
                if(self.secondHSVFlag.get() == 1):
                    self.secondHSVFlag.set(0)
                    self.updateSelections()

        # if default is selected, clear the fields
        else:
            # update entries
            for field in self.entries1:
                field.delete(0, tk.END)
            for field in self.entries2:
                field.delete(0, tk.END)

            # disable second range
            if(self.secondHSVFlag.get() == 1):
                self.secondHSVFlag.set(0)
                self.updateSelections()

            # update variables
            self.systemParameters["Lower_HSV_1"] = np.array([0,0,0])
            self.systemParameters["Upper_HSV_1"] = np.array([0,0,0])
            self.systemParameters["Lower_HSV_2"] = np.array([0,0,0])
            self.systemParameters["Upper_HSV_2"] = np.array([0,0,0])

    # ---------------------------------------------------------------------------------
    # Function to update preferences based on drop down
    # ---------------------------------------------------------------------------------

    def change_Preferences_dropdown(self, *args):
        # if menu selection has changed
        if(self.profileMenuVar.get() != 'Select Profile'):
            # create list from value stored in preferences
            optionsList = ast.literal_eval(self.systemParameters["Saved_Profiles"][str(self.profileMenuVar.get())])

            # update system parameters
            self.systemParameters["Upper_Border"] = int(optionsList[0])
            self.systemParameters["Lower_Border"] = int(optionsList[1])
            self.systemParameters["Current_Template_Name"] = str(optionsList[2])
            self.systemParameters["Current_Template_Coords"] = ast.literal_eval(self.systemParameters["Templates"][str(optionsList[2])])
            self.systemParameters["Current_Template_Path"] = self.systemParameters["Template_Paths"][str(optionsList[2])]
            self.systemParameters["Current_Template_Intersections"] = ast.literal_eval(self.systemParameters["Template_Intersections"][str(optionsList[2])])
            self.systemParameters["Current_Template_Tensor"] = ast.literal_eval(self.systemParameters["Template_Tensors"][str(optionsList[2])])
            self.systemParameters["Current_Template_Blob_Sizes"] = ast.literal_eval(self.systemParameters["Template_Blob_Sizes"][str(optionsList[2])])
            self.systemParameters["Current_Template_Dataset"] = ast.literal_eval(self.systemParameters["Template_Datasets"][str(optionsList[2])])
            self.systemParameters["Current_Tensor_Dataset"] = ast.literal_eval(self.systemParameters["Tensor_Datasets"][str(optionsList[2])])
            self.systemParameters["Current_Blob_Distances"] = ast.literal_eval(self.systemParameters["Blob_Distances"][str(optionsList[2])])
            self.systemParameters["Current_Template_Settings"] = ast.literal_eval(self.systemParameters["Template_Settings"][str(optionsList[2])])
            self.systemParameters["Clip_Limit"] = int(optionsList[3])
            self.systemParameters["Tile_Size"] = [int(optionsList[4]), int(optionsList[5])]

    # ---------------------------------------------------------------------------------
    # Function to restart script to load changes
    # ---------------------------------------------------------------------------------

    def restart(self):
        # write preferences to file
        with open('./AppData/preferences.cfg', 'w') as configfile:
            self.config.write(configfile)

        # close window
        self.root.destroy()

        # restart program
        os.execv(sys.executable, ['C:\\Users\\tbaricia\\AppData\\Local\\Continuum\\miniconda2\\python.exe'] + sys.argv)

    # ---------------------------------------------------------------------------------
    # Function to allow user to save HSV ranges to preferences file
    # ---------------------------------------------------------------------------------

    def saveRanges(self):
        # function to close window
        def closing(status):
            # if name was inputted
            if(name.get() != "" and status == True):
                # create output string
                outputString = "[" + np.array2string(self.systemParameters["Lower_HSV_1"], separator = ',').replace("[","(").replace("]", ")") + "," + \
                    np.array2string(self.systemParameters["Upper_HSV_1"], separator = ',').replace("[","(").replace("]", ")")

                if(self.secondHSVFlag.get() == 1):
                    outputString += "," + np.array2string(self.systemParameters["Lower_HSV_2"], separator = ',').replace("[","(").replace("]", ")")
                    outputString += "," + np.array2string(self.systemParameters["Upper_HSV_2"], separator = ',').replace("[","(").replace("]", ")")

                outputString += "]"

                # add to config file
                self.config.set('HSV Ranges', name.get(), outputString)

                # update menu
                self.colourMenu['menu'].add_command(label = name.get(), command = tk._setit(self.colourMenuVar, name.get()))
                self.systemParameters["Saved_Colours"][name.get()] = outputString
                self.systemParameters["Colour_Options"].append(name.get())
                self.colourMenuVar.set(name.get())

            # unpack frame
            leftFrame.pack_forget()
            separator.pack_forget()

        # only run if all fields filled in
        if(not self.fieldsFilledHSV()):
            messagebox.showinfo("Error", "Not All HSV Fields Populated")
            return

        # create new frame
        leftFrame = tk.Frame(self.root, bg = self.gray)

        # Labels
        nameLabel = tk.Label(leftFrame, text = "Name", bg = self.gray, fg = self.white, font = ("Calibri Light", 20))

        # Entry
        name = tk.StringVar()
        nameEntry = tk.Entry(leftFrame, font=("Calibri Light", 14),textvariable = name, width = 20)

        # buttons
        executeButton = tk.Button(leftFrame, text = "Save", bg = self.gray, fg = self.white, command = lambda: closing(True),
            width = 20, font = ("Calibri Light", 14))
        exitButton = tk.Button(leftFrame, text = "x", bg = self.gray, fg = self.white, command = lambda: closing(False),
            width = 2, height = 1, font = ("Calibri Light", 14))

        # separator
        separator_height = self.root.winfo_reqheight() * 0.85
        separator = tk.Frame(self.root, relief = tk.SUNKEN, width = 1, height = int(separator_height), bg = self.white)

        # packing
        leftFrame.pack(side = tk.LEFT, padx = 25)
        separator.pack(side = tk.LEFT, fill = tk.X)
        exitButton.pack(padx = (200,0), pady = (0,150))
        nameLabel.pack(pady = (20,5))
        nameEntry.pack(pady = 10)
        executeButton.pack(pady = (5,300))

    # ---------------------------------------------------------------------------------
    # Function to remove an HSV range from the preferences file
    # ---------------------------------------------------------------------------------

    def removeRanges(self):
        # function to close window
        def closing(status):
            # remove selected option from menu
            if(removecolourMenuVar.get() != 'Select Profile to Remove' and status == True):
                self.config.remove_option("HSV Ranges", removecolourMenuVar.get())
                self.systemParameters["Colour_Options"].remove(removecolourMenuVar.get())
                self.systemParameters["Saved_Colours"].pop(removecolourMenuVar.get())

                # update menu
                self.colourMenuVar.set('Select HSV Range')
                self.colourMenu['menu'].delete(0, 'end')

                for option in self.systemParameters["Colour_Options"]:
                    self.colourMenu['menu'].add_command(label = option, command = tk._setit(self.colourMenuVar, option))

            # unpack frame
            leftFrame.pack_forget()
            separator.pack_forget()

        # create new frame
        leftFrame = tk.Frame(self.root, bg = self.gray)

        # Labels
        nameLabel = tk.Label(leftFrame, text = "Name", bg = self.gray, fg = self.white, font = ("Calibri Light", 20))

        # drop down menu used to select profile to remove
        removecolourMenuVar = tk.StringVar(self.root)
        removecolourMenuVar.set('Select Profile to Remove')
        removeMenu = tk.OptionMenu(leftFrame, removecolourMenuVar, 'Select Profile to Remove', *self.systemParameters["Colour_Options"])
        removeMenu.config(bg = self.gray, fg = self.white, activebackground = self.gray, activeforeground = self.white)
        removeMenu.config(font=("Calibri Light", 14))
        removeMenu["menu"].config(bg = self.gray, fg = self.white)

        # buttons
        executeButton = tk.Button(leftFrame, text = "Save", bg = self.gray, fg = self.white, command = lambda: closing(True),
            width = 20, font = ("Calibri Light", 14))
        exitButton = tk.Button(leftFrame, text = "x", bg = self.gray, fg = self.white, command = lambda: closing(False),
            width = 2, height = 1, font = ("Calibri Light", 14))

        # separator
        separator_height = self.root.winfo_reqheight() * 0.85
        separator = tk.Frame(self.root, relief = tk.SUNKEN, width = 1, height = int(separator_height), bg = self.white)

        # packing
        leftFrame.pack(side = tk.LEFT, padx = 25)
        separator.pack(side = tk.LEFT, fill = tk.X)
        exitButton.pack(padx = (200,0), pady = (0,150))
        nameLabel.pack(pady = (20,5))
        removeMenu.pack(pady = 10)
        executeButton.pack(pady = (5,300))

    # ---------------------------------------------------------------------------------
    # Function to create a preferences profile
    # ---------------------------------------------------------------------------------

    def createProfile(self):
        # embedded function to allow for removal of template
        def removeTemplate():
            # function to close window
            def closing(status):
                # remove selected option from menu
                if(removetemplateMenuVar.get() != 'Select Template to Remove' and status == True):
                    # delete saved template image
                    path = self.systemParameters["Template_Paths"][removetemplateMenuVar.get()]
                    filename, ext = os.path.splitext(os.path.split(path)[1])
                    path_marked = os.path.split(path)[0] + "\\" + filename + "-marked" + ext
                    os.remove(path)
                    os.remove(path_marked)

                    self.config.remove_option("Template Coordinates", removetemplateMenuVar.get())
                    self.config.remove_option("Template Images", removetemplateMenuVar.get())
                    self.config.remove_option("Template Intersections", removetemplateMenuVar.get())
                    self.config.remove_option("Template Tensor", removetemplateMenuVar.get())
                    self.config.remove_option("Template Blob Sizes", removetemplateMenuVar.get())
                    self.config.remove_option("Template Registration Dataset", removetemplateMenuVar.get())
                    self.config.remove_option('Tensor Dataset', removetemplateMenuVar.get())
                    self.config.remove_option('Template Blob Distances', removetemplateMenuVar.get())
                    self.config.remove_option('Template Settings', removetemplateMenuVar.get())
                    self.systemParameters["Templates"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Templates_Options"].remove(removetemplateMenuVar.get())
                    self.systemParameters["Template_Paths"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Template_Intersections"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Template_Tensors"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Template_Blob_Sizes"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Template_Datasets"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Tensor_Datasets"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Blob_Distances"].pop(removetemplateMenuVar.get())
                    self.systemParameters["Template_Settings"].pop(removetemplateMenuVar.get())

                    # update menu
                    templateMenuVar.set('Select Template')
                    templateMenu['menu'].delete(0, 'end')

                    for option in self.systemParameters["Templates_Options"]:
                        templateMenu['menu'].add_command(label = option, command = tk._setit(templateMenuVar, option))

                # unpack frame
                leftFrame.pack_forget()
                separator.pack_forget()

            # create new frame
            leftFrame = tk.Frame(newWindow, bg = self.gray)

            # Labels
            nameLabel = tk.Label(leftFrame, text = "Name", bg = self.gray, fg = self.white, font = ("Calibri Light", 20))

            # drop down menu used to select profile to remove
            removetemplateMenuVar = tk.StringVar(self.root)
            removetemplateMenuVar.set('Select Profile to Remove')
            removeTemplateMenu = tk.OptionMenu(leftFrame, removetemplateMenuVar, 'Select Profile to Remove', *self.systemParameters["Templates_Options"])
            removeTemplateMenu.config(bg = self.gray, fg = self.white, activebackground = self.gray, activeforeground = self.white)
            removeTemplateMenu.config(font=("Calibri Light", 14))
            removeTemplateMenu["menu"].config(bg = self.gray, fg = self.white)

            # buttons
            executeButton = tk.Button(leftFrame, text = "Save", bg = self.gray, fg = self.white, command = lambda: closing(True),
                width = 20, font = ("Calibri Light", 14))
            exitButton = tk.Button(leftFrame, text = "x", bg = self.gray, fg = self.white, command = lambda: closing(False),
                width = 2, height = 1, font = ("Calibri Light", 14))

            # separator
            separator_height = newWindow.winfo_reqheight() * 0.85
            separator = tk.Frame(newWindow, relief = tk.SUNKEN, width = 1, height = int(separator_height), bg = self.white)

            # packing
            leftFrame.pack(side = tk.LEFT, padx = 25)
            separator.pack(side = tk.LEFT, fill = tk.X)
            exitButton.pack(padx = (200,0), pady = (0,60))
            nameLabel.pack(pady = (20,5))
            removeTemplateMenu.pack(pady = 10)
            executeButton.pack(pady = (5,185))

        # embedded function to allow for creation of template
        def createTemplate():
            # hide other windows
            newWindow.withdraw()
            self.root.withdraw()

            # create top level window
            templateWindow = tk.Toplevel(self.root)

            # open template gui
            tmpl = template.createTemplate(templateWindow)

            # run window
            self.root.wait_window(templateWindow)

            # get template parameters
            params = tmpl.getTemplate()

            # if template wasn't created exit
            if(params == False):
                # reopen other windows
                self.root.deiconify()
                newWindow.deiconify()
                return

            # open params
            templateCoords = params[0]
            templateIntersections = params[1]
            templateDistances = params[2]
            templateTensors = params[3]
            templateBlobSizes = params[4]
            templateSystemParams = params[5]
            templateImage = params[6]
            templateMarked = params[7]
            templatePath = params[8]
            templateName = templateSystemParams["Name"]

            # search for template directory
            if(not os.path.isdir("./AppData/templates")):
                os.mkdir("./AppData/templates")
            if(not os.path.isdir("./AppData/models")):
                os.mkdir("./AppData/models")
            if(not os.path.isdir("./AppData/training")):
                os.mkdir("./AppData/training")

            # write images to directory
            current_dir = os.getcwd()
            filename, ext = os.path.splitext(os.path.split(templatePath)[1])
            template_path = current_dir + "\\AppData\\templates\\" + filename + "-" +  str(templateName) + ext
            marked_template_path = current_dir + "\\AppData\\templates\\" + filename + "-" +  str(templateName) + "-marked" + ext
            cv2.imwrite(template_path, templateImage)
            cv2.imwrite(marked_template_path, templateMarked)

            # create folder for training images
            os.mkdir("./AppData/training/%s" % filename)

            # create output strings
            outputString = str(templateCoords).replace("array(", "").replace(")", "")
            intersectionString = str(templateIntersections).replace("array(", "").replace(")", "")
            tensorString = str(templateTensors).replace("array(", "").replace(")", "")
            blobSizeString = str(templateBlobSizes).replace("array(", "").replace(")", "")
            templateDataString = str([[0,0,0],[]]).replace("array(", "").replace(")", "")
            tensorDataArray = [[[0,0,0], []]] * len(templateTensors)
            tensorDataString = str(tensorDataArray).replace("array(", "").replace(")", "")
            intersectionDistanceString = str(templateDistances).replace("array(", "").replace(")", "")
            settingsList = [templateSystemParams["Register_STD_DEV"], templateSystemParams["Tensor_STD_DEV"],
                templateSystemParams["Rotation"], templateSystemParams["Translation"], templateSystemParams["Scale"]]
            settingsString = str(settingsList).replace("array(", "").replace(")", "")

            # save to config file
            self.config.set('Template Coordinates',templateName, outputString)
            self.config.set('Template Images', templateName, template_path)
            self.config.set('Template Intersections', templateName, intersectionString)
            self.config.set('Template Tensor', templateName, tensorString)
            self.config.set('Template Blob Sizes', templateName, blobSizeString)
            self.config.set('Template Registration Dataset', templateName, templateDataString)
            self.config.set('Tensor Dataset', templateName, tensorDataString)
            self.config.set('Template Blob Distances', templateName, intersectionDistanceString)
            self.config.set('Template Settings', templateName, settingsString)

            # update menu
            templateMenu['menu'].add_command(label = templateName, command = tk._setit(templateMenuVar, templateName))
            self.systemParameters["Templates"][templateName] = outputString
            self.systemParameters["Templates_Options"].append(templateName)
            self.systemParameters["Template_Paths"][templateName] = template_path
            self.systemParameters["Current_Template_Name"] = templateName
            self.systemParameters["Current_Template_Coords"] = templateCoords
            self.systemParameters["Current_Template_Path"] = template_path
            self.systemParameters["Template_Intersections"][templateName] = intersectionString
            self.systemParameters["Template_Tensors"][templateName] = tensorString
            self.systemParameters["Template_Blob_Sizes"][templateName] = blobSizeString
            self.systemParameters["Template_Datasets"][templateName] = templateDataString
            self.systemParameters["Tensor_Datasets"][templateName] = tensorDataString
            self.systemParameters["Blob_Distances"][templateName] = intersectionDistanceString
            self.systemParameters["Template_Settings"][templateName] = settingsString
            self.systemParameters["Current_Template_Intersections"] = templateIntersections
            self.systemParameters["Current_Template_Tensor"] = templateTensors
            self.systemParameters["Current_Template_Blob_Sizes"] = templateBlobSizes
            self.systemParameters["Current_Template_Dataset"] = [[0,0,0],[]]
            self.systemParameters["Current_Tensor_Dataset"] = tensorDataArray
            self.systemParameters["Current_Blob_Distances"] = templateDistances
            self.systemParameters["Current_Template_Settings"] = settingsList
            templateMenuVar.set(templateName)

            print("Template Saved Successfully: %s \n" % templateName)

            # reopen other windows
            newWindow.deiconify()
            self.root.deiconify()

        # embedded function to get name for saved profile
        def getName():
            # function to close window
            def closing(status):
                # if name was inputted
                if(name.get() != "" and status == True):
                    # create output string
                    outputString = "[" + str(self.systemParameters["Upper_Border"]) + "," + str(self.systemParameters["Lower_Border"]) + "," + \
                        '"' + str(templateMenuVar.get()) + '"' + "," + str(self.systemParameters["Clip_Limit"]) + "," + str(self.systemParameters["Tile_Size"][0]) + \
                        "," + str(self.systemParameters["Tile_Size"][1]) + "]"

                    # add to config file
                    self.config.set('Profiles', name.get(), outputString)

                    # update menu
                    self.profileMenu['menu'].add_command(label = name.get(), command = tk._setit(self.profileMenuVar, name.get()))
                    self.systemParameters["Saved_Profiles"][name.get()] = outputString
                    self.systemParameters["Profile_Options"].append(name.get())
                    self.profileMenuVar.set(name.get())

                # unpack frame
                leftFrame.pack_forget()
                separator.pack_forget()

                # close window
                newWindow.destroy()

            # get fields are filled in
            if not all(v.get() != "" for v in entries) and templateMenuVar.get() != "Select Template":
                messagebox.showinfo("Error", "Not All Fields Populated")
                return

            # create new frame
            leftFrame = tk.Frame(newWindow, bg = self.gray)

            # Labels
            nameLabel = tk.Label(leftFrame, text = "Name", bg = self.gray, fg = self.white, font = ("Calibri Light", 20))

            # Entry
            name = tk.StringVar()
            nameEntry = tk.Entry(leftFrame, font=("Calibri Light", 14),textvariable = name, width = 20)

            # buttons
            executeButton = tk.Button(leftFrame, text = "Save", bg = self.gray, fg = self.white, command = lambda: closing(True),
                width = 20, font = ("Calibri Light", 14))
            exitButton = tk.Button(leftFrame, text = "x", bg = self.gray, fg = self.white, command = lambda: closing(False),
                width = 2, height = 1, font = ("Calibri Light", 14))

            # separator
            separator_height = newWindow.winfo_reqheight() * 0.85
            separator = tk.Frame(newWindow, relief = tk.SUNKEN, width = 1, height = int(separator_height), bg = self.white)

            # packing
            leftFrame.pack(side = tk.LEFT, padx = 25)
            separator.pack(side = tk.LEFT, fill = tk.X)
            exitButton.pack(padx = (200,0), pady = (0,60))
            nameLabel.pack(pady = (20,5))
            nameEntry.pack(pady = 10)
            executeButton.pack(pady = (5,185))

        # create window
        newWindow = tk.Toplevel(self.root)
        newWindow.configure(bg=self.gray)

        validateCommand = self.root.register(self.validate)

        # frames
        preferencesFrame = tk.Frame(newWindow, bg = self.gray)
        upperBorderFrame = tk.Frame(preferencesFrame, bg = self.gray)
        lowerBorderFrame = tk.Frame(preferencesFrame, bg = self.gray)
        templateFrame = tk.Frame(preferencesFrame, bg = self.gray)
        clipLimitFrame = tk.Frame(preferencesFrame, bg = self.gray)
        tileSizeFrame = tk.Frame(preferencesFrame, bg = self.gray)
        buttonFrame = tk.Frame(preferencesFrame, bg = self.gray)

        # labels
        titleLabel = tk.Label(preferencesFrame, text = "Preferences", bg = self.gray, fg = self.white, font=("Calibri Light", 24))
        upperBorderLabel = tk.Label(upperBorderFrame, text = "Upper Border")
        lowerBorderLabel = tk.Label(lowerBorderFrame, text = "Lower Border")
        templateLabel = tk.Label(templateFrame, text = "Template")
        clipLimitLabel = tk.Label(clipLimitFrame, text = "Clip Limit")
        tileSizeLabel1 = tk.Label(tileSizeFrame, text = "Tile Size")
        tileSizeLabel2 = tk.Label(tileSizeFrame, text = "(")
        tileSizeLabel3 = tk.Label(tileSizeFrame, text = ",")
        tileSizeLabel4 = tk.Label(tileSizeFrame, text = ")")

        labels = [upperBorderLabel, lowerBorderLabel, templateLabel, clipLimitLabel, tileSizeLabel1, tileSizeLabel2, tileSizeLabel3, tileSizeLabel4]

        for label in labels:
            label.config(bg = self.gray, fg = self.white, font=("Calibri Light", 16))

        # entries
        upperBorderEntry = tk.Entry(upperBorderFrame, validatecommand =((validateCommand, '%P', "Upper_Border", -1)))
        lowerBorderEntry = tk.Entry(lowerBorderFrame, validatecommand =((validateCommand, '%P', "Lower_Border", -1)))
        clipLimitEntry = tk.Entry(clipLimitFrame, validatecommand =((validateCommand, '%P', "Clip_Limit", -1)))
        tileSizeEntry1 = tk.Entry(tileSizeFrame, validatecommand =((validateCommand, '%P', "Tile_Size", 0)))
        tileSizeEntry2 = tk.Entry(tileSizeFrame, validatecommand =((validateCommand, '%P', "Tile_Size", 1)))

        # set default values
        clipLimitEntry.insert(0,5)
        tileSizeEntry1.insert(0,8)
        tileSizeEntry2.insert(0,8)
        self.systemParameters["Clip_Limit"] = 5
        self.systemParameters["Tile_Size"] = [8,8]

        entries = [upperBorderEntry, lowerBorderEntry, clipLimitEntry, tileSizeEntry1, tileSizeEntry2]

        for entry in entries:
            entry.config(validate = "key", font=("Calibri Light", 14), width = 6)

        # drop down
        templateMenuVar = tk.StringVar(preferencesFrame)
        templateMenuVar.set('Select Template')
        templateMenu = tk.OptionMenu(templateFrame, templateMenuVar, 'Select Template', *self.systemParameters["Templates_Options"])
        templateMenu.config(font=("Calibri Light", 14), bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white, width = 15)
        templateMenu["menu"].config(bg = self.gray, fg = self.white)

        # menu bar
        profileMenuBar = tk.Menu(newWindow, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        profileFileMenu = tk.Menu(profileMenuBar, tearoff=0, bg = self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        profileFileMenu.add_command(label = "Generate Template", command = lambda: createTemplate())
        profileFileMenu.add_command(label = "Remove Template", command = lambda: removeTemplate())
        profileMenuBar.add_cascade(label = "File", menu = profileFileMenu)
        newWindow.config(menu = profileMenuBar)

        # button
        createProfileButton = tk.Button(buttonFrame, text = "Create Profile", command = lambda: getName(), bg = self.gray,
            fg = self.white, font=("Calibri Light", 15), width = 17)

        # packing
        preferencesFrame.pack(side = tk.RIGHT, pady = (10,20), padx = 50)
        titleLabel.pack(pady = 20)
        upperBorderFrame.pack(pady = 10)
        upperBorderLabel.pack(side = tk.LEFT, padx = 10)
        upperBorderEntry.pack(side = tk.LEFT, padx = 10)

        lowerBorderFrame.pack(pady = 15)
        lowerBorderLabel.pack(side = tk.LEFT, padx = 10)
        lowerBorderEntry.pack(side = tk.LEFT, padx = 10)

        templateFrame.pack(pady = 15)
        templateLabel.pack(side = tk.LEFT, padx = 10)
        templateMenu.pack(side = tk.LEFT, padx = 10)

        clipLimitFrame.pack(pady = 15)
        clipLimitLabel.pack(side = tk.LEFT, padx = 10)
        clipLimitEntry.pack(side = tk.LEFT, padx = 10)

        tileSizeFrame.pack(pady = 15)
        tileSizeLabel1.pack(side = tk.LEFT, padx = 10)
        tileSizeLabel2.pack(side = tk.LEFT, padx = (5,0))
        tileSizeEntry1.pack(side = tk.LEFT)
        tileSizeLabel3.pack(side = tk.LEFT, padx = 0)
        tileSizeEntry2.pack(side = tk.LEFT)
        tileSizeLabel4.pack(side = tk.LEFT, padx = (0,5))

        buttonFrame.pack(pady = 20)
        createProfileButton.pack(side = tk.LEFT, padx = (5, 20))

        # wait until user inputs name
        self.root.wait_window(newWindow)

    # ---------------------------------------------------------------------------------
    # Function to remove a preferences profile
    # ---------------------------------------------------------------------------------

    def removeProfile(self):
        # function to close window
        def closing(status):
            # remove selected option from menu
            if(removeprofileMenuVar.get() != 'Select Profile to Remove' and status == True):
                self.config.remove_option("Profiles", removeprofileMenuVar.get())
                self.systemParameters["Profile_Options"].remove(removeprofileMenuVar.get())
                self.systemParameters["Saved_Profiles"].pop(removeprofileMenuVar.get())

                # update menu
                self.profileMenuVar.set('Select Profile')
                self.profileMenu['menu'].delete(0, 'end')

                for option in self.systemParameters["Profile_Options"]:
                    self.profileMenu['menu'].add_command(label = option, command = tk._setit(self.profileMenuVar, option))

            # unpack frame
            leftFrame.pack_forget()
            separator.pack_forget()

        # create new frame
        leftFrame = tk.Frame(self.root, bg = self.gray)

        # Labels
        nameLabel = tk.Label(leftFrame, text = "Name", bg = self.gray, fg = self.white, font = ("Calibri Light", 20))

        # drop down menu used to select profile to remove
        removeprofileMenuVar = tk.StringVar(self.root)
        removeprofileMenuVar.set('Select Profile to Remove')
        removeMenu = tk.OptionMenu(leftFrame, removeprofileMenuVar, 'Select Profile to Remove', *self.systemParameters["Profile_Options"])
        removeMenu.config(bg = self.gray, fg = self.white, activebackground = self.gray, activeforeground = self.white)
        removeMenu.config(font=("Calibri Light", 14))
        removeMenu["menu"].config(bg = self.gray, fg = self.white)

        # buttons
        executeButton = tk.Button(leftFrame, text = "Save", bg = self.gray, fg = self.white, command = lambda: closing(True),
            width = 20, font = ("Calibri Light", 14))
        exitButton = tk.Button(leftFrame, text = "x", bg = self.gray, fg = self.white, command = lambda: closing(False),
            width = 2, height = 1, font = ("Calibri Light", 14))

        # separator
        separator_height = self.root.winfo_reqheight() * 0.85
        separator = tk.Frame(self.root, relief = tk.SUNKEN, width = 1, height = int(separator_height), bg = self.white)

        # packing
        leftFrame.pack(side = tk.LEFT, padx = 25)
        separator.pack(side = tk.LEFT, fill = tk.X)
        exitButton.pack(padx = (200,0), pady = (0,150))
        nameLabel.pack(pady = (20,5))
        removeMenu.pack(pady = 10)
        executeButton.pack(pady = (5,300))

    # ---------------------------------------------------------------------------------
    # Function to preview preferences profile
    # ---------------------------------------------------------------------------------

    def previewProfile(self):
        if(self.profileMenuVar.get() != 'Select Profile'):
            # create window
            newWindow = tk.Toplevel(self.root)
            newWindow.configure(bg=self.gray)

            # frames
            leftFrame = tk.Frame(newWindow, bg = self.gray)
            rightFrame = tk.Frame(newWindow, bg = self.gray)

            # labels
            titleLabel = tk.Label(leftFrame, text = str(self.profileMenuVar.get()), bg = self.gray, fg = self.white, font=("Calibri Light", 24))
            upperBorderLabel = tk.Label(leftFrame, text = "Upper Border: " + str(self.systemParameters["Upper_Border"]))
            lowerBorderLabel = tk.Label(leftFrame, text = "Lower Border: " + str(self.systemParameters["Lower_Border"]))
            templateLabel = tk.Label(leftFrame, text = "Template: " + str(self.systemParameters["Current_Template_Name"]))
            clipLimitLabel = tk.Label(leftFrame, text = "Clip Limit: " + str(self.systemParameters["Clip_Limit"]))
            tileSizeLabel = tk.Label(leftFrame, text = "Tile Size: " + str(self.systemParameters["Tile_Size"]))

            labels = [upperBorderLabel, lowerBorderLabel, templateLabel, clipLimitLabel, tileSizeLabel]

            for label in labels:
                label.config(bg = self.gray, fg = self.white, font=("Calibri Light", 16))

            # canvas
            screen_width = float(self.root.winfo_screenwidth())
            screen_height = float(self.root.winfo_screenheight())

            # import image
            filename, extension = os.path.splitext(self.systemParameters["Current_Template_Path"])
            markedPath = filename + "-marked" + extension
            im = Image.open(markedPath)
            width, height = im.size

            # determine height and width for canvas
            ratio = min(screen_width/float(width), screen_height/float(height)) * 0.8
            imgWidth = int(float(width) * ratio)
            imgHeight = int(float(height) * ratio)

            # resize image and place in canvas
            im = im.resize((imgWidth, imgHeight))
            im = ImageTk.PhotoImage(im)
            previewCanvas = tk.Canvas(rightFrame, width = imgWidth * 0.8, height = imgHeight,
                bg = self.gray, scrollregion = (0, 0, imgWidth, imgHeight))

            # packing
            leftFrame.pack(side = tk.LEFT, padx = 40)
            titleLabel.pack(pady = 20)
            upperBorderLabel.pack(pady = 10)
            lowerBorderLabel.pack(pady = 10)
            templateLabel.pack(pady = 10)
            clipLimitLabel.pack(pady = 10)
            tileSizeLabel.pack(pady = 10)

            # canvas packing
            rightFrame.pack(side = tk.RIGHT)

            # setup scrollbars
            h_bar = tk.Scrollbar(rightFrame, orient = tk.HORIZONTAL, command = previewCanvas.xview)
            h_bar.pack(side = tk.BOTTOM, fill = tk.X)
            v_bar = tk.Scrollbar(rightFrame, orient = tk.VERTICAL, command = previewCanvas.yview)
            v_bar.pack(side = tk.RIGHT, fill = tk.Y)
            previewCanvas.config(xscrollcommand = h_bar.set, yscrollcommand = v_bar.set)

            previewCanvas.pack(side = tk.LEFT, expand = tk.YES, fill = tk.BOTH)
            previewCanvas.create_image(0, 0, image = im, anchor = tk.NW)

            # wait until user closes window
            self.root.wait_window(newWindow)
        else:
            messagebox.showinfo("Error", "Please Select a Profile under Settings to Preview")


    # ---------------------------------------------------------------------------------
    # Function to run HSV range preview
    # ---------------------------------------------------------------------------------

    def runPreview(self):
        # embedded function to handle window closing
        def closeWindow():
            cv2.destroyAllWindows()
            newWindow.destroy()

        # embedded function to update HSV mask on slider movement
        def updateValues(event):
            # get slider positions
            h1 = H1Slider.get()
            h2 = H2Slider.get()
            s1 = S1Slider.get()
            s2 = S2Slider.get()
            v1 = V1Slider.get()
            v2 = V2Slider.get()

            # if second HSV range selected
            if(previewsecondHSVFlag.get() == 1):
                h3 = H3Slider.get()
                h4 = H4Slider.get()
                s3 = S3Slider.get()
                s4 = S4Slider.get()
                v3 = V3Slider.get()
                v4 = V4Slider.get()

            # convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # apply HSV mask
            mask = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
            if(previewsecondHSVFlag.get() == 1):
                mask2 = cv2.inRange(hsv, np.array([h3, s3, v3]), np.array([h4, s4, v4]))
                mask = cv2.bitwise_or(mask, mask2)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # create horizontal stack
            numpy_horizontal = np.hstack((img, mask))

            # display image to user
            cv2.imshow("Comparison", numpy_horizontal)

        # embedded function to save values to main GUI window
        def saveValuestoMain():
            # update entries
            for field in self.entries1:
                field.delete(0, tk.END)

            # if second range selected
            if(previewsecondHSVFlag.get() == 1):
                self.secondHSVFlag.set(1)
                self.updateSelections()
                for field in self.entries2:
                    field.delete(0, tk.END)

            # update variables
            self.systemParameters["Lower_HSV_1"] = np.array([H1Slider.get(), S1Slider.get(), V1Slider.get()])
            self.systemParameters["Upper_HSV_1"] = np.array([H2Slider.get(), S2Slider.get(), V2Slider.get()])

            # update entries
            for count, field in enumerate(self.entries1):
                if(count < 3):
                    field.insert(0, self.systemParameters["Lower_HSV_1"][count])
                else:
                    field.insert(0, self.systemParameters["Upper_HSV_1"][count-3])

            # second range
            if(previewsecondHSVFlag.get() == 1):
                self.systemParameters["Lower_HSV_2"] = np.array([H3Slider.get(), S3Slider.get(), V3Slider.get()])
                self.systemParameters["Upper_HSV_2"] = np.array([H4Slider.get(), S4Slider.get(), V4Slider.get()])

                # update entries
                for count, field in enumerate(self.entries2):
                    if(count < 3):
                        field.insert(0, self.systemParameters["Lower_HSV_2"][count])
                    else:
                        field.insert(0, self.systemParameters["Upper_HSV_2"][count-3])

            # disable second range if required
            if(self.secondHSVFlag.get() == 1 and previewsecondHSVFlag.get() != 1):
                self.secondHSVFlag.set(0)
                self.updateSelections()

            # close windows
            newWindow.destroy()
            cv2.destroyAllWindows()

        # embedded function to toggle second HSV range on and off
        def togglesecondHSVFlag():
            # if checkbox selected
            if(previewsecondHSVFlag.get() == 1):
                # resize window
                newWindow.geometry("900x650")

                # pack second HSV widgets
                for widget in secondRangeWidgets:
                    widget.pack(side = tk.LEFT, padx =(15,5))

            # if checkbox not selected
            else:
                # unpack second HSV widgets
                for widget in secondRangeWidgets:
                    widget.pack_forget()

                # resize window
                newWindow.geometry("500x650")

        # flag to track if second HSV range is in use
        secondRangePreview = False

        # open new window
        newWindow = tk.Toplevel(self.root)
        newWindow.geometry("500x650")
        newWindow.configure(background=self.gray)
        newWindow.withdraw()

        # configure window closing protocol
        newWindow.protocol("WM_DELETE_WINDOW", closeWindow)

        # get filename
        filename = filedialog.askopenfilename(initialdir = "/",title = "Select image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

        # if valid file selected
        if(filename != ""):
            newWindow.deiconify()
            #newWindow.geometry("500x650") # window size of 500 x 650

            # create widgets
            HSVSlidersLabel = tk.Label(newWindow,text="HSV Sliders",bg=self.gray,fg=self.white,font=("Calibri Light", 24))
            LowerLabel = tk.Label(newWindow,text="Lower",bg=self.gray,fg=self.white,font=("Calibri Light", 15))
            UpperLabel = tk.Label(newWindow,text="Upper",bg=self.gray,fg=self.white,font=("Calibri Light", 15))

            # button
            savetoMain = tk.Button(newWindow, text = "Save Values",bg = self.gray,fg = self.white,command = lambda: saveValuestoMain(),
                width = 20,font=("Calibri Light", 15))

            # frames
            Frame1 = tk.Frame(newWindow, background=self.gray)
            Frame2 = tk.Frame(newWindow, background=self.gray)
            Frame3 = tk.Frame(newWindow, background=self.gray)
            Frame4 = tk.Frame(newWindow, background=self.gray)
            Frame5 = tk.Frame(newWindow, background=self.gray)
            Frame6 = tk.Frame(newWindow, background=self.gray)

            # H, S, and V Labels
            H1Lower = tk.Label(Frame1, text = "H", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            S1Lower = tk.Label(Frame2, text = "S", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            V1Lower = tk.Label(Frame3, text = "V", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            H1Upper = tk.Label(Frame4, text = "H", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            S1Upper = tk.Label(Frame5, text = "S", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            V1Upper = tk.Label(Frame6, text = "V", background= self.gray, foreground= self.white, font=("Calibri Light", 14))

            # sliders
            H1Slider = tk.Scale(Frame1, from_=0, to=180, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            S1Slider = tk.Scale(Frame2, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            V1Slider = tk.Scale(Frame3, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            H2Slider = tk.Scale(Frame4, from_=0, to=180, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            S2Slider = tk.Scale(Frame5, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            V2Slider = tk.Scale(Frame6, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)

            # second range H, S, and V Labels
            H2Lower = tk.Label(Frame1, text = "H", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            S2Lower = tk.Label(Frame2, text = "S", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            V2Lower = tk.Label(Frame3, text = "V", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            H2Upper = tk.Label(Frame4, text = "H", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            S2Upper = tk.Label(Frame5, text = "S", background= self.gray, foreground= self.white, font=("Calibri Light", 14))
            V2Upper = tk.Label(Frame6, text = "V", background= self.gray, foreground= self.white, font=("Calibri Light", 14))

            # second range sliders
            H3Slider = tk.Scale(Frame1, from_=0, to=180, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            S3Slider = tk.Scale(Frame2, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            V3Slider = tk.Scale(Frame3, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            H4Slider = tk.Scale(Frame4, from_=0, to=180, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            S4Slider = tk.Scale(Frame5, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)
            V4Slider = tk.Scale(Frame6, from_=0, to=255, orient = 'horizontal', background= self.gray, activebackground=self.gray, foreground=self.white, highlightbackground=self.gray,
                highlightcolor = self.white, length = 350, font = ("Calibri Light", 14), command = updateValues)

            # list containing widgets for second HSV range
            secondRangeWidgets = [H2Lower, S2Lower, V2Lower, H2Upper, S2Upper, V2Upper, H3Slider, S3Slider, V3Slider, \
                H4Slider, S4Slider, V4Slider]

            # checkbox
            previewsecondHSVFlag = tk.IntVar()
            previewCheckBox = tk.Checkbutton(
                newWindow,
                text="Second HSV Range",
                variable= previewsecondHSVFlag,
                background=self.gray,
                foreground=self.white,
                selectcolor=self.gray,
                activebackground=self.gray,
                activeforeground=self.white,
                command=lambda: togglesecondHSVFlag(),
                font=("Calibri Light", 14))

            # packing
            HSVSlidersLabel.pack(pady = (20,5))
            LowerLabel.pack(pady = (5,0))
            Frame1.pack(pady = 2)
            H1Lower.pack(side = tk.LEFT, padx = 5)
            H1Slider.pack(side = tk.LEFT, padx = 5)
            Frame2.pack(pady = 2)
            S1Lower.pack(side = tk.LEFT, padx = 5)
            S1Slider.pack(side = tk.LEFT, padx = 5)
            Frame3.pack(pady = 2)
            V1Lower.pack(side = tk.LEFT, padx = 5)
            V1Slider.pack(side = tk.LEFT, padx = 5)
            UpperLabel.pack(pady = (10,0))
            Frame4.pack(pady = 2)
            H1Upper.pack(side = tk.LEFT, padx = 5)
            H2Slider.pack(side = tk.LEFT, padx = 5)
            Frame5.pack(pady = 2)
            S1Upper.pack(side = tk.LEFT, padx = 5)
            S2Slider.pack(side = tk.LEFT, padx = 5)
            Frame6.pack(pady = 2)
            V1Upper.pack(side = tk.LEFT, padx = 5)
            V2Slider.pack(side = tk.LEFT, padx = 5)
            previewCheckBox.pack(pady = (20, 10))
            savetoMain.pack(pady = (10,25))

            # open comparison window
            cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

            # open image
            img = cv2.imread(filename)

            # resize image to 1/4 of original size
            img = cv2.resize(img, (0,0), None, 0.25, 0.25)

            # wait until user closes window
            self.root.wait_window(newWindow)
