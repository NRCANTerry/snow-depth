# add to python path
import sys

sys.path.append('...')

# import necessary modules
import tkinter as tk
from tkinter import font
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import configparser
import ast
import cv2
import os
from PIL import ImageTk, Image
import template
import time
import datetime

# class to hold preferences for each helper function
class Preferences:
    # init function
    def __init__(self, **kwargs):
        # create dictionary with given preferences
        self.__dict__.update(kwargs)

class GUI:
    def __init__(self, master):

        #===========================================================================
        # Create window
        #===========================================================================

        self.root = master
        self.root.configure(background='#282c34')
        self.root.title("Measure Snow Depth")
        self.root.iconbitmap(default="include/GUI/transparent.ico")
        self.root.resizable(False, False)


        # create AppData directory if it doesn't exist
        if(not os.path.isdir("AppData")):
            os.mkdir("./AppData")

        #===========================================================================
        # Variables
        #===========================================================================

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
            "Reg_Params": [0, 0, 0, 0, 0],
            "Int_Params": [0, 0, 0, 0 , 0],
            "Misc_Params": [0, 0]
        }

        # dictionary with options for program
        #self.systemParameters = dict()

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

        #===========================================================================
        # Labels
        #===========================================================================

        self.gray = "#282c34"
        self.white = "#f1f1f1"

        # main frame
        self.rootFrame = tk.Frame(self.root, bg=self.gray)

        # Step 1
        self.step1Label = tk.Label(self.rootFrame, text = "Image Folder")
        self.pathLabel = tk.Label(self.rootFrame, text = "No Directory Selected")

        # Step 2
        self.step2Label = tk.Label(self.rootFrame, text = "HSV Range")

        # H, S, V for Range 1
        self.range1Frame = tk.Frame(self.rootFrame, bg=self.gray)
        self.lowerH1 = tk.Label(self.range1Frame, text = "H")
        self.lowerS1 = tk.Label(self.range1Frame, text = "S")
        self.lowerV1 = tk.Label(self.range1Frame, text = "V")
        self.arrow1 = tk.Label(self.range1Frame, text = "-->")
        self.upperH1 = tk.Label(self.range1Frame, text = "H")
        self.upperS1 = tk.Label(self.range1Frame, text = "S")
        self.upperV1 = tk.Label(self.range1Frame, text = "V")

        # H, S, V for Range 2
        self.range2Frame = tk.Frame(self.rootFrame, bg=self.gray)
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
            label.config(bg=self.gray, fg = self.white, font=("Calibri", 30))

        # configure other labels
        for label in self.otherLabels:
            label.config(bg=self.gray, fg = self.white, font=("Calibri Light", 16))

        # configure gray labels
        for label in self.grayLabels:
            label.config(bg=self.gray, fg = "#787d84", font=("Calibri Light", 16))

        #===========================================================================
        # Buttons
        #===========================================================================

        # choose directory button
        self.directoryButton = tk.Button(self.rootFrame, text = "Select", command = lambda: self.selectDirectory())
        createHoverLabel(self.directoryButton, "Select the folder containing the images to be processed")

        # execute button
        self.runButton = tk.Button(self.rootFrame, text = "Run", command = lambda: self.saveValues())

        # list containing buttons
        self.buttons = [self.directoryButton, self.runButton]

        # configure buttons
        for button in self.buttons:
            button.config(bg=self.gray, fg = self.white, font=("Calibri Light", 14), width = 17)

        #===========================================================================
        # Entries
        #===========================================================================

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

        #===========================================================================
        # Checkbox
        #===========================================================================

        self.secondHSVFlag = tk.IntVar()
        self.checkBox = tk.Checkbutton(self.rootFrame, text="Second HSV Range", bg=self.gray, fg = self.white, selectcolor = self.gray,
            activebackground = self.gray, activeforeground = self.white, variable = self.secondHSVFlag, command = lambda:self.updateSelections(), font=("Calibri Light", 14))
        createHoverLabel(self.checkBox, "Add a second range to the HSV filter")


        self.debug = tk.IntVar()
        self.debugCheckBox = tk.Checkbutton(self.rootFrame, text="Debug", bg=self.gray, fg = self.white, selectcolor = self.gray, activebackground = self.gray,
            activeforeground = self.white, variable = self.debug, font=("Calibri Light", 14))
        createHoverLabel(self.debugCheckBox, "Debug mode outputs images at each step of the algorithm")

        #===========================================================================
        # Drop Down Menus
        #===========================================================================

        self.colourMenuVar = tk.StringVar(self.root)
        self.colourMenuVar.set('Select HSV Range')
        self.colourMenu = tk.OptionMenu(self.rootFrame, self.colourMenuVar, 'Select HSV Range', *self.systemParameters["Colour_Options"])
        self.colourMenu.config(font=("Calibri Light", 14), width = 15, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.colourMenu["menu"].config(bg=self.gray, fg = self.white)
        self.colourMenuVar.trace('w', self.change_HSV_dropdown)

        self.profileMenuVar = tk.StringVar(self.root)
        self.profileMenuVar.set('Select Profile')
        self.profileMenu = tk.OptionMenu(self.rootFrame, self.profileMenuVar, 'Select Profile', *self.systemParameters["Profile_Options"])
        self.profileMenu.config(font=("Calibri Light", 14), width = 15, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.profileMenu["menu"].config(bg=self.gray, fg = self.white)
        self.profileMenuVar.trace('w', self.change_Preferences_dropdown)

        #===========================================================================
        # Top Menu
        #===========================================================================

        # create menu bar
        self.menubar = tk.Menu(self.root, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.filemenu = tk.Menu(self.menubar, tearoff=0, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.HSVmenu = tk.Menu(self.menubar, tearoff=0, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.prefMenu = tk.Menu(self.menubar, tearoff=0, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.tempMenu = tk.Menu(self.menubar, tearoff=0, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)
        self.DLMenu = tk.Menu(self.menubar, tearoff=0, bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white)

        # add commands
        self.filemenu.add_command(label = "Load Preview Tool", command = lambda: self.runPreview())
        self.filemenu.add_separator()
        self.filemenu.add_command(label = "Restart", command = lambda: self.restart())
        self.filemenu.add_command(label = "Exit", command = lambda: self.on_closing())
        self.menubar.add_cascade(label = "File", menu = self.filemenu)

        # HSV menu
        self.HSVmenu.add_command(label = "Save HSV Range", command = lambda: self.addHSVRange())
        self.HSVmenu.add_command(label = "Remove HSV Range", command = lambda: self.removeSavedHSVRange())
        self.menubar.add_cascade(label = "Colours", menu = self.HSVmenu)

        # Preferences menu
        self.prefMenu.add_command(label = "Create Profile", command = lambda: self.createProfile())
        self.prefMenu.add_command(label = "Remove Profile", command = lambda: self.removeProfile())
        self.prefMenu.add_command(label = "Preview Profile", command = lambda: self.previewProfile())
        self.menubar.add_cascade(label = "Profiles", menu = self.prefMenu)

        # Template Menu
        self.tempMenu.add_command(label="Create Template", command=lambda: self.addTemplate())
        self.tempMenu.add_command(label="Remove Template", command=lambda: self.removeTemplate())
        self.menubar.add_cascade(label="Templates", menu=self.tempMenu)

        # Deep Learning menu
        self.DLMenu.add_command(label = "Reset Blob Neural Network", command = lambda: self.resetBlobNet())
        self.menubar.add_cascade(label = "Deep Learning", menu=self.DLMenu)

        # configure menu bar
        self.root.config(menu = self.menubar)

        #===========================================================================
        # Advanced Menu Widgets
        #===========================================================================

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
            font=("Calibri Light", 20), command=lambda: self.toggleAdvancedOptions(0))
        self.advancedButton.bind("<Enter>", on_enter)
        self.advancedButton.bind("<Leave>", on_leave)

        # Frames
        self.date1DateFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)
        self.date2DateFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)
        self.date1TimeFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)
        self.date2TimeFrame = tk.Frame(self.advancedFrameWidgets, bg=self.gray)

        # Labels
        self.dateLabel = tk.Label(self.advancedFrameWidgets, text="Date Range", bg=self.gray, fg=self.white,
            font=("Calibri Light", 28))
        dateFont = font.Font(family="Calibri Light", size=20, underline=True)
        self.date1Label = tk.Label(self.advancedFrameWidgets, text="From", bg=self.gray, fg=self.white,
            font=dateFont)
        self.date2Label = tk.Label(self.advancedFrameWidgets, text="To", bg=self.gray, fg=self.white,
            font=dateFont)
        self.date1DateLabel = tk.Label(self.date1DateFrame, text="Date", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.date2DateLabel = tk.Label(self.date2DateFrame, text="Date", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.date1TimeLabel = tk.Label(self.date1TimeFrame, text="Time", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.date2TimeLabel = tk.Label(self.date2TimeFrame, text="Time", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.date1SplitLabel = tk.Label(self.date1TimeFrame, text=":", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))
        self.date2SplitLabel = tk.Label(self.date2TimeFrame, text=":", bg=self.gray, fg=self.white,
            font=("Calibri Light", 18))

        # Time list
        self.selectedTime = [None, None, None, None]

        # Entries
        from datePicker import Datepicker
        advancedValidateCommand = self.root.register(self.validateAdvanced)
        self.startDate = Datepicker(self.date1DateFrame, entrywidth=12)
        self.endDate = Datepicker(self.date2DateFrame, entrywidth=12)
        self.startHourTime = tk.Entry(self.date1TimeFrame, font=("Calibri Light", 14), width=5, validate="key",
            validatecommand =((advancedValidateCommand, '%P', 0)))
        self.endHourTime = tk.Entry(self.date2TimeFrame, font=("Calibri Light", 14), width=5, validate="key",
            validatecommand =((advancedValidateCommand, '%P', 2)))
        self.startMinuteTime = tk.Entry(self.date1TimeFrame, font=("Calibri Light", 14), width=5, validate="key",
            validatecommand =((advancedValidateCommand, '%P', 1)))
        self.endMinuteTime = tk.Entry(self.date2TimeFrame, font=("Calibri Light", 14), width=5, validate="key",
            validatecommand =((advancedValidateCommand, '%P', 3)))

        # Advanced menu packing
        self.dateLabel.pack(pady=10)
        self.date1Label.pack(pady=(5,0), anchor=tk.W, padx=(18,0))
        self.date1DateFrame.pack(pady=(10,5))
        self.date1DateLabel.pack(side=tk.LEFT, padx=(15,5))
        self.startDate.pack(side=tk.LEFT, padx=5)
        self.date1TimeFrame.pack(pady=5)
        self.date1TimeLabel.pack(side=tk.LEFT, padx=(15,5))
        self.startHourTime.pack(side=tk.LEFT, padx=(5,3))
        self.date1SplitLabel.pack(side=tk.LEFT)
        self.startMinuteTime.pack(side=tk.LEFT, padx=(3,5))

        self.date2Label.pack(pady=(15,0), anchor=tk.W, padx=(18,0))
        self.date2DateFrame.pack(pady=(10,5))
        self.date2DateLabel.pack(side=tk.LEFT, padx=(15,5))
        self.endDate.pack(side=tk.LEFT, padx=5)
        self.date2TimeFrame.pack(pady=(5, 40))
        self.date2TimeLabel.pack(side=tk.LEFT, padx=(15,5))
        self.endHourTime.pack(side=tk.LEFT, padx=(5,3))
        self.date2SplitLabel.pack(side=tk.LEFT)
        self.endMinuteTime.pack(side=tk.LEFT, padx=(3,5))

        #===========================================================================
        # Packing
        #===========================================================================

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

        self.kwargs = {
            "bg": self.gray,
            "fg": self.white
        }

        self.menuKwargs = {
            "bg": self.gray,
            "fg": self.white,
            "activebackground": self.gray,
            "activeforeground": self.white
        }

        self.profileOpen = False

    #===========================================================================
    # Functions
    #===========================================================================

    #===========================================================================
    # Validate method for text entry
    #===========================================================================

    def validate(self, new_text, entry_field, index, divisor=1.0):
        if(index != "-1"):
            # the field is being cleared
            if not new_text:
                self.systemParameters[str(entry_field)][int(index)] = 0

            try:
                if(new_text == ""):
                    self.systemParameters[str(entry_field)][int(index)] = 0
                else:
                    self.systemParameters[str(entry_field)][int(index)] = int(new_text) / float(divisor)
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

    #===========================================================================
    # Validate method for time entry
    #===========================================================================

    def validateAdvanced(self, new_text, index):
        # the field is being cleared
        if not new_text:
            self.selectedTime[int(index)] = None
        try:
            if(new_text == ""):
                self.selectedTime[int(index)] = None
            elif (int(index) % 2 == 0 and int(new_text) in range(0, 25)) \
                or (int(index) % 2 != 0 and int(new_text) in range(0,60)):
                self.selectedTime[int(index)] = int(new_text)
            else:
                return False
            return True

        except ValueError:
            return False

    #===========================================================================
    # Function to confirm that required fields are filled in
    #===========================================================================

    def fieldsFilled(self, directory = True):
        # flag indicating if date fields are properly filled in
        dateValid = (self.startDate.current_date is None and self.endDate.current_date is None) or \
            (self.startDate.current_date is not None and self.endDate.current_date is not None)
        dateFilled = (self.startDate.current_date is not None and self.endDate.current_date is not None)
        timeValid = (self.selectedTime[0] is not None and self.selectedTime[1] is not None and
            self.selectedTime[2] is not None and self.selectedTime[3] is not None)
        timeEmpty = (self.selectedTime[0] is None and self.selectedTime[1] is None and
            self.selectedTime[2] is None and self.selectedTime[3] is None)

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

    #===========================================================================
    # Function to confirm that required HSV fields are filled in
    #===========================================================================

    def fieldsFilledHSV(self):
        return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
            and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
            and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
            and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
            and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1))

    #===========================================================================
    # Function to toggle advanced options window (when arrow clicked)
    #===========================================================================

    def toggleAdvancedOptions(self, status):
        if(status==0):
            # change button appearance
            self.advancedButton.config(text="<", command=lambda: self.toggleAdvancedOptions(1))

            # pack advanced widgets
            self.advancedFrameWidgets.pack(side=tk.RIGHT, padx=40)
            self.advancedFrameOpen = True # update flag
        else:
            # change button appearance
            self.advancedButton.config(text=">", command=lambda: self.toggleAdvancedOptions(0))

            # clear entries
            self.selectedTime = [None, None, None, None]
            self.startHourTime.delete(0, tk.END)
            self.startMinuteTime.delete(0, tk.END)
            self.endHourTime.delete(0, tk.END)
            self.endMinuteTime.delete(0, tk.END)
            self.startDate.erase()
            self.endDate.erase()

            # unpack
            self.advancedFrameWidgets.pack_forget()
            self.advancedFrameOpen = False # update flag

    #===========================================================================
    # Function to allow selection of directory/file where images are stored
    #===========================================================================

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
                displayName = (dirname[len(dirname)-55:]) if len(dirname) > 55 else dirname
                if(displayName is not dirname and displayName[0] is not "/"):
                    displayName = "..." + displayName[displayName.find("/"):]

                self.pathLabel.config(text=displayName)
                self.systemParameters["Directory"] = str(dirname)
            # else warn user
            else:
                messagebox.showinfo("Error", "Not all files in the selected directory are images")

    #===========================================================================
    # Function to save inputted values and close window
    #===========================================================================

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

    #===========================================================================
    # Accessor function to return parameters to main file
    #===========================================================================

    def getValues(self):
        # increment end date by one
        if self.endDate.current_date is not None and all(v is None for v in self.selectedTime):
            self.endDate.current_date += datetime.timedelta(days=1)

        # return values in tuple format
        if("Window_Closed" in self.systemParameters.keys() and self.systemParameters["Window_Closed"]):
            return self.systemParameters["Directory"], self.systemParameters["Lower_HSV_1"], self.systemParameters["Upper_HSV_1"], \
                    self.systemParameters["Lower_HSV_2"], self.systemParameters["Upper_HSV_2"], self.systemParameters["Upper_Border"], \
                    self.systemParameters["Lower_Border"], self.systemParameters["Current_Template_Coords"], self.systemParameters["Current_Template_Path"], \
                    self.systemParameters["Clip_Limit"], tuple(self.systemParameters["Tile_Size"]), (self.debug.get() == 1), \
                    self.systemParameters["Current_Template_Intersections"], self.systemParameters["Current_Template_Tensor"], \
                    self.systemParameters["Current_Template_Blob_Sizes"], self.systemParameters["Current_Template_Dataset"], \
                    self.systemParameters["Current_Template_Name"], self.systemParameters["Current_Tensor_Dataset"], \
                    self.systemParameters["Current_Blob_Distances"], self.systemParameters["Current_Template_Settings"], \
                    [self.startDate.current_date, self.endDate.current_date, self.selectedTime, self.advancedFrameOpen], \
                    self.systemParameters["Reg_Params"], self.systemParameters["Int_Params"], self.systemParameters["Misc_Params"]

        # return False if run button wasn't pressed
        else:
            return False

    #===========================================================================
    # Function to update appearance of GUI based on status of checkbox
    #===========================================================================

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

    #===========================================================================
    # Function to fetch preferences from preferences.cfg file
    #===========================================================================

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

    #===========================================================================
    # Function that is run when user closes window
    #===========================================================================

    def on_closing(self):
        # write preferences to file
        with open('./AppData/preferences.cfg', 'w') as configfile:
            self.config.write(configfile)

        # close window
        self.root.destroy()

    #===========================================================================
    # Function to update HSV values on menu selection
    #===========================================================================

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

    #===========================================================================
    # Function to update preferences based on drop down
    #===========================================================================

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
            self.systemParameters["Reg_Params"] = [int(optionsList[6]), float(optionsList[7]), float(optionsList[8]), float(optionsList[9]),
                float(optionsList[10]), int(optionsList[11])]
            self.systemParameters["Int_Params"] = [int(optionsList[12]), float(optionsList[13]), float(optionsList[14]), float(optionsList[15]),
                float(optionsList[16])]
            self.systemParameters["Misc_Params"] = [int(optionsList[17]), int(optionsList[18])]

    #===========================================================================
    # Function to restart script to load changes
    #===========================================================================

    def restart(self):
        # write preferences to file
        with open('./AppData/preferences.cfg', 'w') as configfile:
            self.config.write(configfile)

        # close window
        self.root.destroy()

        # restart program
        os.execv(sys.executable, ['C:\\Users\\tbaricia\\AppData\\Local\\Continuum\\miniconda2\\python.exe'] + sys.argv)

    #===========================================================================
    # Function to provide widgets to build a save page
    #===========================================================================

    def getSavePageWidgets(self, master, pack=0):
        def on_enter(e):
            exit['bg'] = 'white'
            exit['fg'] = self.gray
        def on_leave(e):
            time.sleep(0.10)
            exit['bg'] = self.gray
            exit['fg'] = 'white'

        # create widgets
        frame = tk.Frame(master, bg=self.gray)
        label = tk.Label(frame, text="Name", font=("Calibri Light", 20), **self.kwargs)
        var = tk.StringVar(master)
        entry = tk.Entry(frame, font=("Calibri Light", 14), textvariable=var, width=20)

        # buttons
        execute = tk.Button(frame, text="Save", **self.kwargs, width=20, font=("Calibri Light", 14))
        exit = tk.Button(frame, text="x", **self.kwargs, width=2, height=1,
            font=("Calibri Light", 16), borderwidth=0)
        exit.bind("<Enter>", on_enter)
        exit.bind("<Leave>", on_leave)

        # separator
        separator = tk.Frame(master, relief=tk.SUNKEN, width=1,
            height=int(master.winfo_reqheight() * 0.85), bg=self.white)

        # packing
        if(pack == 0):
            frame.pack(side=tk.LEFT, padx=25)
            separator.pack(side=tk.LEFT, fill=tk.X)
            exit.pack(padx=(200,0), pady=(0,150))
            label.pack(pady=(20,5))
            entry.pack(pady=10)
            execute.pack(pady=(5,300))
        else: # profile window packing
            frame.pack(side=tk.LEFT, padx=25)
            separator.pack(side=tk.LEFT, fill=tk.X)
            exit.pack(padx=(200,0), pady=(0,60))
            label.pack(pady=(20,5))
            entry.pack(pady=10)
            execute.pack(pady=(5,185))

        # return widgets
        return execute, exit, var, frame, separator


    #===========================================================================
    # Function to add an HSV Range to the preferences file
    #===========================================================================

    def addHSVRange(self):
        def format(array): # format numpy arrays as string
            return np.array2string(array, separator = ',').replace("[","(").replace("]", ")")

        def close(status): # function to handle closing
            if(var.get() != "" and status): # name was given
                # create output string
                output = "[" + format(self.systemParameters["Lower_HSV_1"]) + "," + \
                    format(self.systemParameters["Upper_HSV_1"])

                if(self.secondHSVFlag.get()):
                    output += "," + format(self.systemParameters["Lower_HSV_2"]) + "," + \
                        format(self.systemParameters["Upper_HSV_2"])
                output += "]"

                # add to config file
                self.config.set('HSV Ranges', var.get(), output)
                self.colourMenu['menu'].add_command(label=var.get(), command=tk._setit(self.colourMenuVar, var.get()))
                self.systemParameters["Saved_Colours"][var.get()] = output
                self.systemParameters["Colour_Options"].append(var.get())
                self.colourMenuVar.set(var.get())

            # unpack frames
            frame.pack_forget()
            separator.pack_forget()

        # only run if all fields filled in
        if(not self.fieldsFilledHSV()):
            messagebox.showinfo("Error", "Not All HSV Fields Populated")
            return

        # get widgets
        execute, exit, var, frame, separator = self.getSavePageWidgets(self.root)
        execute.config(command=lambda: close(True))
        exit.config(command=lambda: close(False))

    #===========================================================================
    # Function to provide widgets to build a remove page
    #===========================================================================

    def getRemovePageWidgets(self, master, text, options):
        def on_enter(e):
            exit['bg'] = 'white'
            exit['fg'] = self.gray
        def on_leave(e):
            time.sleep(0.10)
            exit['bg'] = self.gray
            exit['fg'] = 'white'

        # create widgets
        frame = tk.Frame(master, bg=self.gray)
        label = tk.Label(frame, text="Name", font=("Calibri Light", 20), **self.kwargs)
        var = tk.StringVar(master)
        var.set(text)
        removeMenu = tk.OptionMenu(frame, var, text, *options)
        removeMenu.config(**self.menuKwargs, font=("Calibri Light", 14))
        removeMenu["menu"].config(**self.kwargs)

        # buttons
        execute = tk.Button(frame, text="Save", **self.kwargs, width=20, font=("Calibri Light", 14))
        exit = tk.Button(frame, text="x", **self.kwargs, width=2, height=1,
            font=("Calibri Light", 16), borderwidth=0)
        exit.bind("<Enter>", on_enter)
        exit.bind("<Leave>", on_leave)

        # separator
        separator = tk.Frame(master, relief=tk.SUNKEN, width=1,
            height=int(master.winfo_reqheight() * 0.85), bg=self.white)

        # packing
        frame.pack(side=tk.LEFT, padx=25)
        separator.pack(side=tk.LEFT, fill=tk.X)
        exit.pack(padx=(200,0), pady=(0,150))
        label.pack(pady=(20,5))
        removeMenu.pack(pady=10)
        execute.pack(pady=(5,300))

        # return widgets
        return execute, exit, var, frame, separator


    #===========================================================================
    # Function to remove an HSV range from the preferences file
    #===========================================================================

    def removeSavedHSVRange(self):
        # function to handle closing
        def close(status):
            if(var.get() != 'Select Profile to Remove' and status):
                self.config.remove_option("HSV Ranges", var.get())
                self.systemParameters["Colour_Options"].remove(var.get())
                self.systemParameters["Saved_Colours"].pop(var.get())

                # update menu
                self.colourMenuVar.set('Select HSV Range')
                self.colourMenu['menu'].delete(0, 'end')

                for option in self.systemParameters["Colour_Options"]:
                    self.colourMenu['menu'].add_command(label=option,
                        command=tk._setit(self.colourMenuVar, option))

            # unpack frames
            frame.pack_forget()
            separator.pack_forget()

        # get widgets
        execute, exit, var, frame, separator = self.getRemovePageWidgets(self.root,
            'Select Profile to Remove', self.systemParameters["Colour_Options"])
        execute.config(command=lambda: close(True))
        exit.config(command=lambda: close(False))


    #===========================================================================
    # Function to reset blob neural network
    #===========================================================================

    def resetBlobNet(self):
        # function to handle closing
        def close(status):
            if(var.get() != 'Select Model to Remove' and status):
                # determine if model exists
                from pathlib import Path
                tempPath = (self.systemParameters["Template_Paths"][str(var.get())])
                filename = os.path.splitext(os.path.split(tempPath)[1])[0]
                modelPath = str(Path(tempPath).parents[1]) + "\\models\\" + filename + ".model"

                # model exists
                if(os.path.isfile(modelPath)):
                    # delete model
                    os.remove(modelPath)

                    # re-add training directories
                    training_path = str(Path(tempPath).parents[1]) + "\\training\\" + filename + "/"
                    if(not os.path.isdir(training_path)):
                        os.mkdir(training_path)
                    if(not os.path.isdir(training_path + "\\blob")):
                        os.mkdir(training_path + "\\blob")
                    if(not os.path.isdir(training_path + "\\non-blob")):
                        os.mkdir(training_path + "\\non-blob")

            # unpack frames
            frame.pack_forget()
            separator.pack_forget()

        # get widgets
        execute, exit, var, frame, separator = self.getRemovePageWidgets(self.root,
            'Select Model to Remove', self.systemParameters["Templates_Options"])
        execute.config(command=lambda: close(True))
        exit.config(command=lambda: close(False))

    #===========================================================================
    # Function to add a template
    #===========================================================================

    def addTemplate(self):
        def format(array): # format array for preferences file
            return str(array).replace("array(", "").replace(")", "")

        # hide main window
        self.root.withdraw()

        # create top level window
        templateWindow = tk.Toplevel(self.root)
        templateWindow.resizable(False, False)

        # open template GUI
        tmpl = template.createTemplate(templateWindow)
        self.root.wait_window(templateWindow)
        params = tmpl.getTemplate()

        # if template wasn't created exit
        if(params == False):
            self.root.deiconify()
            return

        # unpack parameters
        templateCoords, templateIntersections, templateDistances, templateTensors, \
            templateBlobSizes, templateSystemParams, templateImage, templateMarked, \
            templatePath = params
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
        template_path = current_dir + "\\AppData\\templates\\" + filename + "-" + \
            str(templateName) + ext
        marked_template_path = current_dir + "\\AppData\\templates\\" + filename + "-" + \
            str(templateName) + "-marked" + ext
        cv2.imwrite(template_path, templateImage)
        cv2.imwrite(marked_template_path, templateMarked)

        # create folder for training images
        training_path = "./AppData/training/" + filename + "-" +  str(templateName) + "/"
        os.mkdir(training_path)
        os.mkdir(training_path + "/blob")
        os.mkdir(training_path + "/non-blob")

        # create output strings
        outputString = format(templateCoords)
        intersectionString = format(templateIntersections)
        tensorString = format(templateTensors)
        blobSizeString = format(templateBlobSizes)
        templateDataString = format([[0,0,0],[]])
        tensorDataArray = [[[0,0,0], []]] * len(templateTensors)
        tensorDataString = format(tensorDataArray)
        intersectionDistanceString = format(templateDistances)
        settingsList = [templateSystemParams["Register_STD_DEV"], templateSystemParams["Tensor_STD_DEV"],
            templateSystemParams["Rotation"], templateSystemParams["Translation"], templateSystemParams["Scale"]]
        settingsString = format(settingsList)

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

        # update menu if possible
        if(self.profileOpen):
            self.templateMenu['menu'].add_command(label=templateName,
                command=tk._setit(self.templateMenuVar, templateName))
            self.templateMenuVar.set(templateName)

        # update system parameters
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
        print("Template Saved Successfully: %s \n" % templateName)

        # reopen other windows
        self.root.deiconify()

    #===========================================================================
    # Function to remove a template
    #===========================================================================

    def removeTemplate(self):
        # function to handle closing
        def close(status):
            if(var.get() != 'Select Template to Remove' and status):
                # delete saved template image
                path = self.systemParameters["Template_Paths"][var.get()]
                filename, ext = os.path.splitext(os.path.split(path)[1])
                path_marked = os.path.split(path)[0] + "\\" + filename + "-marked" + ext
                os.remove(path)
                os.remove(path_marked)

                # delete training data for template
                import shutil
                training_path = "./AppData/training/" + filename + "/"
                shutil.rmtree(training_path)

                # delete model if exists
                model_path = "./AppData/models/" + filename + ".model"
                if(os.path.isfile(model_path)): os.remove(model_path)

                # remove template from preferences
                for i in ["Template Coordinates", "Template Images", "Template Intersections",
                    "Template Tensor", "Template Blob Sizes", "Template Registration Dataset",
                    "Tensor Dataset", "Template Blob Distances", "Template Settings"]:
                    self.config.remove_option(i, var.get())

                for i in ["Templates", "Template_Paths", "Template_Intersections", "Template_Tensors",
                    "Template_Blob_Sizes", "Template_Datasets", "Tensor_Datasets", "Blob_Distances",
                    "Template_Settings"]:
                    self.systemParameters[str(i)].pop(var.get())
                self.systemParameters["Templates_Options"].remove(var.get())

                # update menu if required
                if(self.profileOpen):
                    self.templateMenuVar.set('Select Template')
                    self.templateMenu['menu'].delete(0, 'end')

                    for option in self.systemParameters["Templates_Options"]:
                        self.templateMenu['menu'].add_command(label=option,
                            command=tk._setit(self.templateMenuVar, option))

            # unpack frames
            frame.pack_forget()
            separator.pack_forget()

        # get widgets
        execute, exit, var, frame, separator = self.getRemovePageWidgets(self.root,
            'Select Template to Remove', self.systemParameters["Templates_Options"])
        execute.config(command=lambda: close(True))
        exit.config(command=lambda: close(False))

    #===========================================================================
    # Function to create a preferences profile
    #===========================================================================

    def createProfile(self):
        # embedded function to get name for profile
        def getName():
            def close(status):
                # if name was inputted
                if(var.get() != "" and status):
                    # add registration style to system parameters
                    self.systemParameters["Reg_Params"].append(regStyle.get())
                    self.systemParameters["Misc_Params"] = [deepLearningProfileFlag.get(),
                        ssimProfileFlag.get()]

                    # create output string
                    outputString = "[" + str(self.systemParameters["Upper_Border"]) + "," + \
                        str(self.systemParameters["Lower_Border"]) + "," + '"' + \
                        str(self.templateMenuVar.get()) + '"' + "," + str(self.systemParameters["Clip_Limit"]) + \
                        "," + str(self.systemParameters["Tile_Size"][0]) + "," + str(self.systemParameters["Tile_Size"][1]) + \
                        "," + str(self.systemParameters["Reg_Params"]).replace("[","").replace("]", "") + \
                        "," + str(self.systemParameters["Int_Params"]).replace("[","").replace("]", "") + \
                        "," + str(deepLearningProfileFlag.get()) + "," + str(ssimProfileFlag.get()) + "]"

                    # add to config file
                    self.config.set('Profiles', var.get(), outputString)

                    # update menu
                    self.profileMenu['menu'].add_command(label = var.get(), command = tk._setit(self.profileMenuVar, var.get()))
                    self.systemParameters["Saved_Profiles"][var.get()] = outputString
                    self.systemParameters["Profile_Options"].append(var.get())
                    self.profileMenuVar.set(var.get())

                # unpack frames
                frame.pack_forget()
                separator.pack_forget()

                # close window
                newWindow.destroy()
                self.profileOpen = False # update flag for window

            # get fields are filled in
            if not all(v.get() != "" for v in entries) or self.templateMenuVar.get() == "Select Template" \
                or not all(v.get() != "" for v in regEntries) or not all(v.get() != "" for v in intEntries):
                messagebox.showinfo("Error", "Not All Fields Populated", parent=newWindow)
                return

            # get widgets
            menuFrame.pack_forget()
            execute, exit, var, frame, separator = self.getSavePageWidgets(newWindow, pack=1)
            execute.config(command=lambda: close(True))
            exit.config(command=lambda: close(False))

        # function handle closing of profile window
        def closeProfile():
            self.profileOpen = False
            newWindow.destroy()

        # function to change between option menus
        def changeMenu(newFrame):
            # unpack all frames
            defaultFrame.pack_forget()
            regFrame.pack_forget()
            intFrame.pack_forget()

            # pack selected frame
            newFrame.pack(side=tk.RIGHT, pady=(10, 20), padx=50)

        # create window
        self.profileOpen = True
        newWindow = tk.Toplevel(self.root)
        newWindow.configure(bg=self.gray)
        newWindow.protocol("WM_DELETE_WINDOW", closeProfile)
        newWindow.resizable(False, False)

        # establish frames/pages
        defaultFrame = tk.Frame(newWindow, bg=self.gray)
        regFrame = tk.Frame(newWindow, bg=self.gray)
        intFrame = tk.Frame(newWindow, bg=self.gray)

        #==================================================================
        # Side Menu
        #==================================================================

        menuFrame = tk.Frame(newWindow, bg=self.gray)
        menuLabel = tk.Label(menuFrame, text="Menus", bg=self.gray, fg=self.white,
            font=("Calibri Bold", 16), justify=tk.LEFT, anchor="w")
        f = font.Font(menuLabel, menuLabel.cget("font"))
        f.configure(underline=True)
        menuLabel.configure(font=f)

        # buttons
        defaultMenuButton = tk.Button(menuFrame, text="Basic", bg=self.gray,
            fg=self.white, font=("Calibri Light", 14), borderwidth=0,
            justify=tk.LEFT, anchor="w", command=lambda: changeMenu(defaultFrame))
        regMenuButton = tk.Button(menuFrame, text="Registration", bg=self.gray,
            fg=self.white, font=("Calibri Light", 14), borderwidth=0,
            justify=tk.LEFT, anchor="w", command=lambda: changeMenu(regFrame))
        intMenuButton = tk.Button(menuFrame, text="Intersection", bg=self.gray,
            fg=self.white, font=("Calibri Light", 14), borderwidth=0,
            justify=tk.LEFT, anchor="w", command=lambda: changeMenu(intFrame))

        # packing
        menuLabel.pack(pady=(10,0), fill=tk.X)
        defaultMenuButton.pack(pady=2, fill=tk.X)
        regMenuButton.pack(pady=2, fill=tk.X)
        intMenuButton.pack(pady=2, fill=tk.X)

        #==================================================================
        # Default Page
        #==================================================================

        # setup validate command
        validateCommand = self.root.register(self.validate)

        # frames
        upperBorderFrame = tk.Frame(defaultFrame, bg=self.gray)
        lowerBorderFrame = tk.Frame(defaultFrame, bg=self.gray)
        templateFrame = tk.Frame(defaultFrame, bg=self.gray)
        clipLimitFrame = tk.Frame(defaultFrame, bg=self.gray)
        tileSizeFrame = tk.Frame(defaultFrame, bg=self.gray)
        buttonFrame = tk.Frame(defaultFrame, bg=self.gray)

        # labels
        titleLabel = tk.Label(defaultFrame, text="Preferences", bg=self.gray,
            fg=self.white, font=("Calibri Light", 24))
        upperBorderLabel = tk.Label(upperBorderFrame, text="Upper Border")
        upperBorderUnitLabel = tk.Label(upperBorderFrame, text="px")
        lowerBorderLabel = tk.Label(lowerBorderFrame, text="Lower Border")
        lowerBorderUnitLabel = tk.Label(lowerBorderFrame, text="px")
        templateLabel = tk.Label(templateFrame, text="Template")
        clipLimitLabel = tk.Label(clipLimitFrame, text="Clip Limit")
        tileSizeLabel1 = tk.Label(tileSizeFrame, text="Tile Size")
        tileSizeLabel2 = tk.Label(tileSizeFrame, text="(")
        tileSizeLabel3 = tk.Label(tileSizeFrame, text=",")
        tileSizeLabel4 = tk.Label(tileSizeFrame, text=")")

        labels = [upperBorderLabel, upperBorderUnitLabel, lowerBorderLabel, lowerBorderUnitLabel,
            templateLabel, clipLimitLabel, tileSizeLabel1, tileSizeLabel2, tileSizeLabel3, tileSizeLabel4]
        for label in labels:
            label.config(bg=self.gray, fg=self.white, font=("Calibri Light", 16))

        # checkbox
        deepLearningProfileFlag = tk.IntVar()
        deepLearningProfileFlag.set(1) # default is to use deep learning
        deepLearningProfileCheckBox = tk.Checkbutton(defaultFrame,
            variable=deepLearningProfileFlag, bg=self.gray, fg=self.white,
            text="Use Deep Learning", selectcolor=self.gray,
            activebackground=self.gray, activeforeground=self.white,
            font=("Calibri Light", 16))

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
        self.templateMenuVar = tk.StringVar(defaultFrame)
        self.templateMenuVar.set('Select Template')
        self.templateMenu = tk.OptionMenu(templateFrame, self.templateMenuVar, 'Select Template', *self.systemParameters["Templates_Options"])
        self.templateMenu.config(font=("Calibri Light", 14), bg=self.gray, fg = self.white, activebackground = self.gray,
            activeforeground = self.white, width = 15)
        self.templateMenu["menu"].config(bg=self.gray, fg = self.white)

        # button
        createProfileButton = tk.Button(buttonFrame, text = "Create Profile", command = lambda: getName(), bg=self.gray,
            fg = self.white, font=("Calibri Light", 15), width = 17)

        # packing
        titleLabel.pack(pady = 20)
        upperBorderFrame.pack(pady = 10)
        upperBorderLabel.pack(side = tk.LEFT, padx = 10)
        upperBorderEntry.pack(side = tk.LEFT, padx = 10)
        upperBorderUnitLabel.pack(side=tk.LEFT, padx=5)

        lowerBorderFrame.pack(pady = 15)
        lowerBorderLabel.pack(side = tk.LEFT, padx = 10)
        lowerBorderEntry.pack(side = tk.LEFT, padx = 10)
        lowerBorderUnitLabel.pack(side=tk.LEFT, padx=5)

        templateFrame.pack(pady = 15)
        templateLabel.pack(side = tk.LEFT, padx = 10)
        self.templateMenu.pack(side = tk.LEFT, padx = 10)

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
        deepLearningProfileCheckBox.pack(pady=10)

        buttonFrame.pack(pady = 20)
        createProfileButton.pack(side = tk.LEFT, padx = (5, 20))

        #==================================================================
        # Registration Page
        #==================================================================

        # function to enable/disable checkbox based on radio buttons selection
        def updateSSIMCheckBox():
            if regStyle.get() != 0:
                ssimProfileCheckbox.config(fg="#787d84")
                ssimProfileCheckbox.config(state="disabled")
            else:
                ssimProfileCheckbox.config(fg=self.white)
                ssimProfileCheckbox.config(state="normal")

        # frames
        featureFrame = tk.Frame(regFrame, bg=self.gray)
        ECCParamFrame1 = tk.Frame(regFrame, bg=self.gray)
        ECCParamFrame2 = tk.Frame(regFrame, bg=self.gray)
        ECCParamFrame3 = tk.Frame(regFrame, bg=self.gray)
        ECCParamFrame4 = tk.Frame(regFrame, bg=self.gray)
        buttonFrameReg = tk.Frame(regFrame, bg=self.gray)

        # labels
        titleLabel = tk.Label(regFrame, text="Registration", bg=self.gray,
            fg=self.white, font=("Calibri Light", 24))
        featuresLabel = tk.Label(featureFrame, text="ORB Features")
        ECCLabel1 = tk.Label(regFrame, text="ECC Params (w/ ORB)",
            font=("Calibri Light", 16), bg=self.gray, fg=self.white)
        fECC = font.Font(ECCLabel1, ECCLabel1.cget("font"))
        fECC.config(underline=True)
        ECCLabel1.config(font=fECC)
        ECCLabel2 = tk.Label(ECCParamFrame1, text="Threshold (1e-)")
        ECCLabel3 = tk.Label(ECCParamFrame2, text="Iterations")
        ECCLabel4 = tk.Label(regFrame, text="ECC Params (no ORB)",
            font=fECC, bg=self.gray, fg=self.white)
        ECCLabel5 = tk.Label(ECCParamFrame3, text="Threshold (1e-)")
        ECCLabel6 = tk.Label(ECCParamFrame4, text="Iterations")

        labels = [featuresLabel, ECCLabel2, ECCLabel3, ECCLabel5, ECCLabel6]
        for label in labels:
            label.config(bg=self.gray, fg=self.white, font=("Calibri Light", 16))

        # entries
        featureEntry = tk.Entry(featureFrame, validatecommand =((validateCommand, '%P', "Reg_Params", 0)))
        thresholdEntry1 = tk.Entry(ECCParamFrame1, validatecommand =((validateCommand, '%P', "Reg_Params", 1)))
        iterationsEntry1 = tk.Entry(ECCParamFrame2, validatecommand =((validateCommand, '%P', "Reg_Params", 2)))
        thresholdEntry2 = tk.Entry(ECCParamFrame3, validatecommand =((validateCommand, '%P', "Reg_Params", 3)))
        iterationsEntry2 = tk.Entry(ECCParamFrame4, validatecommand =((validateCommand, '%P', "Reg_Params", 4)))

        # set default values
        featureEntry.insert(0, 262144)
        thresholdEntry1.insert(0, 5)
        iterationsEntry1.insert(0, 1000)
        thresholdEntry2.insert(0, 7)
        iterationsEntry2.insert(0, 2000)
        self.systemParameters["Reg_Params"] = [262144, 5, 1000, 7, 2000]

        regEntries = [featureEntry, thresholdEntry1, iterationsEntry1, thresholdEntry2, iterationsEntry2]
        for entry in regEntries:
            entry.config(validate = "key", font=("Calibri Light", 14), width = 7)

        # button
        createProfileButtonReg = tk.Button(buttonFrameReg, text = "Create Profile",
            command = lambda: getName(), bg=self.gray, fg = self.white,
            font=("Calibri Light", 15), width = 17)

        # radio buttons
        regStyle = tk.IntVar()
        regStyle.set(0) # initialize choice to ORB + ECC
        regOptions = ["Feature and Intensity", "Feature Only", "Intensity Only"]

        regStyleLabel = tk.Label(regFrame, text="Mode", font=fECC, bg=self.gray,
            fg=self.white)
        regRadioButtons = list()
        for val, option in enumerate(regOptions):
            regRadioButtons.append(tk.Radiobutton(regFrame, text=option,
                variable=regStyle, value=val, bg=self.gray, fg=self.white,
                font=("Calibri Light", 16), activebackground=self.gray,
                activeforeground=self.white, selectcolor=self.gray,
                command=lambda: updateSSIMCheckBox()))

        # checkbox
        ssimProfileFlag = tk.IntVar() # default is not to use SSIM
        ssimProfileCheckbox = tk.Checkbutton(regFrame,
            variable=ssimProfileFlag, bg=self.gray, fg=self.white,
            text="Use SSIM Selection", selectcolor=self.gray,
            activebackground=self.gray, activeforeground=self.white,
            font=("Calibri Light", 16))

        # packing
        titleLabel.pack(pady = 20)
        featureFrame.pack(pady=10, padx=50)
        featuresLabel.pack(side=tk.LEFT, padx=10)
        featureEntry.pack(side=tk.LEFT, padx=10)

        ECCLabel1.pack(pady=(15,10))
        ECCParamFrame1.pack(pady=5)
        ECCLabel2.pack(side=tk.LEFT, padx=(25,10))
        thresholdEntry1.pack(side=tk.LEFT, padx=(10,40))
        ECCParamFrame2.pack(pady=5)
        ECCLabel3.pack(side=tk.LEFT, padx=(32,10))
        iterationsEntry1.pack(side=tk.LEFT, padx=(10,0))

        ECCLabel4.pack(pady=(15,10))
        ECCParamFrame3.pack(pady=5)
        ECCLabel5.pack(side=tk.LEFT, padx=(25,10))
        thresholdEntry2.pack(side=tk.LEFT, padx=(10,40))
        ECCParamFrame4.pack(pady=5)
        ECCLabel6.pack(side=tk.LEFT, padx=(32,10))
        iterationsEntry2.pack(side=tk.LEFT, padx=(10,0))
        ssimProfileCheckbox.pack(anchor=tk.W, pady=10, padx=(60,0))

        regStyleLabel.pack(pady=(0, 10))
        for x in regRadioButtons:
            x.pack(anchor=tk.W, padx=(60,0))

        buttonFrameReg.pack(pady = 20)
        createProfileButtonReg.pack(side = tk.LEFT, padx = (5, 20))

        #==================================================================
        # Intersection Page
        #==================================================================

        # frames
        peakFrame = tk.Frame(intFrame, bg=self.gray)
        snowThresholdFrame = tk.Frame(intFrame, bg=self.gray)
        minSnowThresholdFrame = tk.Frame(intFrame, bg=self.gray)
        stakeCoverFrame = tk.Frame(intFrame, bg=self.gray)
        snowCoverFrame = tk.Frame(intFrame, bg=self.gray)
        buttonFrameInt = tk.Frame(intFrame, bg=self.gray)

        # labels
        titleLabel = tk.Label(intFrame, text="Intersection", bg=self.gray,
            fg=self.white, font=("Calibri Light", 24))
        peakLabel = tk.Label(peakFrame, text="Peak Height")
        snowThresholdLabel = tk.Label(snowThresholdFrame, text="Peak Threshold (%)")
        minSnowThresholdLabel = tk.Label(minSnowThresholdFrame, text="Min Threshold")
        stakeCoverLabel = tk.Label(stakeCoverFrame, text="Stake Coverage (%)")
        snowCoverLabel = tk.Label(snowCoverFrame, text="Snow Coverage (%)")

        labels = [peakLabel, snowThresholdLabel, minSnowThresholdLabel, stakeCoverLabel,
            snowCoverLabel]
        for label in labels:
            label.config(bg=self.gray, fg=self.white, font=("Calibri Light", 16))

        # entries
        peakEntry = tk.Entry(peakFrame, validatecommand =((validateCommand, '%P', "Int_Params", 0)))
        snowThresholdEntry = tk.Entry(snowThresholdFrame, validatecommand =((validateCommand, '%P', "Int_Params", 1, 100.0)))
        minSnowThresholdEntry = tk.Entry(minSnowThresholdFrame, validatecommand =((validateCommand, '%P', "Int_Params", 2)))
        stakeCoverEntry = tk.Entry(stakeCoverFrame, validatecommand =((validateCommand, '%P', "Int_Params", 3, 100.0)))
        snowCoverEntry = tk.Entry(snowCoverFrame, validatecommand =((validateCommand, '%P', "Int_Params", 4, 100.0)))

        # set default values
        peakEntry.insert(0, 90)
        snowThresholdEntry.insert(0, 65)
        minSnowThresholdEntry.insert(0, 125)
        stakeCoverEntry.insert(0, 50)
        snowCoverEntry.insert(0, 50)
        self.systemParameters["Int_Params"] = [90, 0.65, 125, 0.50, 0.50]

        intEntries = [peakEntry, snowThresholdEntry, minSnowThresholdEntry, stakeCoverEntry,
            snowCoverEntry]
        for entry in intEntries:
            entry.config(validate = "key", font=("Calibri Light", 14), width = 7)

        # button
        createProfileButtonInt = tk.Button(buttonFrameInt, text = "Create Profile",
            command = lambda: getName(), bg=self.gray, fg = self.white,
            font=("Calibri Light", 15), width = 17)

        # packing
        titleLabel.pack(pady = 20)
        peakFrame.pack(pady=15, padx=50)
        peakLabel.pack(side=tk.LEFT, padx=(40,10))
        peakEntry.pack(side=tk.LEFT, padx=(10,0))

        snowThresholdFrame.pack(pady=15)
        snowThresholdLabel.pack(side=tk.LEFT, padx=(0,10))
        snowThresholdEntry.pack(side=tk.LEFT, padx=(10,20))

        minSnowThresholdFrame.pack(pady=15)
        minSnowThresholdLabel.pack(side=tk.LEFT, padx=(20,10))
        minSnowThresholdEntry.pack(side=tk.LEFT, padx=(10,3))

        stakeCoverFrame.pack(pady=15)
        stakeCoverLabel.pack(side=tk.LEFT, padx=(0,10))
        stakeCoverEntry.pack(side=tk.LEFT, padx=(10,20))

        snowCoverFrame.pack(pady=15)
        snowCoverLabel.pack(side=tk.LEFT, padx=(0,10))
        snowCoverEntry.pack(side=tk.LEFT, padx=(10,20))

        buttonFrameInt.pack(pady = 20)
        createProfileButtonInt.pack(side = tk.LEFT, padx = (5, 20))

        menuFrame.pack(side=tk.LEFT, anchor=tk.N, pady=(20,0), padx=(20,0))
        defaultFrame.pack(side = tk.RIGHT, pady = (10,20), padx = 50)

        # wait until user inputs name
        self.root.wait_window(newWindow)

    #===========================================================================
    # Function to remove a preferences profile
    #===========================================================================

    def removeProfile(self):
        # function to close window
        def close(status):
            # remove selected option from menu
            if(var.get() != 'Select Profile to Remove' and status):
                self.config.remove_option("Profiles", var.get())
                self.systemParameters["Profile_Options"].remove(var.get())
                self.systemParameters["Saved_Profiles"].pop(var.get())

                # update menu
                self.profileMenuVar.set('Select Profile')
                self.profileMenu['menu'].delete(0, 'end')

                for option in self.systemParameters["Profile_Options"]:
                    self.profileMenu['menu'].add_command(label=option,
                        command=tk._setit(self.profileMenuVar, option))

            # unpack frames
            frame.pack_forget()
            separator.pack_forget()

        # get widgets
        execute, exit, var, frame, separator = self.getRemovePageWidgets(self.root,
            'Select Profile to Remove', self.systemParameters["Profile_Options"])
        execute.config(command=lambda: close(True))
        exit.config(command=lambda: close(False))

    #===========================================================================
    # Function to preview preferences profile
    #===========================================================================

    def previewProfile(self):
        if(self.profileMenuVar.get() != 'Select Profile'):
            # create window
            newWindow = tk.Toplevel(self.root)
            newWindow.configure(bg=self.gray)
            newWindow.resizable(False, False)

            # frames
            leftFrame = tk.Frame(newWindow, bg=self.gray)
            rightFrame = tk.Frame(newWindow, bg=self.gray)

            # labels
            titleLabel = tk.Label(leftFrame, text = str(self.profileMenuVar.get()), bg=self.gray, fg = self.white, font=("Calibri Light", 24))
            underlineFont = font.Font(family="Calibri Light", size=18, underline=True)
            generalLabel = tk.Label(leftFrame, text="General", font=underlineFont, bg=self.gray, fg=self.white)
            upperBorderLabel = tk.Label(leftFrame, text = "Upper Border: " + str(self.systemParameters["Upper_Border"]))
            lowerBorderLabel = tk.Label(leftFrame, text = "Lower Border: " + str(self.systemParameters["Lower_Border"]))
            templateLabel = tk.Label(leftFrame, text = "Template: " + str(self.systemParameters["Current_Template_Name"]))
            clipLimitLabel = tk.Label(leftFrame, text = "Clip Limit: " + str(self.systemParameters["Clip_Limit"]))
            tileSizeLabel = tk.Label(leftFrame, text = "Tile Size: " + str(self.systemParameters["Tile_Size"]))

            regLabel = tk.Label(leftFrame, text="Registration", font=underlineFont, bg=self.gray, fg=self.white)
            featuresLabel = tk.Label(leftFrame, text="ORB Features: " + str(self.systemParameters["Reg_Params"][0]))
            ECCLabel1 = tk.Label(leftFrame, text="ECC Threshold (ORB): 1e-" + str(self.systemParameters["Reg_Params"][1]))
            ECCLabel2 = tk.Label(leftFrame, text="ECC Iterations (ORB): " + str(self.systemParameters["Reg_Params"][2]))
            ECCLabel3 = tk.Label(leftFrame, text="ECC Threshold: 1e-" + str(self.systemParameters["Reg_Params"][3]))
            ECCLabel4 = tk.Label(leftFrame, text="ECC Iterations: " + str(self.systemParameters["Reg_Params"][4]))

            intLabel = tk.Label(leftFrame, text="Intersection", font=underlineFont, bg=self.gray, fg=self.white)
            peakLabel = tk.Label(leftFrame, text="Peak Height: " + str(self.systemParameters["Int_Params"][0]))
            threshLabel = tk.Label(leftFrame, text="Peak Threshold (%): " + str(self.systemParameters["Int_Params"][1]*100.0))
            minThreshLabel = tk.Label(leftFrame, text= "Min Threshold: " + str(self.systemParameters["Int_Params"][2]))
            stakeCoverLabel = tk.Label(leftFrame, text="Stake Coverage (%): " + str(self.systemParameters["Int_Params"][3]*100.0))
            snowCoverLabel = tk.Label(leftFrame, text="Snow Coverage (%): " + str(self.systemParameters["Int_Params"][4]*100.0))

            labels = [upperBorderLabel, lowerBorderLabel, templateLabel, clipLimitLabel, tileSizeLabel,
                featuresLabel, ECCLabel1, ECCLabel2, ECCLabel3, ECCLabel4, peakLabel, threshLabel,
                minThreshLabel, stakeCoverLabel, snowCoverLabel]

            for label in labels:
                label.config(bg=self.gray, fg = self.white, font=("Calibri Light", 16))

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
                bg=self.gray, scrollregion = (0, 0, imgWidth, imgHeight))

            # packing
            leftFrame.pack(side = tk.LEFT, padx = 40)
            titleLabel.pack(pady = 10)
            generalLabel.pack(pady=(20,0))
            upperBorderLabel.pack(pady = 5)
            lowerBorderLabel.pack(pady = 5)
            templateLabel.pack(pady = 5)
            clipLimitLabel.pack(pady = 5)
            tileSizeLabel.pack(pady = 5)
            regLabel.pack(pady=(20,0))
            featuresLabel.pack(pady = 5)
            ECCLabel1.pack(pady = 5)
            ECCLabel2.pack(pady = 5)
            ECCLabel3.pack(pady = 5)
            ECCLabel4.pack(pady = 5)
            intLabel.pack(pady=(20,0))
            peakLabel.pack(pady = 5)
            threshLabel.pack(pady = 5)
            minThreshLabel.pack(pady = 5)
            stakeCoverLabel.pack(pady = 5)
            snowCoverLabel.pack(pady = 5)

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


    #===========================================================================
    # Function to run HSV range preview
    #===========================================================================

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
        newWindow.resizable(False, False)
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
            savetoMain = tk.Button(newWindow, text = "Save Values",bg=self.gray,fg = self.white,command = lambda: saveValuestoMain(),
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
            previewCheckBox.pack(pady = (20, 10))
            savetoMain.pack(pady = (10,25))

            # open comparison window
            cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

            # open image
            img = cv2.imread(filename)

            # resize image to 1/4 of original size
            img = cv2.resize(img, (0,0), None, 0.25, 0.25)

            # present image and blank mask
            cv2.imshow("Comparison", np.hstack((img, np.zeros(img.shape, np.uint8))))

            # wait until user closes window
            self.root.wait_window(newWindow)

# class to enable hovering over widgets for additional info
class createHoverLabel(object):
    def __init__(self, widget, text):
        self.time = 2000 # wait half a second before showing
        self.wraplength = 200 # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.close)
        self.widget.bind("<ButtonPress>", self.close)
        self.gray = "#282c34"
        self.tw = None
        self.id = None

    # function to schedule label appearance (uses wait time)
    def schedule(self, event=None):
        self.id = self.widget.after(self.time , self.enter)

    # function which is run when the user hovers over a widget
    def enter(self, event=None):
        # get window geometry
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # create Toplevel window
        self.tw = tk.Toplevel(self.widget)

        # Leave only label
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))

        # Insert label
        label = tk.Label(self.tw, text=self.text, justify='left',
            bg='white', fg=self.gray, relief='solid', borderwidth=1,
            font=("Calibri Light", 12))
        label.pack(ipadx=1)
        label.after(5000, self.close)

    # function which is run when the user stops hovering over a widget
    def close(self, event=None):
        if self.tw:
            self.tw.destroy() # destroy top level window if it exists
        if self.id:
            self.widget.after_cancel(self.id)
