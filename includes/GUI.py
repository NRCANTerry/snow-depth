# import necessary modules
import Tkinter as tk
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
from scipy.spatial import distance as dist

class GUI:
    def __init__(self, master):

        # ---------------------------------------------------------------------------------
        # Create window
        # ---------------------------------------------------------------------------------

    	self.root = master
    	self.root.configure(background='#ffffff')
        self.root.title("Measure Snow Depth")
        #self.root.geometry("600x700")

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
        	"Lower_Blob_Size": 0,
        	"Upper_Blob_Size": 0,
        	"Clip_Limit": 0,
        	"Tile_Size": [0,0],
        	"Saved_Colours": dict(),
        	"Saved_Profiles": dict(),
        	"Colour_Options": list(),
        	"Profile_Options": list(),
        	"Templates": dict(), 
        	"Templates_Options": list(),
        	"Template_Paths": dict(),
        	"Current_Template_Name": "",
        	"Current_Template_Coords": list(),
        	"Current_Template_Path": "",
        	"Window_Closed": False
        }

        # ConfigParser object
        self.config = ConfigParser.ConfigParser()

        # open preferences file
        self.systemParameters["Saved_Colours"], self.systemParameters["Saved_Profiles"], self.systemParameters["Templates"], self.systemParameters["Template_Paths"] = self.getPreferences()
        self.systemParameters["Colour_Options"] = list(self.systemParameters["Saved_Colours"].keys())
        self.systemParameters["Profile_Options"] = list(self.systemParameters["Saved_Profiles"].keys())
        self.systemParameters["Templates_Options"] = list(self.systemParameters["Templates"].keys())

        # window closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---------------------------------------------------------------------------------
        # Labels
        # ---------------------------------------------------------------------------------

        # Step 1
        self.step1Label = tk.Label(self.root, text = "Image Folder")
        self.pathLabel = tk.Label(self.root, text = "No Directory Selected")

        # Step 2
        self.step2Label = tk.Label(self.root, text = "HSV Range")

        # H, S, V for Range 1
        self.range1Frame = tk.Frame(self.root, bg='#ffffff')
        self.lowerH1 = tk.Label(self.range1Frame, text = "H")
        self.lowerS1 = tk.Label(self.range1Frame, text = "S")
        self.lowerV1 = tk.Label(self.range1Frame, text = "V")
        self.arrow1 = tk.Label(self.range1Frame, text = "-->")
        self.upperH1 = tk.Label(self.range1Frame, text = "H")
        self.upperS1 = tk.Label(self.range1Frame, text = "S")
        self.upperV1 = tk.Label(self.range1Frame, text = "V")

        # H, S, V for Range 2
        self.range2Frame = tk.Frame(self.root, bg='#ffffff')
        self.lowerH2 = tk.Label(self.range2Frame, text = "H")
        self.lowerS2 = tk.Label(self.range2Frame, text = "S")
        self.lowerV2 = tk.Label(self.range2Frame, text = "V")
        self.arrow2 = tk.Label(self.range2Frame, text = "-->")
        self.upperH2 = tk.Label(self.range2Frame, text = "H")
        self.upperS2 = tk.Label(self.range2Frame, text = "S")
        self.upperV2 = tk.Label(self.range2Frame, text = "V")

        # Step 3
        self.step3Label = tk.Label(self.root, text = "Settings")

        # lists containing labels
        self.titleLabels = [self.step1Label, self.step2Label, self.step3Label]
        self.otherLabels = [self.pathLabel, self.lowerH1, self.lowerS1, self.lowerV1, self.arrow1, self.upperH1, self.upperS1, self.upperV1]
        self.grayLabels = [self.lowerH2, self.lowerS2, self.lowerV2, self.arrow2, self.upperH2, self.upperS2, self.upperV2]

        # configure title labels
        for label in self.titleLabels:
        	label.config(bg = "#ffffff", fg = "#000000", font=("Calibri Light", 24))

        # configure other labels
        for label in self.otherLabels:
        	label.config(bg = "#ffffff", fg = "#000000", font=("Calibri Light", 14))

        # configure gray labels
        for label in self.grayLabels:
        	label.config(bg = "#ffffff", fg = "#D3D3D3", font=("Calibri Light", 14))

        # ---------------------------------------------------------------------------------
        # Buttons
        # ---------------------------------------------------------------------------------

        # choose directory button
        self.directoryButton = tk.Button(self.root, text = "Select", command = lambda: self.selectDirectory())

        # execute button
        self.runButton = tk.Button(self.root, text = "Run", command = lambda: self.saveValues())

        # list containing buttons
        self.buttons = [self.directoryButton, self.runButton]

        # configure buttons
        for button in self.buttons:
        	button.config(bg = "#ffffff", fg = "#000000", font=("Calibri Light", 14), width = 17)

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
        for entry in (self.entries1 + self.entries2):
        	entry.config(validate = "key", font=("Calibri Light", 13), width = 4)

        # ---------------------------------------------------------------------------------
        # Checkbox
        # ---------------------------------------------------------------------------------

        self.secondHSVFlag = tk.IntVar()
        self.checkBox = tk.Checkbutton(self.root, text="Second HSV Range", bg='#ffffff', fg='#000000',
        	variable = self.secondHSVFlag, command = lambda:self.updateSelections(), font=("Calibri Light", 14))

        self.debug = tk.IntVar()
        self.debugCheckBox = tk.Checkbutton(self.root, text="Debug", bg='#ffffff', fg='#000000',
        	variable = self.debug, font=("Calibri Light", 12))

        # ---------------------------------------------------------------------------------
        # Drop Down Menus
        # ---------------------------------------------------------------------------------

        self.colourMenuVar = tk.StringVar(self.root)
        self.colourMenuVar.set('Select HSV Range')
        self.colourMenu = tk.OptionMenu(self.root, self.colourMenuVar, 'Select HSV Range', *self.systemParameters["Colour_Options"])
        self.colourMenu.config(font=("Calibri Light", 13), bg='#ffffff', width = 15)
        self.colourMenuVar.trace('w', self.change_HSV_dropdown)

        self.profileMenuVar = tk.StringVar(self.root)
        self.profileMenuVar.set('Select Profile')
        self.profileMenu = tk.OptionMenu(self.root, self.profileMenuVar, 'Select Profile', *self.systemParameters["Profile_Options"])
        self.profileMenu.config(font=("Calibri Light", 13), bg='#ffffff', width = 15)
        self.profileMenuVar.trace('w', self.change_Preferences_dropdown)

        # ---------------------------------------------------------------------------------
        # Top Menu
        # ---------------------------------------------------------------------------------

        # create menu bar
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.HSVmenu = tk.Menu(self.menubar, tearoff=0)
        self.prefMenu = tk.Menu(self.menubar, tearoff=0)

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
        # Packing
        # ---------------------------------------------------------------------------------

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
        self.colourMenu.pack(pady = 10)

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
    # Function to confirm that required fields are filled in
    # ---------------------------------------------------------------------------------

    def fieldsFilled(self, directory = True):
		if(directory):
			return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
				and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
				and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
				and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
				and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1) and self.systemParameters["Directory"] != "" \
				and self.profileMenuVar.get() != "Select Profile")
		else:
			return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
				and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
				and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
				and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
				and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1) and self.profileMenuVar.get() != "Select Profile")

    # ---------------------------------------------------------------------------------
    # Function to allow selection of directory/file where images are stored
    # ---------------------------------------------------------------------------------

    def selectDirectory(self):
        # open directory selector
        dirname = tkFileDialog.askdirectory(parent=self.root, initialdir="/", title='Select Directory')

        # if new directory selected, update label
        if (len(dirname) > 0):
            self.pathLabel.config(text=dirname)
            self.systemParameters["Directory"] = str(dirname)

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
            with open('./preferences.cfg', 'wb') as configfile:
                self.config.write(configfile)

            # close window
            self.root.destroy()

        # else show error
        else:
            tkMessageBox.showinfo("Error", "Not All Fields Populated")

    # ---------------------------------------------------------------------------------
    # Accessor function to return parameters to main file
    # ---------------------------------------------------------------------------------

    def getValues(self):
    	# return values in tuple format
        if(self.systemParameters["Window_Closed"]):
			return self.systemParameters["Directory"], self.systemParameters["Lower_HSV_1"], self.systemParameters["Upper_HSV_1"], \
					self.systemParameters["Lower_HSV_2"], self.systemParameters["Upper_HSV_2"], self.systemParameters["Upper_Border"], \
					self.systemParameters["Lower_Border"], self.systemParameters["Lower_Blob_Size"], self.systemParameters["Upper_Blob_Size"], \
					self.systemParameters["Current_Template_Coords"], self.systemParameters["Current_Template_Path"], self.systemParameters["Clip_Limit"], \
					tuple(self.systemParameters["Tile_Size"]), (self.debug.get() == 1)

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
                label.config(fg ='#000000')
            # update fields
            for field in self.entries2:
                field.config(state = "normal")
        else:
            # update labels
            for label in self.grayLabels:
                label.config(fg ='#D3D3D3')            
            # update fields
            for field in self.entries2:
                field.delete(0, tk.END)
                field.config(state = "disabled")

    # ---------------------------------------------------------------------------------
    # Function to fetch preferences from preferences.cfg file
    # ---------------------------------------------------------------------------------

    def getPreferences(self):
        # if no preferences file present, create one
        if(str(self.config.read('./preferences.cfg')) == "[]"):
            self.config.add_section('HSV Ranges')
            self.config.add_section('Profiles')
            self.config.add_section('Template Coordinates')
            self.config.add_section('Template Images')
        # else read in existing file
        else:
            self.config.read('./preferences.cfg')

        # load in preferences
        return dict(self.config.items('HSV Ranges')), dict(self.config.items('Profiles')), \
        	dict(self.config.items('Template Coordinates')), dict(self.config.items('Template Images'))

    # ---------------------------------------------------------------------------------
    # Function that is run when user closes window
    # ---------------------------------------------------------------------------------

    def on_closing(self):
        # write preferences to file
        with open('./preferences.cfg', 'wb') as configfile:
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
            self.systemParameters["Lower_Blob_Size"] = int(optionsList[2])
            self.systemParameters["Upper_Blob_Size"] = int(optionsList[3])
            self.systemParameters["Current_Template_Name"] = str(optionsList[4])
            self.systemParameters["Current_Template_Coords"] = ast.literal_eval(self.systemParameters["Templates"][str(optionsList[4])])
            self.systemParameters["Current_Template_Path"] = self.systemParameters["Template_Paths"][str(optionsList[4])]
            self.systemParameters["Clip_Limit"] = int(optionsList[5])
            self.systemParameters["Tile_Size"] = [int(optionsList[6]), int(optionsList[7])]	

    # ---------------------------------------------------------------------------------
    # Function to restart script to load changes
    # ---------------------------------------------------------------------------------

    def restart(self):
	    # write preferences to file
	    with open('./preferences.cfg', 'wb') as configfile:
	        self.config.write(configfile)

	    # close window
	    self.root.destroy()

	    # restart program
	    os.execv(sys.executable, ['C:\\Users\\tbaricia\\AppData\\Local\\Continuum\\miniconda2\\python.exe'] + sys.argv)

    # ---------------------------------------------------------------------------------
    # Function to centre window
    # ---------------------------------------------------------------------------------

    def centre_window(self, window, width, height):
    	# get screen width and height
    	screen_width = self.root.winfo_screenwidth()
    	screen_height = self.root.winfo_screenheight()

    	# calculate position x and y coords
    	x = (screen_width/2) - (width/2)
    	y = (screen_height/2) - (height/2)

    	# centre window
    	window.geometry('%dx%d+%d+%d' % (width, height, x, y))

    # ---------------------------------------------------------------------------------
    # Function to allow user to save HSV ranges to preferences file
    # ---------------------------------------------------------------------------------

    def saveRanges(self):
        # if required fields are filled in
        if(self.fieldsFilled(False)):
            # ask for name
            name = tk.StringVar()
            newWindow = tk.Toplevel(self.root)
            newWindow.configure(bg='#ffffff')
            self.centre_window(newWindow, 300, 170)
            
            # widgets
            nameLabel = tk.Label(newWindow,text="Name",bg='#ffffff',fg='#000000',font=("Calibri Light", 20))
            nameEntry = tk.Entry(newWindow,font=("Calibri Light", 14),textvariable = name,width = 20)
            nameButton = tk.Button(newWindow,text = "Save",bg='#ffffff',fg='#000000',command = lambda: newWindow.destroy(),
                width = 20,font=("Calibri Light", 14))

            # packing
            nameLabel.pack(pady = (15,5))
            nameEntry.pack(pady = 10)
            nameButton.pack(pady = 5)

            # wait until user inputs name
            self.root.wait_window(newWindow)

            # if name was inputted
            if(name.get() != ""):
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

        # show error message is all fields aren't populated
        else:
            tkMessageBox.showinfo("Error", "Not All HSV Fields Populated")

    # ---------------------------------------------------------------------------------
    # Function to remove an HSV range from the preferences file
    # ---------------------------------------------------------------------------------

    def removeRanges(self):
		# function to close window
		def closing():
			# close window
			newWindow.destroy()

			# remove selected option from menu
			if(removecolourMenuVar.get() != 'Select Profile to Remove'):
				self.config.remove_option("HSV Ranges", removecolourMenuVar.get())
				self.systemParameters["Colour_Options"].remove(removecolourMenuVar.get())
				self.systemParameters["Saved_Colours"].pop(removecolourMenuVar.get())

				# update menu
				self.colourMenuVar.set('Select HSV Range')
				self.colourMenu['menu'].delete(0, 'end')

				for option in self.systemParameters["Colour_Options"]:
					self.colourMenu['menu'].add_command(label = option, command = tk._setit(self.colourMenuVar, option))

    	# create window 
		newWindow = tk.Toplevel(self.root)
		newWindow.configure(bg='#ffffff')
		self.centre_window(newWindow, 300, 170)

		# labels
		nameLabel = tk.Label(newWindow,text="Name",bg='#ffffff',fg='#000000',font=("Calibri Light", 20))

		# drop down menu used to select profile to delete
		removecolourMenuVar = tk.StringVar(self.root)
		removecolourMenuVar.set('Select Profile to Remove')
		removeMenu = tk.OptionMenu(newWindow, removecolourMenuVar, 'Select Profile to Remove', *self.systemParameters["Colour_Options"])
		removeMenu.config(font=("Calibri Light", 13))
		removeMenu.config(bg='#ffffff')

		# button
		nameButton = tk.Button(newWindow,text = "Save",bg='#ffffff',fg='#000000',command = lambda: closing(),
		    width = 20,font=("Calibri Light", 14))

		# packing
		nameLabel.pack(pady = (20,5))
		removeMenu.pack(pady = 10)
		nameButton.pack(pady = 5)

		# wait until user inputs name
		self.root.wait_window(newWindow)

    # ---------------------------------------------------------------------------------
    # Function to create a preferences profile
    # ---------------------------------------------------------------------------------

    def createProfile(self):
		# function to order points of minAreaRect
		def orderPoints(pts):
			# sort the points based on their x-coordinates
			xSorted = pts[np.argsort(pts[:, 0]), :]

			# get left-most and right-most points
			leftMost = xSorted[:2, :]
			rightMost = xSorted[2:, :]

			# sort left-most coordinates according to y coordinates
			leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
			(tl, bl) = leftMost

			# find bottom right
			D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
			(br, tr) = rightMost[np.argsort(D)[::-1], :]

			# return coordiantes in top-left, top-right, bottom-right, bottom-left order
			return tl, tr, br, bl

		# embedded function to allow for removal of template
		def removeTemplate():
			# function to close window
			def closing():
				# close window
				removeTemplateWindow.destroy()

				# remove selected option from menu
				if(removetemplateMenuVar.get() != 'Select Template to Remove'):
					# delete saved template image
					path = self.systemParameters["Template_Paths"][removetemplateMenuVar.get()]
					os.remove(path)

					self.config.remove_option("Template Coordinates", removetemplateMenuVar.get())
					self.config.remove_option("Template Images", removetemplateMenuVar.get())
					self.systemParameters["Templates"].pop(removetemplateMenuVar.get())
					self.systemParameters["Templates_Options"].remove(removetemplateMenuVar.get())
					self.systemParameters["Template_Paths"].pop(removetemplateMenuVar.get())



					# update menu 
					templateMenuVar.set('Select Template')
					templateMenu['menu'].delete(0, 'end')

					for option in self.systemParameters["Templates_Options"]:
						templateMenu['menu'].add_command(label = option, command = tk._setit(templateMenuVar, option))

			# create window
			removeTemplateWindow = tk.Toplevel(newWindow)
			removeTemplateWindow.configure(bg='#ffffff')
			self.centre_window(removeTemplateWindow, 300, 170)	

			# labels
			nameLabel = tk.Label(removeTemplateWindow,text="Name",bg='#ffffff',fg='#000000',font=("Calibri Light", 20))

			# drop down menu used to select profile to delete
			removetemplateMenuVar = tk.StringVar(self.root)
			removetemplateMenuVar.set('Select Template to Remove')
			removeTemplateMenu = tk.OptionMenu(removeTemplateWindow, removetemplateMenuVar, 'Select Template to Remove', *self.systemParameters["Templates_Options"])
			removeTemplateMenu.config(font=("Calibri Light", 13))
			removeTemplateMenu.config(bg='#ffffff')

			# button
			nameButton = tk.Button(removeTemplateWindow,text = "Save",bg='#ffffff',fg='#000000',command = lambda: closing(),
				width = 20,font=("Calibri Light", 14))

			# packing
			nameLabel.pack(pady = (20,5))
			removeTemplateMenu.pack(pady = 10)
			nameButton.pack(pady = 5)
			
			# wait until user selects template
			newWindow.wait_window(removeTemplateWindow)

    	# embedded function to allow for creation of template
		def createTemplate():
			# embedded function to get template paths
			def getImage(type):
				filename = tkFileDialog.askopenfilename(initialdir = "/",title = "Select " + str(type) + " template",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
				shortName = os.path.split(filename)[1]
				if(type == "marked"):
					self.markedTemplate = filename
					directoryPathLabel1.config(text=shortName)
				else:
					self.unmarkedTemplate = filename
					directoryPathLabel2.config(text=shortName)

			# open new window
			templateWindow = tk.Toplevel(newWindow)
			templateWindow.configure(bg='#ffffff')
			self.centre_window(templateWindow, 450, 750)

			# hide other windows
			newWindow.withdraw()
			self.root.withdraw()

			# frames
			lowerFrame = tk.Frame(templateWindow, bg='#ffffff')
			upperFrame = tk.Frame(templateWindow, bg='#ffffff')
			nameFrame = tk.Frame(templateWindow, bg='#ffffff')

			# labels
			titleLabel = tk.Label(templateWindow, text = "Generate Template", bg = '#ffffff', fg = '#000000', font=("Calibri Light", 24))
			directoryLabel1 = tk.Label(templateWindow, text = "Select Marked Template",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 18))
			directoryPathLabel1 = tk.Label(templateWindow, text = "No Template Selected",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 14))
			directoryLabel2 = tk.Label(templateWindow, text = "Select Unmarked Template",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 18))
			directoryPathLabel2 = tk.Label(templateWindow, text = "No Template Selected",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 14))
			parametersLabel = tk.Label(templateWindow, text = "Parameters",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 18))
			lowerSizeLabel = tk.Label(lowerFrame, text = "Lower Blob Size",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 16))
			upperSizeLabel = tk.Label(upperFrame, text = "Upper Blob Size",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 16))
			nameLabel = tk.Label(nameFrame, text = "Template Name",bg = "#ffffff", fg = "#000000", font=("Calibri Light", 16))

			# entries
			name = tk.StringVar()
			lowerSize = tk.IntVar()
			upperSize = tk.IntVar()
			nameEntry = tk.Entry(nameFrame, font=("Calibri Light", 14),textvariable=name, width=10)
			lowerEntry = tk.Entry(lowerFrame, font=("Calibri Light", 14),textvariable=lowerSize, width=10)
			upperEntry = tk.Entry(upperFrame, font=("Calibri Light", 14),textvariable=upperSize, width=10)

			# set default values
			lowerEntry.insert(0,100)
			upperEntry.insert(0,10000)
			lowerSize.set(100)
			upperSize.set(10000)

			# buttons
			self.markedTemplate = ""
			self.unmarkedTemplate = ""

			nameButton = tk.Button(templateWindow, text="Generate",bg='#ffffff',fg='#000000',command = lambda: templateWindow.destroy(),
				width=20,font=("Calibri Light",14))
			dirButton1 = tk.Button(templateWindow, text="Select Marked Template",bg='#ffffff',fg='#000000',command = lambda: getImage("marked"),
				width=20,font=("Calibri Light",14))
			dirButton2 = tk.Button(templateWindow, text="Select Unmarked Template",bg='#ffffff',fg='#000000',command = lambda: getImage("unmarked"),
				width=20,font=("Calibri Light",14))

			# packing
			titleLabel.pack(pady = 20)
			directoryLabel1.pack(pady = 10)
			directoryPathLabel1.pack(pady = 10)
			dirButton1.pack(pady = 10)
			directoryLabel2.pack(pady = (20, 10))
			directoryPathLabel2.pack(pady = 10)
			dirButton2.pack(pady = 10)
			parametersLabel.pack(pady = (20,10))
			lowerFrame.pack(pady = 10)
			lowerSizeLabel.pack(side = tk.LEFT, padx = (20,5))
			lowerEntry.pack(side = tk.LEFT, padx = (5, 20))
			upperFrame.pack(pady = 10)
			upperSizeLabel.pack(side = tk.LEFT, padx = (20,5))
			upperEntry.pack(side = tk.LEFT, padx = (5, 20))			
			nameFrame.pack(pady = 10)
			nameLabel.pack(side = tk.LEFT, padx = (20,5))
			nameEntry.pack(side = tk.LEFT, padx = (5,20))
			nameButton.pack(pady = (20,10))

			self.root.wait_window(templateWindow)

			# if valid filename
			if(self.markedTemplate != "" and self.unmarkedTemplate != "" and name.get() != ""):
				# hsv ranges
				lower_pink = np.array([125, 10, 50])
				upper_pink = np.array([160, 255, 255])
				lower_green = np.array([60, 160, 130])
				upper_green = np.array([70, 255, 255])
				min_contour_area = lowerSize.get()
				max_contour_area = upperSize.get()

				# import image
				img = cv2.imread(self.markedTemplate)
				img_unmarked = cv2.imread(self.unmarkedTemplate)
				img_save = img_unmarked.copy()

				# convert to HSV colour space
				hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

				# create mask to find stakes
				stake_mask = cv2.inRange(hsv, lower_pink, upper_pink)

				# remove noise
				kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
				stake_mask= cv2.morphologyEx(stake_mask, cv2.MORPH_OPEN, kernel)

				# find contours
				stake_contours = cv2.findContours(stake_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

				# list containing stake coordinates
				stakes_coords = list()

				# variables for number of stakes and blobs found
				stakes = 0
				blobs = 0
				blob_area = 0.0
				stake_area = 0.0

				# iterate through stake contours
				for cnt in stake_contours:
					# get contour coordinates
					rect = cv2.minAreaRect(cnt)
					box = np.array(cv2.boxPoints(rect), dtype = "int")

					# order points
					points = orderPoints(box)
					points_list = [[points[0], points[2]]]
					stakes_coords.append(points_list)

					# increment stake counter
					stakes += 1
					stake_area += cv2.contourArea(cnt)

				# find blobs pertaining to each stake
				for stake in stakes_coords:
					# choose roi to be stake bounding rectangle
					roi = hsv[stake[0][0][1]: stake[0][1][1], stake[0][0][0]: stake[0][1][0]]

					# apply mask
					blob_mask = cv2.inRange(roi, lower_green, upper_green)

					# remove noise
					blob_mask= cv2.morphologyEx(blob_mask, cv2.MORPH_OPEN, kernel)

					# find contours in roi
					blob_contours = cv2.findContours(blob_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset = (stake[0][0][0], stake[0][0][1]))[1]

					# iterate through contours
					for cnt in blob_contours:
						# filter contours by area
						if min_contour_area <= cv2.contourArea(cnt) <= max_contour_area:
							# get contour coordinates
							rect = cv2.minAreaRect(cnt)
							box = np.array(cv2.boxPoints(rect), dtype = "int")

							# order points
							points  = orderPoints(box)

							# add blobs to stake list
							stake.append([points[0], points[2]])

							# increment blob counter
							blobs += 1
							blob_area += cv2.contourArea(cnt)

				# output boxes
				for stake in stakes_coords:
					for count, blob in enumerate(stake):
						if(count == 0):
							# draw stake
							cv2.rectangle(img_unmarked, tuple(blob[0]), tuple(blob[1]), (0,0,255),2)
						else:
							# draw blob
							cv2.rectangle(img_unmarked, tuple(blob[0]), tuple(blob[1]), (0,255,0), 2)

				# output results of template generation
				print("---------------------------------------")
				print("Template Generated Successfully\n")
				print("Number Of Stakes: %s" % stakes)
				print("Number Of Blobs: %s" % blobs)
				print("Average Stake Area: %s" % float(stake_area/stakes))
				print("Average Blob Area: %s" % float(blob_area/blobs))

				# show user image and confirm whether to save template
				img_unmarked = cv2.resize(img_unmarked, None, fx = 0.75, fy = 0.75)
				cv2.imshow("Template Overlay", img_unmarked)
				answer = tkMessageBox.askyesno("Question", "Would you like to save this template")

				# close windows and save if appropriate
				cv2.destroyAllWindows()

				if(answer):
					# search for template directory
					if(not os.path.isdir("./includes/templates")):
						os.mkdir("./includes/templates")

					# write image to directory
					path = os.getcwd() + "\\includes\\templates\\" + os.path.split(self.unmarkedTemplate)[1]
					cv2.imwrite(path, img_save)

					# create output string
					outputString = str(stakes_coords).replace("array(", "").replace(")", "")

					# save to config file
					self.config.set('Template Coordinates', name.get(), outputString)
					self.config.set('Template Images', name.get(), path)

					# update menu
					templateMenu['menu'].add_command(label = name.get(), command = tk._setit(templateMenuVar, name.get()))
					self.systemParameters["Templates"][name.get()] = outputString
					self.systemParameters["Templates_Options"].append(name.get())
					self.systemParameters["Template_Paths"][name.get()] = path
					self.systemParameters["Current_Template_Name"] = name.get()
					self.systemParameters["Current_Template_Coords"] = outputString
					self.systemParameters["Current_Template_Path"] = path
					templateMenuVar.set(name.get())

					print("Template Saved Successfully: %s \n" % name.get())
			
			# reopen other windows
			newWindow.deiconify()
			self.root.deiconify()

		# embedded function to get name for saved profile
		def getName():
			# get fields are filled in
			if all(v.get() != "" for v in entries) and templateMenuVar.get() != "Select Template":
				# ask for name
				name = tk.StringVar()
				nameWindow = tk.Toplevel(newWindow)
				nameWindow.configure(bg='#ffffff')
				self.centre_window(nameWindow, 300, 170)

				# widgets
				nameLabel = tk.Label(nameWindow,text="Name",bg='#ffffff',fg='#000000',font=("Calibri Light", 20))
				nameEntry = tk.Entry(nameWindow,font=("Calibri Light", 14),textvariable = name,width = 20)
				nameButton = tk.Button(nameWindow,text = "Save",bg='#ffffff',fg='#000000',command = lambda: nameWindow.destroy(),
				    width = 20,font=("Calibri Light", 14))

				# packing
				nameLabel.pack(pady = (15,5))
				nameEntry.pack(pady = 10)
				nameButton.pack(pady = 5)

				# wait until user inputs name
				self.root.wait_window(nameWindow)

				# if name was inputted
				if(name.get() != ""):
					# create output string
					outputString = "[" + str(self.systemParameters["Upper_Border"]) + "," + str(self.systemParameters["Lower_Border"]) + "," + str(self.systemParameters["Lower_Blob_Size"]) + \
						"," + str(self.systemParameters["Upper_Blob_Size"]) + "," + '"' + str(templateMenuVar.get()) + '"' + "," + str(self.systemParameters["Clip_Limit"]) + "," + str(self.systemParameters["Tile_Size"][0]) + \
						"," + str(self.systemParameters["Tile_Size"][1]) + "]"

					# add to config file
					self.config.set('Profiles', name.get(), outputString)

					# update menu
					self.profileMenu['menu'].add_command(label = name.get(), command = tk._setit(self.profileMenuVar, name.get()))
					self.systemParameters["Saved_Profiles"][name.get()] = outputString
					self.systemParameters["Profile_Options"].append(name.get())
					self.profileMenuVar.set(name.get())
									
				# close windows
				nameWindow.destroy()
				newWindow.destroy()
			else:
				tkMessageBox.showinfo("Error", "Not All Fields Populated")

		# create window
		newWindow = tk.Toplevel(self.root)
		newWindow.configure(bg='#ffffff')
		self.centre_window(newWindow, 500, 550)

		validateCommand = self.root.register(self.validate)

		# frames
		upperBorderFrame = tk.Frame(newWindow, bg='#ffffff')
		lowerBorderFrame = tk.Frame(newWindow, bg='#ffffff')
		lowerBlobFrame = tk.Frame(newWindow, bg='#ffffff')
		upperBlobFrame = tk.Frame(newWindow, bg='#ffffff')
		templateFrame = tk.Frame(newWindow, bg='#ffffff')
		clipLimitFrame = tk.Frame(newWindow, bg='#ffffff')
		tileSizeFrame = tk.Frame(newWindow, bg='#ffffff')
		buttonFrame = tk.Frame(newWindow, bg='#ffffff')

		# labels
		titleLabel = tk.Label(newWindow, text = "Preferences", bg = '#ffffff', fg = '#000000', font=("Calibri Light", 24))
		upperBorderLabel = tk.Label(upperBorderFrame, text = "Upper Border")
		lowerBorderLabel = tk.Label(lowerBorderFrame, text = "Lower Border")
		lowerBlobLabel = tk.Label(lowerBlobFrame, text = "Lower Blob Size")
		upperBlobLabel = tk.Label(upperBlobFrame, text = "Upper Blob Size")
		templateLabel = tk.Label(templateFrame, text = "Template")
		clipLimitLabel = tk.Label(clipLimitFrame, text = "Clip Limit")
		tileSizeLabel1 = tk.Label(tileSizeFrame, text = "Tile Size")
		tileSizeLabel2 = tk.Label(tileSizeFrame, text = "(")
		tileSizeLabel3 = tk.Label(tileSizeFrame, text = ",")
		tileSizeLabel4 = tk.Label(tileSizeFrame, text = ")")

		labels = [upperBorderLabel, lowerBorderLabel, lowerBlobLabel, upperBlobLabel, templateLabel, clipLimitLabel, tileSizeLabel1, tileSizeLabel2, tileSizeLabel3, tileSizeLabel4]

		for label in labels:
			label.config(bg = "#ffffff", fg = "#000000", font=("Calibri Light", 16))

		# entries
		upperBorderEntry = tk.Entry(upperBorderFrame, validatecommand =((validateCommand, '%P', "Upper_Border", -1)))
		lowerBorderEntry = tk.Entry(lowerBorderFrame, validatecommand =((validateCommand, '%P', "Lower_Border", -1)))
		lowerBlobEntry = tk.Entry(lowerBlobFrame, validatecommand =((validateCommand, '%P', "Lower_Blob_Size", -1)))
		upperBlobEntry = tk.Entry(upperBlobFrame, validatecommand =((validateCommand, '%P', "Upper_Blob_Size", -1)))
		clipLimitEntry = tk.Entry(clipLimitFrame, validatecommand =((validateCommand, '%P', "Clip_Limit", -1)))
		tileSizeEntry1 = tk.Entry(tileSizeFrame, validatecommand =((validateCommand, '%P', "Tile_Size", 0)))
		tileSizeEntry2 = tk.Entry(tileSizeFrame, validatecommand =((validateCommand, '%P', "Tile_Size", 1)))

		# set default values
		clipLimitEntry.insert(0,5)
		tileSizeEntry1.insert(0,8)
		tileSizeEntry2.insert(0,8)
		self.systemParameters["Clip_Limit"] = 5
		self.systemParameters["Tile_Size"] = [8,8]

		entries = [upperBorderEntry, lowerBorderEntry, lowerBlobEntry, upperBlobEntry, clipLimitEntry, tileSizeEntry1, tileSizeEntry2]

		for entry in entries:
			entry.config(validate = "key", font=("Calibri Light", 14), width = 6)

		# drop down
		templateMenuVar = tk.StringVar(newWindow)
		templateMenuVar.set('Select Template')
		templateMenu = tk.OptionMenu(templateFrame, templateMenuVar, 'Select Template', *self.systemParameters["Templates_Options"])
		templateMenu.config(font=("Calibri Light", 14), bg='#ffffff', width = 15)

		# menu bar
		profileMenuBar = tk.Menu(newWindow)
		profileFileMenu = tk.Menu(profileMenuBar, tearoff=0)
		profileFileMenu.add_command(label = "Generate Template", command = lambda: createTemplate())
		profileFileMenu.add_command(label = "Remove Template", command = lambda: removeTemplate())
		profileMenuBar.add_cascade(label = "File", menu = profileFileMenu)
		newWindow.config(menu = profileMenuBar)

        # button
		createProfileButton = tk.Button(buttonFrame, text = "Create Profile", command = lambda: getName(), bg = '#ffffff',
			fg = '#000000', font=("Calibri Light", 15), width = 17)
		#createTemplateButton = tk.Button(buttonFrame, text = "Create Template", command = lambda: createTemplate(), bg = '#ffffff',
		#	fg = '#000000', font=("Calibri Light", 15), width = 17)
        
		# packing
		titleLabel.pack(pady = 20)
		upperBorderFrame.pack(pady = 10)
		upperBorderLabel.pack(side = tk.LEFT, padx = 10)
		upperBorderEntry.pack(side = tk.LEFT, padx = 10)

		lowerBorderFrame.pack(pady = 10)
		lowerBorderLabel.pack(side = tk.LEFT, padx = 10)
		lowerBorderEntry.pack(side = tk.LEFT, padx = 10)

		lowerBlobFrame.pack(pady = 10)
		lowerBlobLabel.pack(side = tk.LEFT, padx = 10)
		lowerBlobEntry.pack(side = tk.LEFT, padx = 10)   

		upperBlobFrame.pack(pady = 10)
		upperBlobLabel.pack(side = tk.LEFT, padx = 10)
		upperBlobEntry.pack(side = tk.LEFT, padx = 10)   

		templateFrame.pack(pady = 10)
		templateLabel.pack(side = tk.LEFT, padx = 10)
		templateMenu.pack(side = tk.LEFT, padx = 10)

		clipLimitFrame.pack(pady = 10)
		clipLimitLabel.pack(side = tk.LEFT, padx = 10)
		clipLimitEntry.pack(side = tk.LEFT, padx = 10)    	

		tileSizeFrame.pack(pady = 10)
		tileSizeLabel1.pack(side = tk.LEFT, padx = 10)
		tileSizeLabel2.pack(side = tk.LEFT, padx = (5,0))    
		tileSizeEntry1.pack(side = tk.LEFT)	
		tileSizeLabel3.pack(side = tk.LEFT, padx = 0)    
		tileSizeEntry2.pack(side = tk.LEFT)	
		tileSizeLabel4.pack(side = tk.LEFT, padx = (0,5))    

		buttonFrame.pack(pady = 20)
		#createTemplateButton.pack(side = tk.LEFT, padx = (20, 5))
		createProfileButton.pack(side = tk.LEFT, padx = (5, 20))

		# wait until user inputs name
		self.root.wait_window(newWindow)

    # ---------------------------------------------------------------------------------
    # Function to remove a preferences profile
    # ---------------------------------------------------------------------------------

    def removeProfile(self):
		# function to close window
		def closing():
			# close window
			newWindow.destroy()

			# remove selected option from menu
			if(removeprofileMenuVar.get() != 'Select Profile to Remove'):
				self.config.remove_option("Profiles", removeprofileMenuVar.get())
				self.systemParameters["Profile_Options"].remove(removeprofileMenuVar.get())
				self.systemParameters["Saved_Profiles"].pop(removeprofileMenuVar.get())

				# update menu
				self.profileMenuVar.set('Select Profile')
				self.profileMenu['menu'].delete(0, 'end')

				for option in self.systemParameters["Profile_Options"]:
					self.profileMenu['menu'].add_command(label = option, command = tk._setit(self.profileMenuVar, option))

		# create window 
		newWindow = tk.Toplevel(self.root)
		newWindow.configure(bg='#ffffff')
		self.centre_window(newWindow, 300, 170)

		# labels
		nameLabel = tk.Label(newWindow,text="Name",bg='#ffffff',fg='#000000',font=("Calibri Light", 20))

		# drop down menu used to select profile to delete
		removeprofileMenuVar = tk.StringVar(self.root)
		removeprofileMenuVar.set('Select Profile to Remove')
		removeMenu = tk.OptionMenu(newWindow, removeprofileMenuVar, 'Select Profile to Remove', *self.systemParameters["Profile_Options"])
		removeMenu.config(font=("Calibri Light", 13))
		removeMenu.config(bg='#ffffff')

		# button
		nameButton = tk.Button(newWindow,text = "Save",bg='#ffffff',fg='#000000',command = lambda: closing(),
		    width = 20,font=("Calibri Light", 14))

		# packing
		nameLabel.pack(pady = (20,5))
		removeMenu.pack(pady = 10)
		nameButton.pack(pady = 5)

		# wait until user inputs name
		self.root.wait_window(newWindow)

	# ---------------------------------------------------------------------------------
    # Function to preview preferences profile
    # ---------------------------------------------------------------------------------

    def previewProfile(self):
    	if(self.profileMenuVar.get() != 'Select Profile'):
			# create window
			newWindow = tk.Toplevel(self.root)
			newWindow.configure(bg='#ffffff')
			self.centre_window(newWindow, 450, 500)

			# labels
			titleLabel = tk.Label(newWindow, text = str(self.profileMenuVar.get()), bg = '#ffffff', fg = '#000000', font=("Calibri Light", 24))
			upperBorderLabel = tk.Label(newWindow, text = "Upper Border: " + str(self.systemParameters["Upper_Border"]))
			lowerBorderLabel = tk.Label(newWindow, text = "Lower Border: " + str(self.systemParameters["Lower_Border"]))
			lowerBlobLabel = tk.Label(newWindow, text = "Lower Blob Size: " + str(self.systemParameters["Lower_Blob_Size"]))
			upperBlobLabel = tk.Label(newWindow, text = "Upper Blob Size: " + str(self.systemParameters["Upper_Blob_Size"]))
			templateLabel = tk.Label(newWindow, text = "Template: " + str(self.systemParameters["Current_Template_Name"]))
			clipLimitLabel = tk.Label(newWindow, text = "Clip Limit: " + str(self.systemParameters["Clip_Limit"]))
			tileSizeLabel = tk.Label(newWindow, text = "Tile Size: " + str(self.systemParameters["Tile_Size"]))

			labels = [upperBorderLabel, lowerBorderLabel, lowerBlobLabel, upperBlobLabel, templateLabel, clipLimitLabel, tileSizeLabel]

			for label in labels:
				label.config(bg = "#ffffff", fg = "#000000", font=("Calibri Light", 16))
	        
			# packing
			titleLabel.pack(pady = 20)
			upperBorderLabel.pack(pady = 10)
			lowerBorderLabel.pack(pady = 10)
			lowerBlobLabel.pack(pady = 10)
			upperBlobLabel.pack(pady = 10)
			templateLabel.pack(pady = 10)
			clipLimitLabel.pack(pady = 10)
			tileSizeLabel.pack(pady = 10)  

			# wait until user closes window
			self.root.wait_window(newWindow)
    	else:
			tkMessageBox.showinfo("Error", "Please Select a Profile under Settings to Preview")


    # ---------------------------------------------------------------------------------
    # Function to run HSV range preview
    # ---------------------------------------------------------------------------------

    def runPreview(self):
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
        newWindow.configure(background='#ffffff')
        newWindow.withdraw()
        filename = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))   

        if(filename != ""):
	        newWindow.deiconify()
	        newWindow.geometry("500x650") # window size of 500 x 650

	        # create widgets
	        HSVSlidersLabel = tk.Label(newWindow,text="HSV Sliders",bg='#ffffff',fg='#000000',font=("Calibri Light", 24))
	        LowerLabel = tk.Label(newWindow,text="Lower",bg='#ffffff',fg='#000000',font=("Calibri Light", 15))
	        UpperLabel = tk.Label(newWindow,text="Upper",bg='#ffffff',fg='#000000',font=("Calibri Light", 15))         

	        # button
	        savetoMain = tk.Button(newWindow, text = "Save Values",bg = '#ffffff',fg = '#000000',command = lambda: saveValuestoMain(),
	            width = 20,font=("Calibri Light", 15))

	        # frames
	        Frame1 = tk.Frame(newWindow, background='#ffffff')    
	        Frame2 = tk.Frame(newWindow, background='#ffffff')    
	        Frame3 = tk.Frame(newWindow, background='#ffffff')    
	        Frame4 = tk.Frame(newWindow, background='#ffffff')    
	        Frame5 = tk.Frame(newWindow, background='#ffffff')    
	        Frame6 = tk.Frame(newWindow, background='#ffffff')    

	        # H, S, and V Labels
	        H1Lower = tk.Label(Frame1, text = "H", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        S1Lower = tk.Label(Frame2, text = "S", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        V1Lower = tk.Label(Frame3, text = "V", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        H1Upper = tk.Label(Frame4, text = "H", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        S1Upper = tk.Label(Frame5, text = "S", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        V1Upper = tk.Label(Frame6, text = "V", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  

	        # sliders
	        H1Slider = tk.Scale(Frame1, from_=0, to=180, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        S1Slider = tk.Scale(Frame2, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        V1Slider = tk.Scale(Frame3, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        H2Slider = tk.Scale(Frame4, from_=0, to=180, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        S2Slider = tk.Scale(Frame5, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        V2Slider = tk.Scale(Frame6, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)

			# second range H, S, and V Labels
	        H2Lower = tk.Label(Frame1, text = "H", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        S2Lower = tk.Label(Frame2, text = "S", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        V2Lower = tk.Label(Frame3, text = "V", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        H2Upper = tk.Label(Frame4, text = "H", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        S2Upper = tk.Label(Frame5, text = "S", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  
	        V2Upper = tk.Label(Frame6, text = "V", background= '#ffffff', foreground='#000000', font=("Calibri Light", 14))  

	        # second range sliders
	        H3Slider = tk.Scale(Frame1, from_=0, to=180, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        S3Slider = tk.Scale(Frame2, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        V3Slider = tk.Scale(Frame3, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        H4Slider = tk.Scale(Frame4, from_=0, to=180, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        S4Slider = tk.Scale(Frame5, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)
	        V4Slider = tk.Scale(Frame6, from_=0, to=255, orient = 'horizontal', background= '#ffffff', length = 350, font = ("Calibri Light", 14), command = updateValues)

	        # list containing widgets for second HSV range
	        secondRangeWidgets = [H2Lower, S2Lower, V2Lower, H2Upper, S2Upper, V2Upper, H3Slider, S3Slider, V3Slider, \
	        	H4Slider, S4Slider, V4Slider]

	        # checkbox
	        previewsecondHSVFlag = tk.IntVar()
	        previewCheckBox = tk.Checkbutton(
	        	newWindow,
	            text="Second HSV Range",
	            variable= previewsecondHSVFlag,
	            background='#ffffff',
	            foreground='#000000',
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