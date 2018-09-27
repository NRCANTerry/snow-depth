# import necessary modules
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import ttk
import numpy as np
import ConfigParser
import ast
import cv2
import sys
import os
import string

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
        	"Lower_HSV_1": np.array([5,10,20]),
        	"Upper_HSV_1": np.array([0,0,0]),
        	"Lower_HSV_2": np.array([0,0,0]),
        	"Upper_HSV_2": np.array([0,0,0]),
        	"Upper_Border": 0,
        	"Lower_Border": 0,
        	"Lower_Blob_Size": 0,
        	"Upper_Blob_Size": 0,
        	"Clip_Limit": 0,
        	"Tile_Size": (0,0),
        	"Saved_Colours": dict(),
        	"Saved_Profiles": dict(),
        	"Colour_Options": list(),
        	"Profile_Options": list(),
        	"Window_Closed": False
        }

        # ConfigParser object
        self.config = ConfigParser.ConfigParser()

        # open preferences file
        self.systemParameters["Saved_Colours"], self.systemParameters["Saved_Profiles"] = self.getPreferences()
        self.systemParameters["Colour_Options"] = list(self.systemParameters["Saved_Colours"].keys())
        self.systemParameters["Profile_Options"] = list(self.systemParameters["Saved_Profiles"].keys())

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

        validateCommand = self.root.register(self.validateHSV)

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

        # ---------------------------------------------------------------------------------
        # Drop Down Menus
        # ---------------------------------------------------------------------------------

        self.colourMenuVar = tk.StringVar(self.root)
        self.colourMenuVar.set('Select HSV Range')
        self.colourMenu = tk.OptionMenu(self.root, self.colourMenuVar, 'Select HSV Range', *self.systemParameters["Colour_Options"])
        self.colourMenu.config(font=("Calibri Light", 13), bg='#ffffff')
        self.colourMenuVar.trace('w', self.change_dropdown)

        self.profileMenuVar = tk.StringVar(self.root)
        self.profileMenuVar.set('Select Profile')
        self.profileMenu = tk.OptionMenu(self.root, self.profileMenuVar, 'Select Profile', *self.systemParameters["Profile_Options"])
        self.profileMenu.config(font=("Calibri Light", 13), bg='#ffffff')
      	# include trace

        # ---------------------------------------------------------------------------------
        # Top Menu
        # ---------------------------------------------------------------------------------

        # create menu bar
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)

        # add commands
        self.filemenu.add_command(label = "Save HSV Range", command = lambda: self.saveRanges())
        self.filemenu.add_command(label = "Remove HSV Range", command = lambda: self.removeRanges())
        self.filemenu.add_command(label = "Load Preview Tool", command = lambda: self.runPreview())
        self.filemenu.add_separator()
        self.filemenu.add_command(label = "Restart", command = lambda: self.restart())
        self.filemenu.add_command(label = "Exit", command = lambda: self.on_closing())
        self.menubar.add_cascade(label = "File", menu = self.filemenu)

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
        self.runButton.pack(pady = (10,30))

    # ---------------------------------------------------------------------------------
    # Functions
    # ---------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # Validate method for HSV text entry
    # ---------------------------------------------------------------------------------

    def validateHSV(self, new_text, entry_field, index):
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

    # ---------------------------------------------------------------------------------
    # Function to confirm that required fields are filled in
    # ---------------------------------------------------------------------------------

    def fieldsFilled(self, directory = True):
		if(directory):
			return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
				and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
				and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
				and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
				and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1) and self.directory != "")
		else:
			return (self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" \
				and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" and self.entryUpperV1.get() != "" \
				and ((self.secondHSVFlag.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" \
				and self.entryLowerV2.get() != "" and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" \
				and self.entryUpperV2.get() != "") or self.secondHSVFlag.get() != 1))

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
            		self.systemParameters["Lower_HSV_2"], self.systemParameters["Upper_HSV_2"]

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
        # else read in existing file
        else:
            self.config.read('./preferences.cfg')

        # load in preferences
        return dict(self.config.items('HSV Ranges')), dict(self.config.items('Profiles'))

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
    # Function to update values on menu selection
    # ---------------------------------------------------------------------------------

    def change_dropdown(self, *args):
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
            newWindow.configure(background='#ffffff')
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
    	# ask for name
		name = tk.StringVar()
		newWindow = tk.Toplevel(self.root)
		newWindow.configure(background='#ffffff')
		newWindow.geometry("350x180") # window size in pixels

		# labels
		nameLabel = tk.Label(
		    newWindow,
		    text="Name",
		    background='#ffffff',
		    foreground='#000000',
		    font=("Calibri Light", 24))

		# drop down menu used to select profile to delete
		removecolourMenuVar = tk.StringVar(self.root)
		removecolourMenuVar.set('Select Profile to Remove')
		removeMenu = tk.OptionMenu(newWindow, removecolourMenuVar, 'Select Profile to Remove', *self.systemParameters["Colour_Options"])
		removeMenu.config(font=("Calibri Light", 13))
		removeMenu.config(background='#ffffff')

		# button
		nameButton = tk.Button(
		    newWindow,
		    text = "Save",
		    background='#ffffff',
		    foreground='#000000',
		    command = lambda: newWindow.destroy(),
		    width = 20,
		    font=("Calibri Light", 14))

		# packing
		nameLabel.pack(pady = (20,5))
		removeMenu.pack(pady = 10)
		nameButton.pack(pady = 5)

		# wait until user inputs name
		self.root.wait_window(newWindow)

		# remove selected option from menu
		if(removecolourMenuVar.get() != 'Select Profile to Remove'):
			self.config.remove_option("HSV Ranges", removecolourMenuVar.get())
			self.systemParameters["Colour_Options"].remove(removecolourMenuVar.get())
			self.systemParameters["Saved_Colours"].pop(removecolourMenuVar.get())

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
	        HSVSlidersLabel = tk.Label(
	            newWindow,
	            text="HSV Sliders",
	            background='#ffffff',
	            foreground='#000000',
	            font=("Calibri Light", 24))

	        LowerLabel = tk.Label(
	            newWindow,
	            text="Lower",
	            background='#ffffff',
	            foreground='#000000',
	            font=("Calibri Light", 15))

	        UpperLabel = tk.Label(
	            newWindow,
	            text="Upper",
	            background='#ffffff',
	            foreground='#000000',
	            font=("Calibri Light", 15))         

	        # button
	        savetoMain = tk.Button(
	            newWindow, 
	            text = "Save Values",
	            background = '#ffffff',
	            foreground = '#000000',
	            command = lambda: saveValuestoMain(),
	            width = 20,
	            font=("Calibri Light", 15))

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