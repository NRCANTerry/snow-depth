# import necessary modules
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import ttk
import numpy as np
import ConfigParser
import ast
import cv2

class GUI:
    def __init__(self):
        # open window
        self.root = tk.Tk()
        self.root.configure(background='#ffffff')
        self.root.title("Generate Training Images")
        self.root.geometry("600x650") # window size of 1000 x 700

        # ---------------------------------------------------------------------------------
        # Variables
        # ---------------------------------------------------------------------------------

        self.directory = ""
        self.lower_hsv1 = np.array([0,0,0])
        self.upper_hsv1 = np.array([0,0,0])
        self.lower_hsv2 = np.array([0,0,0])
        self.upper_hsv2 = np.array([0,0,0])
        self.lowerBorder = 0
        self.upperBorder = 0
        self.windowClosed = False

        # ConfigParser object
        self.config = ConfigParser.ConfigParser()

        # open preferences file
        self.saved_Colours = self.getPreferences()
        self.colourOptions = list(self.saved_Colours.keys())

        # window closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---------------------------------------------------------------------------------
        # Labels
        # ---------------------------------------------------------------------------------

        # step 1 label
        self.label1 = tk.Label(
            self.root,
            text="Image Folder",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 24))

        # image folder path label
        self.pathLabel = tk.Label(
            self.root,
            text="No Directory Selected",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        # step 2 label
        self.label2 = tk.Label(
            self.root,
            text="HSV Range",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 24))

        # frame containing HSV range widgets
        self.range1Frame = tk.Frame(self.root, background='#ffffff')

        # h, s, and v labels for lower range 1
        self.lowerH1 = tk.Label(
            self.range1Frame,
            text="H",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.lowerS1 = tk.Label(
            self.range1Frame,
            text="S",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.lowerV1 = tk.Label(
            self.range1Frame,
            text="V",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.arrow1 = tk.Label(
            self.range1Frame,
            text="-->",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        # h, s, and v labels for upper range 1
        self.upperH1 = tk.Label(
            self.range1Frame,
            text="H",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.upperS1 = tk.Label(
            self.range1Frame,
            text="S",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.upperV1 = tk.Label(
            self.range1Frame,
            text="V",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        # frame containing HSV range widgets
        self.range2Frame = tk.Frame(self.root, background='#ffffff')

        # h, s, and v labels for lower range 2
        self.lowerH2 = tk.Label(
            self.range2Frame,
            text="H",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        self.lowerS2 = tk.Label(
            self.range2Frame,
            text="S",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        self.lowerV2 = tk.Label(
            self.range2Frame,
            text="V",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        self.arrow2 = tk.Label(
            self.range2Frame,
            text="-->",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        # h, s, and v labels for upper range 2
        self.upperH2 = tk.Label(
            self.range2Frame,
            text="H",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        self.upperS2 = tk.Label(
            self.range2Frame,
            text="S",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        self.upperV2 = tk.Label(
            self.range2Frame,
            text="V",
            background='#ffffff',
            foreground='#D3D3D3',
            font=("Calibri Light", 14))

        # list of labels for second HSV range
        self.labels2 = [self.lowerH2, self.lowerS2, self.lowerV2, self.arrow2, self.upperH2, self.upperS2, self.upperV2]

        # step 3 label
        self.label3 = tk.Label(
            self.root,
            text="Borders",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 24))

        # frame containing default menu widgets
        self.defaultFrame = tk.Frame(self.root, background='#ffffff')

        # frame containing border widgets
        self.borderFrame1 = tk.Frame(self.root, background='#ffffff')
        self.borderFrame2 = tk.Frame(self.root, background='#ffffff')

        # border labels
        self.upperBorder = tk.Label(
            self.borderFrame1,
            text=" Upper",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.upperBorderPixel = tk.Label(
            self.borderFrame1,
            text="px",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))        

        self.lowerBorder = tk.Label(
            self.borderFrame2,
            text="Lower",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))

        self.lowerBorderPixel = tk.Label(
            self.borderFrame2,
            text="px",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 14))            

        # ---------------------------------------------------------------------------------
        # Buttons
        # ---------------------------------------------------------------------------------

        # choose directory button
        self.directoryButton = tk.Button(
            self.root,
            text="Select",
            background='#ffffff',
            foreground='#000000',
            command=lambda: self.selectDirectory(),
            width = 17,
            font=("Calibri Light", 14))

        self.buttonFrame = tk.Frame(self.root, background='#ffffff')

        # execute button
        self.runButton = tk.Button(
            self.buttonFrame,
            text="Generate",
            background='#ffffff',
            foreground='#000000',
            command=lambda: self.saveValues(),
            width = 17,
            font=("Calibri Light", 14))

        # preview HSV button
        self.previewButton = tk.Button(
            self.buttonFrame,
            text=" Preview",
            background='#ffffff',
            foreground='#000000',
            command=lambda: self.runPreview(),
            width = 17,
            font=("Calibri Light", 14))

       # save HSV range button
        self.saveRange = tk.Button(
            self.buttonFrame,
            text=" Save",
            background='#ffffff',
            foreground='#000000',
            command=lambda: self.saveRanges(),
            width = 17,
            font=("Calibri Light", 14))

        # ---------------------------------------------------------------------------------
        # Entry
        # ---------------------------------------------------------------------------------

        validateCommand = self.root.register(self.validate)  # we have to wrap the command

        # h, s, and v entries lower range 1
        self.entryLowerH1 = tk.Entry(self.range1Frame, validate="key", validatecommand=(validateCommand, '%P', 'lh1'),
            font=("Calibri Light", 13), width = 4)
        self.entryLowerS1 = tk.Entry(self.range1Frame, validate="key", validatecommand=(validateCommand, '%P', 'ls1'),
            font=("Calibri Light", 13), width = 4)
        self.entryLowerV1 = tk.Entry(self.range1Frame, validate="key", validatecommand=(validateCommand, '%P', 'lv1'),
            font=("Calibri Light", 13), width = 4)

        # h, s, and v entries for upper range 1
        self.entryUpperH1 = tk.Entry(self.range1Frame, validate="key", validatecommand=(validateCommand, '%P', 'uh1'),
            font=("Calibri Light", 13), width = 4)
        self.entryUpperS1 = tk.Entry(self.range1Frame, validate="key", validatecommand=(validateCommand, '%P', 'us1'),
            font=("Calibri Light", 13), width = 4)
        self.entryUpperV1 = tk.Entry(self.range1Frame, validate="key", validatecommand=(validateCommand, '%P', 'uv1'),
            font=("Calibri Light", 13), width = 4)

        # list of entries
        self.entries1 = [self.entryLowerH1, self.entryLowerS1, self.entryLowerV1, self.entryUpperH1, self.entryUpperS1, self.entryUpperV1]

        # h, s, and v entries lower range 2
        self.entryLowerH2 = tk.Entry(self.range2Frame, validate="key", validatecommand=(validateCommand, '%P', 'lh2'),
            font=("Calibri Light", 13), width = 4, state = 'disabled')
        self.entryLowerS2 = tk.Entry(self.range2Frame, validate="key", validatecommand=(validateCommand, '%P', 'ls2'),
            font=("Calibri Light", 13), width = 4, state = 'disabled')
        self.entryLowerV2 = tk.Entry(self.range2Frame, validate="key", validatecommand=(validateCommand, '%P', 'lv2'),
            font=("Calibri Light", 13), width = 4, state = 'disabled')

        # h, s, and v entries for upper range 2
        self.entryUpperH2 = tk.Entry(self.range2Frame, validate="key", validatecommand=(validateCommand, '%P', 'uh2'),
            font=("Calibri Light", 13), width = 4, state = 'disabled')
        self.entryUpperS2 = tk.Entry(self.range2Frame, validate="key", validatecommand=(validateCommand, '%P', 'us2'),
            font=("Calibri Light", 13), width = 4, state = 'disabled')
        self.entryUpperV2 = tk.Entry(self.range2Frame, validate="key", validatecommand=(validateCommand, '%P', 'uv2'),
            font=("Calibri Light", 13), width = 4, state = 'disabled')

        # list of entries
        self.entries2 = [self.entryLowerH2, self.entryLowerS2, self.entryLowerV2, self.entryUpperH2, self.entryUpperS2, self.entryUpperV2]

        # Upper and lower border entry fields
        self.entryUpper = tk.Entry(self.borderFrame1, validate="key", validatecommand=(validateCommand, '%P', 'upper'),
            width = 7, font=("Calibri Light", 13))
        self.entryLower = tk.Entry(self.borderFrame2, validate="key", validatecommand=(validateCommand, '%P', 'lower'),
            width = 7, font=("Calibri Light", 13))

        # ---------------------------------------------------------------------------------
        # Checkbox
        # ---------------------------------------------------------------------------------

        self.secondHSV = tk.IntVar()
        self.checkBox = tk.Checkbutton(
            self.root,
            text="Second HSV Range",
            variable=self.secondHSV,
            background='#ffffff',
            foreground='#000000',
            command=lambda: self.updateSelections(),
            font=("Calibri Light", 14))

        # ---------------------------------------------------------------------------------
        # Drop Down Menu
        # ---------------------------------------------------------------------------------

        self.menuVar = tk.StringVar(self.root)
        self.menuVar.set('Select Saved Profile')
        self.menu = tk.OptionMenu(self.defaultFrame, self.menuVar, 'Select Saved Profile', *self.colourOptions)
        self.menu.config(font=("Calibri Light", 13))
        self.menu.config(background='#ffffff')
        self.menuVar.trace('w', self.change_dropdown)

        # ---------------------------------------------------------------------------------
        # Packing
        # ---------------------------------------------------------------------------------

        self.label1.pack(pady = (30,5))
        self.pathLabel.pack()
        self.directoryButton.pack(pady = 10)
        self.label2.pack(pady = (20,5))

        # HSV lower range 1
        self.range1Frame.pack(pady = 5)
        self.lowerH1.pack(side = tk.LEFT, padx = (0,5))
        self.entryLowerH1.pack(side = tk.LEFT, padx = 5)
        self.lowerS1.pack(side = tk.LEFT, padx = 5)
        self.entryLowerS1.pack(side = tk.LEFT, padx = 5)
        self.lowerV1.pack(side = tk.LEFT, padx = 5)
        self.entryLowerV1.pack(side = tk.LEFT, padx = (5,0))
        self.arrow1.pack(side = tk.LEFT, padx = 20)

        # HSV upper range 1
        self.upperH1.pack(side = tk.LEFT, padx = (0,5))
        self.entryUpperH1.pack(side = tk.LEFT, padx = 5)
        self.upperS1.pack(side = tk.LEFT, padx = 5)
        self.entryUpperS1.pack(side = tk.LEFT, padx = 5)
        self.upperV1.pack(side = tk.LEFT, padx = 5)
        self.entryUpperV1.pack(side = tk.LEFT, padx = (5,0))
        self.checkBox.pack(pady = 10)

        # HSV lower range 2
        self.range2Frame.pack(pady = 5)
        self.lowerH2.pack(side = tk.LEFT, padx = (0,5))
        self.entryLowerH2.pack(side = tk.LEFT, padx = 5)
        self.lowerS2.pack(side = tk.LEFT, padx = 5)
        self.entryLowerS2.pack(side = tk.LEFT, padx = 5)
        self.lowerV2.pack(side = tk.LEFT, padx = 5)
        self.entryLowerV2.pack(side = tk.LEFT, padx = (5,0))
        self.arrow2.pack(side = tk.LEFT, padx = 20)

        # HSV upper range 2
        self.upperH2.pack(side = tk.LEFT, padx = (0,5))
        self.entryUpperH2.pack(side = tk.LEFT, padx = 5)
        self.upperS2.pack(side = tk.LEFT, padx = 5)
        self.entryUpperS2.pack(side = tk.LEFT, padx = 5)
        self.upperV2.pack(side = tk.LEFT, padx = 5)
        self.entryUpperV2.pack(side = tk.LEFT, padx = (5,0))

        # drop down menu
        self.defaultFrame.pack(pady = (20,0))
        self.menu.pack(side = tk.LEFT, padx = 5)

        # border frame packing
        self.label3.pack(pady = (20,5))
        self.borderFrame1.pack(pady = 5)
        self.upperBorder.pack(side = tk.LEFT, padx = (0,5))
        self.entryUpper.pack(side = tk.LEFT, padx = 5)
        self.upperBorderPixel.pack(side = tk.LEFT, padx = (5,0))
        self.borderFrame2.pack(pady = 5)
        self.lowerBorder.pack(side = tk.LEFT, padx = 5)
        self.entryLower.pack(side = tk.LEFT, padx = 5)
        self.lowerBorderPixel.pack(side = tk.LEFT, padx = (5,0))

        # button packing
        self.buttonFrame.pack(pady = 20)
        self.saveRange.pack(side = tk.LEFT, padx = 10)
        self.runButton.pack(side = tk.LEFT, padx = 10)
        self.previewButton.pack(side = tk.LEFT, padx = 10)

        self.root.mainloop()

    # ---------------------------------------------------------------------------------
    # Functions
    # ---------------------------------------------------------------------------------

    # validate method for text entry
    def validate(self, new_text, entry_field):
        if not new_text:  # the field is being cleared
            if (entry_field == "upper"):
                self.upperBorder = 0
            elif (entry_field == "lower"):
                self.lowerBorder = 0

            elif(entry_field == "lh1"):
                self.lower_hsv1[0] = 0
            elif(entry_field == "ls1"):
                self.lower_hsv1[1] = 0
            elif(entry_field == "lv1"):
                self.lower_hsv1[2] = 0

            elif(entry_field == "uh1"):
                self.upper_hsv1[0] = 0
            elif(entry_field == "us1"):
                self.upper_hsv1[1] = 0
            elif(entry_field == "uv1"):
                self.upper_hsv1[2] = 0        

            elif(entry_field == "lh2"):
                self.lower_hsv2[0] = 0
            elif(entry_field == "ls2"):
                self.lower_hsv2[1] = 0
            elif(entry_field == "lv2"):
                self.lower_hsv2[2] = 0

            elif(entry_field == "uh2"):
                self.upper_hsv2[0] = 0
            elif(entry_field == "us2"):
                self.upper_hsv2[1] = 0
            elif(entry_field == "uv2"):
                self.upper_hsv2[2] = 0         

        try:
            if (entry_field == "upper"):
                if (new_text == ""):
                    self.upperBorder = 0
                else:
                    self.upperBorder = int(new_text)
            elif (entry_field == "lower"):
                if (new_text == ""):
                    self.lowerBorder = 0
                else:
                    self.lowerBorder = int(new_text)

            elif(entry_field == "lh1"):
                if (new_text == ""):
                    self.lower_hsv1[0] = 0
                else:
                    self.lower_hsv1[0] = int(new_text)
            elif(entry_field == "ls1"):
                if (new_text == ""):
                    self.lower_hsv1[1] = 0
                else:
                    self.lower_hsv1[1] = int(new_text)
            elif(entry_field == "lv1"):
                if (new_text == ""):
                    self.lower_hsv1[2] = 0
                else:
                    self.lower_hsv1[2] = int(new_text)

            elif(entry_field == "uh1"):
                if (new_text == ""):
                    self.upper_hsv1[0] = 0
                else:
                    self.upper_hsv1[0] = int(new_text)
            elif(entry_field == "us1"):
                if (new_text == ""):
                    self.upper_hsv1[1] = 0
                else:
                    self.upper_hsv1[1] = int(new_text)
            elif(entry_field == "uv1"):
                if (new_text == ""):
                    self.upper_hsv1[2] = 0
                else:
                    self.upper_hsv1[2] = int(new_text)      

            elif(entry_field == "lh2"):
                if (new_text == ""):
                    self.lower_hsv2[0] = 0
                else:
                    self.lower_hsv2[0] = int(new_text)
            elif(entry_field == "ls2"):
                if (new_text == ""):
                    self.lower_hsv2[1] = 0
                else:
                    self.lower_hsv2[1] = int(new_text)
            elif(entry_field == "lv2"):
                if (new_text == ""):
                    self.lower_hsv2[2] = 0
                else:
                    self.lower_hsv2[2] = int(new_text)

            elif(entry_field == "uh2"):
                if (new_text == ""):
                    self.upper_hsv2[0] = 0
                else:
                    self.upper_hsv2[0] = int(new_text)
            elif(entry_field == "us2"):
                if (new_text == ""):
                    self.upper_hsv2[1] = 0
                else:
                    self.upper_hsv2[1] = int(new_text)
            elif(entry_field == "uv2"):
                if (new_text == ""):
                    self.upper_hsv2[2] = 0
                else:
                    self.upper_hsv2[2] = int(new_text)  

            return True

        except ValueError:
            return False

    # function allow selection of directory/file where images are stored
    def selectDirectory(self):
        # open directory selector
        dirname = tkFileDialog.askdirectory(parent=self.root, initialdir="/", title='Select Directory')

        # if new directory selected, update label
        if (len(dirname) > 0):
            self.pathLabel.config(text=dirname)
            self.directory = str(dirname)

    def saveValues(self):
        # if second HSV range is not selected
        if (self.secondHSV.get() != 1):
            # make both ranges equal
            self.lower_hsv2 = self.lower_hsv1
            self.upper_hsv2 = self.upper_hsv1

        # if required fields are filled in
        if(self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" \
            and self.entryUpperV1.get() != "" and ((self.secondHSV.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" and self.entryLowerV2.get() != "" \
                and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" and self.entryUpperV2.get() != "") or self.secondHSV.get() != 1) and self.directory != ""):
            # close window and return to other program
            self.windowClosed = True

            # write preferences to file
            with open('./preferences.cfg', 'wb') as configfile:
                self.config.write(configfile)

            self.root.destroy()

        # else show error
        else:
            tkMessageBox.showinfo("Error", "Not All Fields Populated")

    # function to return parameters
    def getValues(self):
        if(self.windowClosed):
            return self.directory, self.lower_hsv1, self.upper_hsv1, self.lower_hsv2, self.upper_hsv2, \
                   self.upperBorder, self.lowerBorder
        else:
            return False

    # function to update appearance of second set of HSV selections
    # based on status of checkbox
    def updateSelections(self):
        if (self.secondHSV.get() == 1): 
            # update labels
            for label in self.labels2:
                label.config(fg ='#000000')
            # update fields
            for field in self.entries2:
                field.config(state = "normal")
        else:
            # update labels
            for label in self.labels2:
                label.config(fg ='#D3D3D3')            
            # update fields
            for field in self.entries2:
                field.delete(0, tk.END)
                field.config(state = "disabled")      

    # function to launch HSV preview tool
    def launchPreview(self):
        HSVPreview()

    # function to fetch preferences
    def getPreferences(self):

        # if no preferences file present, create one
        if(str(self.config.read('./preferences.cfg')) == "[]"):
            self.config.add_section('HSV Ranges')
        # else read in existing file
        else:
            self.config.read('./preferences.cfg')

        # load in colour preferences
        return (dict(self.config.items('HSV Ranges')))

    # function run when user closes window
    def on_closing(self):
        # write preferences to file
        with open('./preferences.cfg', 'wb') as configfile:
            self.config.write(configfile)

        # close window
        self.root.destroy()

    # function to update values on menu selection
    def change_dropdown(self, *args):

        if(self.menuVar.get() != 'Select Saved Profile'):        
            # create list from value stored in preferences
            hsvList = ast.literal_eval(self.saved_Colours[self.menuVar.get()])

            # determine whether stored value is 1 range or 2
            length = len(hsvList)

            # update entries
            for field in self.entries1:
                field.delete(0, tk.END)
            for field in self.entries2:
                field.delete(0, tk.END)

            # single HSV range
            self.lower_hsv1 = np.asarray(hsvList[0])
            self.upper_hsv1 = np.asarray(hsvList[1])

            # update entries
            for count, field in enumerate(self.entries1):
                if(count < 3):
                    field.insert(0, self.lower_hsv1[count])
                else:
                    field.insert(0, self.upper_hsv1[count-3])

            # double HSV ranges
            if(length == 4):
                self.lower_hsv2 = np.asarray(hsvList[2])
                self.upper_hsv2 = np.asarray(hsvList[3])

                # enable second range
                if (self.secondHSV.get() != 1): 
                    self.secondHSV.set(1)
                    self.updateSelections()

                # update entries
                for count, field in enumerate(self.entries2):
                    if(count < 3):
                        field.insert(0, self.lower_hsv2[count])
                    else:
                        field.insert(0, self.upper_hsv2[count-3])

            # single HSV range
            else:
                # disable second range
                if(self.secondHSV.get() == 1):
                    self.secondHSV.set(0)
                    self.updateSelections()

        else:
            # update entries
            for field in self.entries1:
                field.delete(0, tk.END)
            for field in self.entries2:
                field.delete(0, tk.END)

            # disable second range
            if(self.secondHSV.get() == 1):
                self.secondHSV.set(0)
                self.updateSelections()

            # update variables
            self.lower_hsv1 = np.array([0,0,0])
            self.upper_hsv1 = np.array([0,0,0])
            self.lower_hsv2 = np.array([0,0,0])
            self.upper_hsv2 = np.array([0,0,0])

    # function to allow user to save HSV ranges to preferences file
    def saveRanges(self):
        # if required fields are filled in
        if(self.entryLowerH1.get() != "" and self.entryLowerS1.get() != "" and self.entryLowerV1.get() != "" and self.entryUpperH1.get() != "" and self.entryUpperS1.get() != "" \
            and self.entryUpperV1.get() != "" and ((self.secondHSV.get() == 1 and self.entryLowerH2.get() != "" and self.entryLowerS2.get() != "" and self.entryLowerV2.get() != "" \
                and self.entryUpperH2.get() != "" and self.entryUpperS2.get() != "" and self.entryUpperV2.get() != "") or self.secondHSV.get() != 1)):
            # ask for name
            name = tk.StringVar()
            newWindow = tk.Toplevel(self.root)
            newWindow.configure(background='#ffffff')

            nameLabel = tk.Label(
                newWindow,
                text="Name",
                background='#ffffff',
                foreground='#000000',
                font=("Calibri Light", 14))

            nameEntry = tk.Entry(
                newWindow, 
                font=("Calibri Light", 13),
                textvariable = name,
                width = 10)

            nameButton = tk.Button(
                newWindow,
                text = "Save",
                background='#ffffff',
                foreground='#000000',
                command = lambda: newWindow.destroy(),
                width = 10,
                font=("Calibri Light", 14))

            nameLabel.pack()
            nameEntry.pack()
            nameButton.pack()

            # wait until user inputs name
            self.root.wait_window(newWindow)

            if(name.get() != ""):
                # create output string
                outputString = "[" + np.array2string(self.lower_hsv1, separator = ',').replace("[","(").replace("]", ")") + "," + \
                    np.array2string(self.upper_hsv1, separator = ',').replace("[","(").replace("]", ")")

                if(self.secondHSV.get() == 1):
                    outputString += "," + np.array2string(self.lower_hsv2, separator = ',').replace("[","(").replace("]", ")")
                    outputString += "," + np.array2string(self.upper_hsv2, separator = ',').replace("[","(").replace("]", ")")
                
                outputString += "]"

                # add to config file
                self.config.set('HSV Ranges', name.get(), outputString)

        else:
            tkMessageBox.showinfo("Error", "Not All HSV Fields Populated")

    # function to run HSV range preview
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

            # convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # apply HSV mask
            mask = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
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

            # update variables
            self.lower_hsv1 = np.array([H1Slider.get(), S1Slider.get(), V1Slider.get()])
            self.upper_hsv1 = np.array([H2Slider.get(), S2Slider.get(), V2Slider.get()])

            # update entries
            for count, field in enumerate(self.entries1):
                if(count < 3):
                    field.insert(0, self.lower_hsv1[count])
                else:
                    field.insert(0, self.upper_hsv1[count-3])

            # disable second range
            if(self.secondHSV.get() == 1):
                self.secondHSV.set(0)
                self.updateSelections()

            newWindow.destroy()
            cv2.destroyAllWindows()

        # open new window
        newWindow = tk.Toplevel(self.root)
        newWindow.configure(background='#ffffff')
        newWindow.withdraw()
        filename = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))   

        newWindow.deiconify()
        newWindow.geometry("500x575") # window size of 500 x 575

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
        savetoMain.pack(pady = 25)

        # open comparison window
        cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

        # open image
        img = cv2.imread(filename)

        # resize image to 1/4 of original size
        img = cv2.resize(img, (0,0), None, 0.25, 0.25)

        # wait until user closes window
        self.root.wait_window(newWindow)