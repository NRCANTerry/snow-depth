# import necessary modules
import Tkinter as tk
import tkFileDialog
import tkMessageBox
from preview import HSVPreview
import ttk
import numpy as np

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

        # step 3 label
        self.label3 = tk.Label(
            self.root,
            text="Borders",
            background='#ffffff',
            foreground='#000000',
            font=("Calibri Light", 24))

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
            command=lambda: self.launchPreview(),
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
            self.lowerH2.config(fg='#000000')
            self.lowerS2.config(fg='#000000')
            self.lowerV2.config(fg='#000000')
            self.upperH2.config(fg='#000000')
            self.upperS2.config(fg='#000000')
            self.upperV2.config(fg='#000000')  
            self.arrow2.config(fg='#000000')            
            self.entryLowerH2.config(state="normal")
            self.entryLowerS2.config(state="normal")
            self.entryLowerV2.config(state="normal")
            self.entryUpperH2.config(state="normal")
            self.entryUpperS2.config(state="normal")
            self.entryUpperV2.config(state="normal")

        else:
            self.lowerH2.config(fg='#D3D3D3')
            self.lowerS2.config(fg='#D3D3D3')
            self.lowerV2.config(fg='#D3D3D3')
            self.upperH2.config(fg='#D3D3D3')
            self.upperS2.config(fg='#D3D3D3')
            self.upperV2.config(fg='#D3D3D3')  
            self.arrow2.config(fg='#D3D3D3')  
            self.entryLowerH2.delete(0, tk.END)
            self.entryLowerS2.delete(0, tk.END)
            self.entryLowerV2.delete(0, tk.END)
            self.entryUpperH2.delete(0, tk.END)
            self.entryUpperS2.delete(0, tk.END)
            self.entryUpperV2.delete(0, tk.END)
            self.entryLowerH2.config(state="disabled")
            self.entryLowerS2.config(state="disabled")
            self.entryLowerV2.config(state="disabled")
            self.entryUpperH2.config(state="disabled")
            self.entryUpperS2.config(state="disabled")
            self.entryUpperV2.config(state="disabled")

    # function to launch HSV preview tool
    def launchPreview(self):
        HSVPreview()