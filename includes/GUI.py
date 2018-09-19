# import necessary modules
import Tkinter as tk
import tkFileDialog
import colorsys
from tkColorChooser import askcolor
import tkMessageBox
from preview import HSVPreview

class GUI:
	def __init__(self):

		# open window
		self.root = tk.Tk()
		self.root.configure(background = '#ffffff')
		self.root.title("Generate Training Images")

		#---------------------------------------------------------------------------------
		# Variables
		#---------------------------------------------------------------------------------

		self.directory = ""
		self.lowerHex = ""
		self.upperHex = ""
		self.lowerHex2 = ""
		self.upperHex2 = ""
		self.lowerBorder = 0
		self.upperBorder = 0

		#---------------------------------------------------------------------------------
		# Labels
		#---------------------------------------------------------------------------------

		# step 1 Label
		self.label = tk.Label(
			self.root,
			text = "1) Select Image Folder",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri", 14))
		self.label.grid(row = 0, sticky = tk.W, padx = 10, pady = 5)

		# image path Label
		self.pathLabel = tk.Label(
			self.root,
			text = "No Directory Selected",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri", 12))
		self.pathLabel.grid(row = 1, column = 0, columnspan = 50, sticky = tk.W, padx = 30)

		# step 2 Label
		self.label2 = tk.Label(
			self.root,
			text = "2) Select HSV Range",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri", 14))			
		self.label2.grid(row = 2, column = 0, sticky = tk.W, padx = 10, pady = 5)

		# lower HSV label
		self.lowerLabel = tk.Label(
			self.root,
			background = '#ffffff',
			foreground = '#000000',
			width = 20,
			height = 1)		
		self.lowerLabel.grid(row = 3, column = 1, sticky = tk.W, padx = 20, pady = 5)

		# upper HSV label
		self.upperLabel = tk.Label(
			self.root,
			background = '#ffffff',
			foreground = '#000000',
			width = 20,
			height = 1)
		self.upperLabel.grid(row = 4, column = 1, sticky = tk.W, padx = 20, pady = 5)

		# second lower HSV label
		self.lowerLabel2 = tk.Label(
			self.root,
			background = '#ffffff',
			foreground = '#000000',
			width = 20,
			height = 1)		
		self.lowerLabel2.grid(row = 6, column = 1, sticky = tk.W, padx = 20, pady = 5)

		# second upper HSV label
		self.upperLabel2 = tk.Label(
			self.root,
			background = '#ffffff',
			foreground = '#000000',
			width = 20,
			height = 1)
		self.upperLabel2.grid(row = 7, column = 1, sticky = tk.W, padx = 20, pady = 5)

		# step 3 Label
		self.label3 = tk.Label(
			self.root,
			text = "3) Select Image Borders",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri", 14))		
		self.label3.grid(row = 8, column = 0, sticky = tk.W, padx = 10, pady = 5)

		# upper border label
		self.upperBorderLabel = tk.Label(
			self.root,
			text = "Upper Border",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri", 12))
		self.upperBorderLabel.grid(row = 9, column = 0, sticky = tk.W, padx = 20, pady = 5)

		# lower border label
		self.lowerBorderLabel = tk.Label(
			self.root,
			text = "Lower Border",
			background = '#ffffff',
			foreground = '#000000',
			font = ("Calibri", 12))
		self.lowerBorderLabel.grid(row = 10, column = 0, sticky = tk.W, padx = 20, pady = 5)

		#---------------------------------------------------------------------------------
		# Buttons
		#---------------------------------------------------------------------------------

		# choose directory button
		self.directoryButton = tk.Button(
			self.root,
			text = "Choose Folder:",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.selectDirectory(),
			font = ("Calibri", 12))
		self.directoryButton.grid(row = 0, column = 1, sticky = tk.W, padx = 20, pady = 10)

		# lower HSV range button
		self.lowerButton = tk.Button(
			self.root,
			text = "Lower Range:",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.getColour("lower"),
			font = ("Calibri", 12))
		self.lowerButton.grid(row = 3, column = 0, sticky = tk.W, padx = 20, pady = 5)

		# upper HSV range button
		self.upperButton = tk.Button(
			self.root,
			text = "Upper Range:",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.getColour("upper"),
			font = ("Calibri", 12))
		self.upperButton.grid(row = 4, column = 0, sticky = tk.W, padx = 20, pady = 5)

		# second lower HSV range button
		self.lowerButton2 = tk.Button(
			self.root,
			text = "Lower Range 2:",
			background = '#ffffff',
			foreground = '#D3D3D3',
			command = lambda: self.getColour("lower2"),
			state = "disabled",
			font = ("Calibri", 12))
		self.lowerButton2.grid(row = 6, column = 0, sticky = tk.W, padx = 20, pady = 5)

		# second upper HSV range button
		self.upperButton2 = tk.Button(
			self.root,
			text = "Upper Range 2:",
			background = '#ffffff',
			foreground = '#D3D3D3',
			command = lambda: self.getColour("upper2"),
			state = "disabled",
			font = ("Calibri", 12))
		self.upperButton2.grid(row = 7, column = 0, sticky = tk.W, padx = 20, pady = 5)

		# execute button
		self.runButton = tk.Button(
			self.root,
			text = "Generate Images",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.saveValues(),
			font = ("Calibri", 12))
		self.runButton.grid(row = 11, column = 0, sticky = tk.W, padx = 20, pady = 5)

		# preview HSV button
		self.HSVButton = tk.Button(
			self.root,
			text = " Preview Mask ",
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.launchPreview(),
			font = ("Calibri", 12))
		self.HSVButton.grid(row = 11, column = 1, sticky = tk.W, padx = 20, pady = 5)
		
		#---------------------------------------------------------------------------------
		# Entry
		#---------------------------------------------------------------------------------

		validateCommand = self.root.register(self.validate) # we have to wrap the command
		self.entryUpper = tk.Entry(self.root, validate = "key", validatecommand = (validateCommand, '%P', 'upper'))
		self.entryUpper.grid(row = 9, column = 1, sticky = tk.W, padx = 10, pady = 5)
		self.entryLower = tk.Entry(self.root, validate = "key", validatecommand = (validateCommand, '%P', 'lower'))
		self.entryLower.grid(row = 10, column = 1, sticky = tk.W, padx = 10, pady = 5)

		#---------------------------------------------------------------------------------
		# Checkbox
		#---------------------------------------------------------------------------------		

		self.secondHSV = tk.IntVar()
		self.checkBox = tk.Checkbutton(
			self.root, 
			text = "Second HSV Range", 
			variable = self.secondHSV,
			background = '#ffffff',
			foreground = '#000000',
			command = lambda: self.updateSelections(),
			font = ("Calibri", 12))
		self.checkBox.grid(row = 5, column = 0, stick = tk.W, padx = 20, pady = 5)

		self.root.mainloop()

	#---------------------------------------------------------------------------------
	# Functions
	#---------------------------------------------------------------------------------

	# validate method for text entry
	def validate(self, new_text, entry_field):
		if not new_text: # the field is being cleared
			if(entry_field == "upper"):
				self.upperBorder = 0
			else:
				self.lowerBorder = 0

		try:
			if(entry_field == "upper"):
				if(new_text == ""):
					self.upperBorder = 0
				else:
					self.upperBorder = int(new_text)
			else:
				if(new_text == ""):
					self.lowerBorder = 0
				else:
					self.lowerBorder = int(new_text)
			return True

		except ValueError:
			return False

	# function allow selection of directory/file where images are stored
	def selectDirectory(self):
		# open directory selector
		dirname = tkFileDialog.askdirectory(parent = self.root, initialdir = "/", title = 'Select Directory')

		# if new directory selected, update label		
		if(len(dirname) > 0):
			self.pathLabel.config(text = dirname)
			self.directory = str(dirname)

	def saveValues(self):
		# if second HSV range is not selected
		if(self.secondHSV.get() != 1):
			# make both ranges equal
			self.upperHex2 = self.upperHex
			self.lowerHex2 = self.lowerHex

		# close gui
		if(self.lowerHex != "" and self.upperHex != "" and self.directory != "" and (self.secondHSV.get() == 1 and self.upperHex2 != "" or self.secondHSV.get() != 1) \
			and (self.secondHSV.get() == 1 and self.lowerHex2 != "" or self.secondHSV.get() != 1)):
			self.root.destroy()
		else:
			tkMessageBox.showinfo("Error", "Not All Fields Populated")

	# function to return parameters
	def getValues(self):
		return self.directory, self.hex2HSV(self.lowerHex), self.hex2HSV(self.upperHex), self.hex2HSV(self.lowerHex2), \
		self.hex2HSV(self.upperHex2), self.upperBorder, self.lowerBorder

	# Convert hex to HSV
	def hex2HSV(self, hexCode):
		hexCode = hexCode.lstrip('#')
		rgb = tuple(int(hexCode[i:i+2], 16) for i in (0, 2, 4))
		hsv = colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
		return hsv

	# function to select colour
	def getColour(self, method):
		colour = askcolor()

		if(method == "upper"):
			self.upperHex = str(colour[1])
			self.upperLabel.config(bg = str(colour[1]))
		elif(method == "lower"):
			self.lowerHex = str(colour[1])
			self.lowerLabel.config(bg = str(colour[1]))
		elif(method == "upper2"):
			self.upperHex2 = str(colour[1])
			self.upperLabel2.config(bg = str(colour[1]))
		elif(method == "lower2"):
			self.lowerHex2 = str(colour[1])
			self.lowerLabel2.config(bg = str(colour[1]))

	# function to update appearance of second set of HSV selections
	# based on status of checkbox 
	def updateSelections(self):
		if(self.secondHSV.get() == 1):
			self.lowerButton2.config(fg = '#000000')
			self.upperButton2.config(fg = '#000000')
			self.lowerButton2.config(state = "normal")
			self.upperButton2.config(state = "normal")
		else:
			self.lowerButton2.config(fg = '#D3D3D3')
			self.upperButton2.config(fg = '#D3D3D3')
			self.lowerButton2.config(state = "disabled")
			self.upperButton2.config(state = "disabled")
			self.lowerLabel2.config(bg = '#ffffff')
			self.upperLabel2.config(bg = '#ffffff')

	# function to launch HSV preview tool
	def launchPreview(self):
		HSVPreview()