# Snow Depth Measurement
Automated measurement of snow depth using outdoor cameras

Steps:
1) Generate Ground Truth Data
	1. Place all training images in a folder
	2. Run generate_coordinates.py
		- specify image folder, HSV ranges, and image borders
		- outputs the coordinates of coloured regions in the image to a JSON file
		- to determine HSV ranges use the preview tool in the GUI
		- to save or remove HSV range use file menu - saved to preferences.cfg
	3. Run generate_truth.m
		- this creates the ground truth object that will be used to train the R-CNN
