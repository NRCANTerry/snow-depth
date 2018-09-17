# Snow Depth Measurement
Automated measurement of snow depth using outdoor cameras

Steps:
1) Generate Ground Truth Data
	1. Place all training images in a folder
	2. Run generate_coordinates.py
		- specify image folder, HSV ranges, and image borders
		- outputs the coordinates of red regions in the image to a JSON file
	3. Run generate_truth.m
		- this creates the ground truth object that will be used to train the R-CNN