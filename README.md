# Snow Depth Measurement
Automated measurement of snow depth using outdoor cameras

Project Structure:
- the main python script that is executed by the user is snow_depth.py
- the remaining files that contains functions used to perform individual functions (e.g. equalization) are stored inside the "./Include" directory  

<pre>
The files are organized as follows:  
| - - - snow-depth (project folder)  
|       | - - - AppData (generated by the algorithm)  
|	|	| - - - models (Deep Learning Models)  
|	|	| - - - templates (Templates saved by the algorithm)  
|	|	| - - - training (Deep learning training images are stored here)  
...  
|	| - - - Include  
|	|	| - - - DL (contains Deep Learning algorithms)  
|	|	|	| - - - classify.py  
|	|	|	| - - - lenet.py  
...  
|	|	| - - - GUI (contains GUI code)  
|	|	|	| - - - datePicker.py  
|	|	|	| - - - main.py  
|	|	|	| - - - template.py  
...  
|	|	| - - - calculate_depth.py  
|	|	| - - - check_stakes.py  
|	|	| - - - colour_balance.py  
|	|	| - - - equalize.py  
|	|	| - - - filter_night.py  
|	|	| - - - generate_report.py  
|	|	| - - - get_tensor.py  
|	|	| - - - intersect.py  
|	|	| - - - order_points.py  
|	|	| - - - overlay_roi.py  
|	|	| - - - register.py  
|	|	| - - - update_dataset.py  
...  
|	| - - - snow_depth.py (main script)  
</pre>

# Manual
## Running the Algorithm
- To run the algorithm you will need the following directories/files  
	- snow_depth.py  
	- "/Include/"  

<pre>
| - - - Your Folder
|	| - - - Include Directory
|	|	| - - - DL
...
|	|	| - - - GUI
...
|	| - - - snow_depth.py
</pre>

You will also need to setup a python 3.6 environment. You can use the attached environment (include environment).

## First Steps
- To measure the snow depth, the algorithm requires several files/preferences  
- It will automatically generate these files

### 1. Generate a Template
- To get a snow depth measurement, a template is required
- The template should be a daytime image with no dust/dirt on the camera lense
- The template can have snow or be a no snow image

1. Launch the script
2. In the top menu bar select "Templates" then "Create Template"
3. Select the template image using the pop-up window
4. Indicate whether the image has snow in it
5. Fill in the required parameters
	1. Tensor STD Dev - how much the tensor of an input image can vary from the average (5 is a good starting point)
	2. Register STD Dev - how much the registration matrix of an input image can vary from the average (5 is a good starting point)
	3. Maximum Transformation parameters - how much an image can be rotated, translated and scaled before the translation is considered invalid (used to filter out poor registrations)
6. Click Continue
7. Follow the GUI instructions
8. If the template has no snow, after selecting the blobs, you will be asked to identify the intersection point
	1. If it is visible, you may click on it with the mouse
	2. If it is not visible, you may click on any point along the stake and provide the GUI with the height of that point. It will automatically calculate where the base of the stake should be.

### 2. Create a Profile
- A profile contains the majority of the settings that are required to run the algorithm

1. With the script running and the main GUI window open select "Profiles" then "Create Profile"
2. Fill in the empty boxes in the preferences window
3. Filled boxes can also be modified but contain default values and should only be changed after running the algorithm on the default settings
4. Click "Create Profile" and follow the on-screen instructions

### 3. Determine the HSV Range
- In order to identify blobs in an input image, the algorithm must know what colour the reference blobs are
- After multiple runs the algorithm will initialize its own deep learning algorithm to help identify blobs but it is still recommended that you provide it with an HSV range

1. Fill in the boxes in the main GUI window under "HSV Range"
2. If you are not sure what the HSV Range for your image is, go to "File" then "Load Preview Tool"
	1. This will open the algorithm's built in tool to identify blob HSV ranges
	2. Select one of the images that you will be running through the algorithm
	3. Use the sliders to experiment with different HSV ranges
	4. The input image is shown on the left and the binary (thresholded) image on the right
	5. When you find an HSV Range(s) that results in only the blobs being included in the binary image, click "Save Values" to have then inputted into the main GUI window
3. Once you are happy with your HSV Range, save it by clicking "Colours" then "Save HSV Range"

### 4. Run the Algorithm
1. If you have followed the above steps you have succesfully created a template, profile and an HSV range that will be used by the algorithm to determine the snow depth of your input images
2. To run the algorithm first select the folder containing the input images by clicking "Select"
3. Then either input an HSV range input the text fields or select a saved field under "Select HSV Range"
4. Select a profile under "Settings"
5. If you would like the algorithm to output images at each stage (i.e. equalization, registration, etc.) then select "Debug"
6. Click "Run"
7. The algorithm will provide progress bars with time remaining estimates
8. Once it has completed, your results can be found in the "Reuslts" folder generated by the algorithm
9. The results are sorted by date so open the most recent folder
