# script can be used to get all images from a large set taken at even intervals
# command prompt call is as follows
# python filterImages.py "directory" startingImage interval

# directory --> path to image set
# startingImage --> number (e.g. 140 for IMG_0140.JPG) of first image to be included
    # in new set
# interval --> interval between valid images (e.g. if IMG_0140.JPG and IMG_0188.JPG
    # are both to be included the interval is 48)

import os
import argparse

# setup argument parser
parser = argparse.ArgumentParser(description = 'Filter large image sets.')
parser.add_argument('directory', help = 'path to image set')
parser.add_argument('start', type = int, help = 'starting image number')
parser.add_argument('interval', type = int, help = 'interval of valid images')
args = parser.parse_args()

# get images
images = [file_name for file_name in os.listdir(args.directory)]

# iterate through file names
for x in images:
    # split string
    filename = x[4:8]

    # get filtered names
    if(int(filename) > args.start and not (int(filename) - args.start) \
        % args.interval == 0):
        path = args.directory + "\\" + x
        os.remove(path)
    else:
        print x
