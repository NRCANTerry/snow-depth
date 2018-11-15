import os
import argparse

# setup argument parser
parser = argparse.ArgumentParser(description = 'Filter large image sets.')
parser.add_argument('directory', help = 'path to image set')
parser.add_argument('hour', type = int, help = 'hour of image')
parser.add_argument('minute', type = int, help = 'minute of image')
args = parser.parse_args()

# create subfolder in directory
filtered_dir = args.directory + "\\filtered"
if(os.path.isdir(filtered_dir)):
    os.rmdir(filtered_dir)
os.mkdir(filtered_dir)

# get images
images = [file_name for file_name in os.listdir(args.directory)]

# iterate through file names
for x in images:
    if(x != "filtered"):
        # get EXIF data
        from PIL import Image
        from datetime import datetime
        img = Image.open(args.directory + "\\" + x)
        exif = img._getexif()[36867]
        date = datetime.strptime(exif, '%Y:%m:%d %H:%M:%S')

        # check if matches input time
        if date.hour == args.hour and date.minute == args.minute:
            # write to filtered directory
            file, ext = os.path.splitext(x)
            img.save(filtered_dir + "\\" + file + ext)
