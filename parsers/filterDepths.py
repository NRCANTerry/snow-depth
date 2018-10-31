# script can be used to get all depths of interest from a text file
from __future__ import print_function
import os
import argparse
import datetime

# setup argument parser
parser = argparse.ArgumentParser(description = 'Get measurements from .txt files')
parser.add_argument('file', help = 'path to .txt file')
parser.add_argument('write', help = 'path where output .txt file will be written', nargs = '?', default = "")
parser.add_argument('start', help = 'first date to be included in "YEAR-MONTH-DAY" format')
parser.add_argument('end', help = 'last date to be included in "YEAR-MONTH-DAY" format')
parser.add_argument('time', help = 'time of measurement in "HOUR-MINUTE" format"')
args = parser.parse_args()

# get start date, end date and time
start = args.start.split("-")
end = args.end.split("-")
time = args.time.split("-")

# open file
file = open(args.file, "r")
lines = file.readlines()

# create dates
start_date = datetime.date(int(start[0]), int(start[1]), int(start[2]))
end_date = datetime.date(int(end[0]), int(end[1]), int(end[2]))

# open file to write
if(args.write != ""):
    write_path = args.write + "\\snow-measurements.txt"
    file_write = open(write_path, "w")

# iterate through lines
for i, x in enumerate(lines):
    # split into components
    line = x.split()

    # create date
    cur_date = datetime.date(int(line[0]), int(line[1]), int(line[2]))

    # check if matches
    if start_date <= cur_date and cur_date <= end_date and line[3] == time[0] \
        and line[4] == time[1]:
        # get 30 minute average
        avg = (float(line[5]) + float(lines[i-1].split()[5]) + float(lines[i+1].split()[5])) / 3.0
        print(avg)
        if(args.write != ""): print(str(avg), file = file_write)

if(args.write != ""): file_write.close()
