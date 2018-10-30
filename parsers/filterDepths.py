# script can be used to get all depths of interest from a text file
import os
import argparse
import datetime

# setup argument parser
parser = argparse.ArgumentParser(description = 'Get measurements from .txt files')
parser.add_argument('file', help = 'path to .txt file')
parser.add_argument('syear', type = int, help = 'starting year')
parser.add_argument('smonth', type = int, help = 'starting month')
parser.add_argument('sdate', type = int, help = 'starting date')
parser.add_argument('eyear', type = int, help = 'ending year')
parser.add_argument('emonth', type = int, help = 'ending month')
parser.add_argument('edate', type = int, help = 'ending date')
parser.add_argument('hour', type = int, help = 'hour of time of interest')
parser.add_argument('minute', type = int, help = 'minute of time of interest')
args = parser.parse_args()

# open file
file = open(args.file, "r")
lines = file.readlines()

# create dates
start_date = datetime.date(args.syear, args.smonth, args.sdate)
end_date = datetime.date(args.eyear, args.emonth, args.edate)

# iterate through lines
for x in lines:
    # split into components
    line = x.split()

    # create date
    cur_date = datetime.date(int(line[0]), int(line[1]), int(line[2]))

    # check if matches
    if start_date <= cur_date and cur_date <= end_date and int(line[3]) == args.hour \
        and int(line[4]) == args.minute:
        print line[5]
