# import necessary modules
import cv2
import xlsxwriter
import statistics
from matplotlib import pyplot as plt
import math
import tqdm
import numpy as np
import datetime

def getDepths(imgs, img_names, intersectionCoords, stakeValidity, templateIntersections,
    upperBorder, tensors, actualTensors, intersectionDist, blobDistTemplate, debug, debug_directory,
    image_dates, imageSummary):
    """
    Function to calculate the change in snow depth for each stake using the tensor
    from the specified template

    Keyword arguments:
    imgs -- list of input images
    img_names -- list of corresponding image file names
    intersectionCoords -- list containing intersection coordinates for input images
    stakeValidity -- list indicating which stakes in input images are valid
    templateIntersections -- list containing intersection coordinates for template
    upperBorder -- upper crop parameter
    tensors -- tensors from template image
    actualTensors -- tensors calculated for input images
    intersectionDist -- list containing distances from blobs to intersection points
        for input images
    blobDistTemplate -- list containing blob to intersection point distances from
        template
    debug -- bool flag indicating whether output images should be saved
    debug_directory -- directory where output images should be written
    image_dates -- list containing dates of images extracted from EXIF data
    imageSummary -- dictionary containing information about each run
    """

    # list containing median depths for each image
    median_depths = list()
    median_depths_est = list()

    # contains output data for JSON file
    depth_output = {}

    # num of images
    num_images = len(imgs)

    # create output dictionary for images
    depths = dict()

    # create excel workbook and add worksheet
    dest = str(debug_directory) + 'snow-depths.xlsx'
    workbook = xlsxwriter.Workbook(dest)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, len(tensors) + 3, 25)

    # create format
    cell_format = workbook.add_format()
    cell_format.set_align('center')

    # add titles
    worksheet.write(0, 0, "Image", cell_format)
    worksheet.write(0, 1, "Date", cell_format)
    worksheet.write(0, len(tensors) + 2, "Median Depth (mm)", cell_format)
    worksheet.write(0, len(tensors) + 3, "Median Estimate (mm)", cell_format)
    for i, j in enumerate(tensors):
        worksheet.write(0, i+2, ("Stake %s" % str(i)), cell_format)

    # start from the first cell
    row = 1
    col = 0

    # image iterator
    iterator = 0

    # iterate through images
    for img_ in tqdm.tqdm(imgs):
        # create an image to overlay points on if debugging
        if(debug):
            img_overlay = img_.copy()

        # list to hold calculated depths
        depths_stake = list()
        estimate_stake = list()

        # get image name
        img_name = img_names[iterator]

        # reset column
        col = 0

        # write to excel file
        worksheet.write(row, col, img_name, cell_format)
        if isinstance(image_dates[iterator], datetime.datetime):
            worksheet.write(row, col + 1, image_dates[iterator].strftime('%x %X'), cell_format)
        col = 2

        # get intersection coordiantes
        coords_stake = intersectionCoords[img_name]

        # get blob intersection distances
        intersection_dist_stake = intersectionDist[img_name]

        # iterate through stakes in image
        for i, stake in enumerate(coords_stake):
            # if stake is valid and intersection point was found
            if stakeValidity[img_name][i] and stake["average"][1] != False:
                # add reference circles to output image if debugging
                # shows intersection point of image with reference to template
                if(debug):
                    cv2.circle(img_overlay, (int(templateIntersections[i][0]), int(templateIntersections[i][1]) - upperBorder), 5, (255,0,0), 3)
                    cv2.circle(img_overlay, (int(stake["average"][0]), int(stake["average"][1])), 5, (0,255,0), 2)

                # calculate change in snow depth in mm
                tensor = actualTensors[img_name][i] if actualTensors[img_name][i] != True else tensors[i]
                depth_change = ((templateIntersections[i][1] - upperBorder) - stake["average"][1]) * tensor

                # calculate change in snow depth using blob distances
                distances_stake = list()
                for w, x in enumerate(intersection_dist_stake[i]):
                    if x != False:
                        distances_stake.append((abs(blobDistTemplate[i][w]) - abs(x)) * tensor)
                distance_estimate = statistics.median(distances_stake) if len(distances_stake) > 0 else 0

                # write to excel file
                worksheet.write(row, col + i, "%.2f (%.2f)" % (depth_change, distance_estimate), cell_format)

                # add to list
                depths_stake.append(depth_change)
                estimate_stake.append(distance_estimate)

            # if stake wasn't valid or intersection point not found
            else:
                # if stake was valid
                if stakeValidity[img_name][i]:
                    worksheet.write(row, col + i, "Not Found", cell_format)
                # invalid stake
                else:
                    worksheet.write(row, col + i, "Invalid Stake", cell_format)

                # append false to array
                depths_stake.append(False)
                estimate_stake.append(False)

        # output debug image
        if(debug):
            cv2.imwrite(debug_directory + img_name, img_overlay)

        # add list to dictionary
        depths[img_name] = depths_stake

        # determine median depth
        valid_depths = [x for x in depths_stake if x != False]
        valid_estimates = [x for x in estimate_stake if x != False]

        if(len(valid_depths) > 0):
            median = statistics.median(valid_depths)
            median_est = statistics.median(valid_estimates)
        else:
            median = False
            median_est = False

        # add to median depth list
        median_depths.append(median)
        median_depths_est.append(median_est)

        # write median to excel file
        if median != False and median > 0:
            worksheet.write(row, len(tensors) + 2, "%.2f" % median, cell_format)
            worksheet.write(row, len(tensors) + 3, "%.2f" % median_est, cell_format)
        elif median != False:
            worksheet.write(row, len(tensors) + 2, "0.0", cell_format)
            worksheet.write(row, len(tensors) + 3, "0.0", cell_format)
        else:
            worksheet.write(row, len(tensors) + 2, "n/a", cell_format)
            worksheet.write(row, len(tensors) + 3, "n/a", cell_format)

        # increment row
        row += 1

        # increment iterator
        iterator += 1

        # update image summary
        imageSummary[img_name]["  "] = ""
        imageSummary[img_name]["Stake (Depth Calculation)"] = "Depth (mm)    Estimate (mm)"
        for e, depth in enumerate(depths_stake):
            if isinstance(depth, float):
                imageSummary[img_name]["  %d  " % (e+1)] = "%0.2f                %0.2f       " % \
                    (depth, estimate_stake[e])
            else:
                imageSummary[img_name]["  %d  " % (e+1)] = "%s                   %s          " % \
                    ("n/a", "n/a")

    # close workbook
    workbook.close()

    # remove negative values
    filterSet = zip(median_depths, median_depths_est, image_dates)
    filterSet = [(x, y, z) for x, y, z in filterSet if x != False]
    median_depths, median_depths_est, image_dates = zip(*filterSet)
    median_depths = np.asarray(median_depths).clip(0)
    median_depths_est = np.asarray(median_depths_est).clip(0)

    # generate plot
    fig,ax = plt.subplots(1)
    plt.plot(image_dates, median_depths)
    plt.plot(image_dates, median_depths_est)
    plt.gcf().autofmt_xdate()
    plt.legend(['Median Depth', 'Median Estimate'], loc='upper left')
    ax.set_xlabel("Date")
    ax.set_ylabel("Snow Depth (mm)")
    plt.xticks(rotation=75)
    plt.tight_layout()

    # save figure
    plt.savefig(debug_directory + "depth-graph.jpg")
    plt.close()

    # return dictionary containing snow depth changes
    return depths, imageSummary
