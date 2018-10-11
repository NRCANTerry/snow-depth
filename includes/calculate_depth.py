# import necessary modules
import cv2
from progress_bar import progress
import xlsxwriter
import statistics
from matplotlib import pyplot as plt

# function to calculate the change in snow depth for each stake
# using the tensor from the specified template
def getDepths(imgs, img_names, intersectionCoords, stakeValidity, templateIntersections, upperBorder, tensors, debug, debug_directory):

    # list containing median depths for each image
    median_depths = list()

    # contains output data for JSON file
    depth_output = {}

    # num of images
    num_images = len(imgs)

    # create output dictionary for images
    depths = dict()

    # create excel workbook and add worksheet
    dest = str(debug_directory) + 'Depths.xlsx'
    workbook = xlsxwriter.Workbook(dest)
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, len(tensors) + 1, 20)

    # create format
    cell_format = workbook.add_format()
    cell_format.set_align('center')

    # add titles
    worksheet.write(0, 0, "Image", cell_format)
    worksheet.write(0, len(tensors) + 1, "Median Depth (mm)", cell_format)
    for i, j in enumerate(tensors):
        worksheet.write(0, i+1, ("Stake %s" % str(i)), cell_format)

    # start from the first cell
    row = 1
    col = 0

    # iterate through images
    for count, img_ in enumerate(imgs):
        # update progress bar
        progress(count + 1, num_images, status=img_names[count])

        # create an image to overlay points on if debugging
        if(debug):
            img_overlay = img_.copy()

        # list to hold calculated depths
        depths_stake = list()

        # get image name
        img_name = img_names[count]

        # reset column
        col = 0

        # write to excel file
        worksheet.write(row, col, img_name, cell_format)
        col = 1

        # get intersection coordiantes
        coords_stake = intersectionCoords[img_name]

        # iterate through stakes in image
        for i, stake in enumerate(coords_stake):
            # if stake is valid
            if stakeValidity[img_name][i]:
                # add reference circles to output image if debugging
                # shows intersection point of image with reference to template
                if(debug):
                    cv2.circle(img_overlay, (int(templateIntersections[i][0]), int(templateIntersections[i][1]) - upperBorder), 5, (255,0,0), 3)
                    cv2.circle(img_overlay, (int(stake["average"][0]), int(stake["average"][1])), 5, (0,255,0), 2)

                # calculate change in snow depth in mm
                depth_change = ((templateIntersections[i][1] - upperBorder) - stake["average"][1]) * tensors[i]

                # write to excel file
                worksheet.write(row, col + i, "%.2f" % depth_change, cell_format)

                # output to command line
                #print "stake %s : %s mm" % (j, depth_change)

                # add to list
                depths_stake.append(depth_change)

            # else append false to array
            else:
                depths_stake.append(False)

                # write to excel file
                worksheet.write(row, col + i, "n/a", cell_format)

        # output debug image
        if(debug):
            cv2.imwrite(debug_directory + img_name, img_overlay)

        # add list to dictionary
        depths[img_name] = depths_stake

        # determine median depth
        valid_depths = [x for x in depths_stake if x != False]
        median = statistics.median(valid_depths)

        # add to median depth list
        median_depths.append(median)

        # write median to excel file
        worksheet.write(row, len(tensors) + 1, "%.2f" % median, cell_format)

        # increment row
        row += 1

    # close workbook
    workbook.close()

    # generate plot
    plt.plot(img_names, median_depths, color='g')
    plt.xlabel('Images')
    plt.xticks(rotation=90)
    plt.ylabel('Change (mm)')
    plt.title('Change in Snow Depth (mm)')
    plt.savefig(debug_directory + "depth-graph.jpg")
    plt.close()

    # return dictionary containing snow depth changes
    return depths
