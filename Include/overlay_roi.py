# import necessary modules
import cv2
import tqdm

# optional function to overlay template roi (stake and blob bounding boxes)
# onto registered images
def overlay(imgs, templateIntersections, roiCoordinates, upperBorder, img_names, debug_directory):
    # number of images
    num_images = len(imgs)

    # image iterator
    iterator = 0

    # iterate through images
    for img_ in tqdm.tqdm(imgs):
        # create write copy of image
        img_write = img_.copy()

        # iterate through stakes
        for j, stake in enumerate(roiCoordinates):
            # overaly template intersection point
            cv2.circle(img_write, (int(templateIntersections[j][0]),
                int(templateIntersections[j][1] - upperBorder)), 5, (0,255,0), 3)

            # iterate through roi in each stake
            for i, rectangle in enumerate(stake):
                # stake itself
                if(i == 0):
                    cv2.rectangle(img_write, (int(rectangle[0][0]), int(rectangle[0][1])-upperBorder),
                        (int(rectangle[1][0]), int(rectangle[1][1])-upperBorder), (0, 0, 255), 3)
                # blobs
                else:
                    cv2.rectangle(img_write, (int(rectangle[0][0]), int(rectangle[0][1])-upperBorder),
                        (int(rectangle[1][0]), int(rectangle[1][1])-upperBorder), (0, 255, 0), 3)

        # write image to debug directory
        cv2.imwrite(debug_directory + img_names[iterator], img_write)

        # increment iterator
        iterator += 1
