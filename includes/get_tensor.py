# import necessary packages
from math import sqrt

# function to determine the tensor given two points and an actual distance
def getTensor(ptA, ptB, distance):
    # calculate pixel distance
    dist_px = sqrt((ptB[0] - ptA[0])**2 + (ptB[1] - ptA[1])**2)

    # determine tensor
    return float(distance / dist_px)
