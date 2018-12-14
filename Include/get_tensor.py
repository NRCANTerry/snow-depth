# import necessary packages
from math import sqrt

def getTensor(ptA, ptB, distance):
    """
    Function to determine the tensor for a stake given two points and an
    actual distance

    Keyword parameters:
    ptA -- coordinates of first point on stake
    ptB -- coordinates of second point on stake
    distance -- the actual distance between the two points
    """

    # calculate pixel distance
    dist_px = sqrt((ptB[0] - ptA[0])**2 + (ptB[1] - ptA[1])**2)

    # determine tensor
    return float(distance / dist_px)
