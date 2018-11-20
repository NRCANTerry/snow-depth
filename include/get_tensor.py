# import necessary packages
from math import sqrt

def getTensor(ptA, ptB, distance):
    '''
    Determine tensor for stake given two points and an actual distance
    @param ptA coordinates of first point
    @param ptB coordinates of second point
    @param distance actual distance between the two points
    @type ptA list[x, y]
    @type ptB list[x, y]
    @type distance float
    @return tensor for stake
    @return type float
    '''

    # calculate pixel distance
    dist_px = sqrt((ptB[0] - ptA[0])**2 + (ptB[1] - ptA[1])**2)

    # determine tensor
    return float(distance / dist_px)
