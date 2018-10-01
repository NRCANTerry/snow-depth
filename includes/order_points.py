import numpy as np
from scipy.spatial import distance as dist

# function to order coordinate points
# returns coordinates in top-left, top-right, bottom-right,
# bottom-left order
def orderPoints(pts, numpy = True):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]

	# get left-most and right-most points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]

	# sort left-most coordinates according to y coordinates
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost

	# find bottom right
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]

	# return coordiantes in order
	if(numpy):
		return tl, tr, br, bl
	else:
		return tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()