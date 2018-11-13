import cv2
import numpy as np

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximiate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if(len(approx) == 3):
            shape == "triangle"

        elif(len(approx) == 4):
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            shape = "square" if ar > 0.85 and ar <= 1.15 else "rectangle"

        return shape

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colours(hist, centroids):
    # initialize the bar chart representing the relative frequency of each
    # of the colours
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    # dominant colour
    dominant = np.array([0,0,0], dtype = "uint32")
    dominant_pct = 0.0

    # loop over the percentage of each cluster and the colour of each cluster
    for (percent, colour) in zip(hist, centroids):
        # plot the relativfe percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            colour.astype("uint8").tolist(), -1)
        startX = endX
        print colour, percent

        # if colour isn't white or black and has highest percentage
        rgb_prox = np.amax(colour.astype(np.int32)) - np.amin(colour.astype(np.int32))
        rgb_min = np.amin(colour.astype(np.int32))
        rgb_max = np.amax(colour.astype(np.int32))
        if(percent > dominant_pct and rgb_prox > 40 and rgb_min < 200 and rgb_max > 100):
            dominant = colour
            dominant_pct = percent

    red = np.uint8([[dominant]])
    print dominant, dominant_pct
    [[hsv]] = cv2.cvtColor(red, cv2.COLOR_RGB2HSV)
    print hsv

    upper = np.array([hsv[0] + 10, hsv[1] + 10, hsv[2]])
    lower = np.array([hsv[0], hsv[1] - 10, hsv[2] - 40])
    large_cnt = max(cnts, key = cv2.contourArea)

    print(lower, upper)

    img_hsv = cv2.imread("IMG_0001.JPG")
    image_mask = cv2.inRange(img_hsv, lower, upper)
    cv2.imshow("Mask", cv2.resize(image_mask, None, fx = 6, fy = 6))

    # return the bar chart
    return bar

img = cv2.imread("IMG_0001.JPG",0)
img_colour = cv2.imread("IMG_0001.JPG")
thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (50,50))
cnts = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
print(len(cnts))

sd = ShapeDetector()

#cv2.imshow("thresh", cv2.resize(close, None, fx = 6, fy = 6))

cnts.sort(key = lambda a: cv2.contourArea(a), reverse = True)

print(len(cnts))
print(type(cnts[0]))
large_cnt = max(cnts, key = cv2.contourArea)

mask2 = np.zeros(img_colour.shape, dtype = "uint8")
cv2.drawContours(mask2, [large_cnt], 0, (255,255,255), -1)
print cv2.contourArea(large_cnt)
print sd.detect(large_cnt)

mask = cv2.bitwise_and(mask2, img_colour)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
# reshape the image to be a list of pixels
mask = mask.reshape((mask.shape[0] * mask.shape[1], 3))

# cluster the pixel intensities
from sklearn.cluster import KMeans
clt = KMeans(n_clusters = 25)
clt.fit(mask)

hist = centroid_histogram(clt)
bar = plot_colours(hist, clt.cluster_centers_)
#cv2.imshow("mask", cv2.resize(mask, None, fx = 6, fy = 6))

from matplotlib import pyplot as plt
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

#cv2.imshow("img", cv2.resize(img, None, fx = 6, fy = 6))
