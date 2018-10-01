import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2

# get image
image2 = cv2.imread("IMG_0005.JPG")
image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#-- Generate some data...
x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)

# make a line with "num" points
x0, y0 = 67, 40
x1, y1 = 101.5, 637
num = 500
x, y = np.linspace(x0,x1, num), np.linspace(y0, y1, num)

# extract the values along the line, using cubic interpolation
zi = scipy.ndimage.map_coordinates(np.transpose(image), np.vstack((x,y)))
print zi

# plot
fig, axes = plt.subplots(nrows=2)
axes[0].imshow(image)
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].axis('image')

axes[1].plot(zi)

print max(zi)
coords = [i for i, v in enumerate(zi) if v > 150]

first_coord = 0
for i, coord in enumerate(coords):
	if (coords[i+10] - coord) < 20:
		first_coord = coord
		break

print first_coord
print ("%s, %s" % (x[first_coord], y[first_coord]))

cv2.circle(image2, (int(x[first_coord]), int(y[first_coord])), 5, (0,255,0), 3)
cv2.imwrite("./Img2.JPG", image2)

plt.show()
