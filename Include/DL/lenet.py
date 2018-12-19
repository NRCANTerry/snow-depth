# import necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import cv2
import random
sys.stderr = stderr

# training parameters
EPOCHS = 25
INIT_LR = 1e-3
BATCH_SIZE = 32

def absolutePaths(dir):
	"""
	Function to get absolute file paths from a directory

	Keyword arguments:
	dir -- the path to the file directory
	"""
	for path, _, filenames in os.walk(dir):
		for f in filenames:
			yield os.path.abspath(os.path.join(path, f))

# convolutional neural network architecture
class LeNet:
	def __init__(self, path, validSetPath, invalidSetPath):
		"""
		Function to train and save neural network

		Keyword arguments:
		path -- path for saved .model file
		validSetPath -- path to valid training images
		invalidSetPath -- path to invalid training images
		"""

		# get all image paths
		validImages = list(absolutePaths(validSetPath))
		invalidImages = list(absolutePaths(invalidSetPath))
		numValid = len(validImages)

		# lists for data and labels
		data = list()
		labels = list()

		# loop over input images
		for i, image in enumerate(validImages + invalidImages):
			# load the image, resize and store in data list
			img = cv2.imread(image)
			img = cv2.resize(img, (28, 28))
			img = img_to_array(img)
			data.append(img)

			# class label
			labels.append(1) if i < numValid else labels.append(0)

		# zip lists and shuffle
		c = list(zip(data, labels))
		random.seed(42)
		random.shuffle(c)

		# unzip
		data, labels = zip(*c)

		# scale raw pixel intensities to range [0, 1]
		data = np.array(data, dtype="float") / 255.0
		labels = np.array(labels)

		# partition the data into training and testing splits
		# 75% for training and 25% for testing
		(trainX, testX, trainY, testY) = train_test_split(data,
			labels, test_size=0.25, random_state=42)

		# convert the labels from integers to vectors
		trainY = to_categorical(trainY, num_classes=2)
		testY = to_categorical(testY, num_classes=2)

		# construct image generator
		imgGenerator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
			horizontal_flip=True, fill_mode="nearest")

		# initialize model
		model = self.build(self, width=28, height=28, depth=3, classes=2)
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,
			metrics=["accuracy"])

		# train network
		netResults = model.fit_generator(imgGenerator.flow(trainX, trainY, batch_size=BATCH_SIZE),
			validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
			epochs=EPOCHS, verbose=1)

		# save model
		model.save(path)

	@staticmethod
	def build(self, width, height, depth, classes):
		# initialize model
		model = Sequential()
		shape = (height, width, depth)

		# update input shape if using "channels first"
		if(K.image_data_format() == "channels_first"):
			shape = (depth, height, width)

		# first set of CONV ==> RELU ==> POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of CONV ==> RELU ==> POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# FC ==> RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier layers
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return network architecture
		return model
