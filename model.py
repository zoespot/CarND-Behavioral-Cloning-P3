import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


lines=[]
dirsrc =['slow','run_n2'] #'run','run_n1','data','run_nspecial','runspecial','recovery','recovery4','recovery2','recovery3','fast','slow','bridge','run_n2']
for i in range(len(dirsrc)):
	csvname = dirsrc[i]+'/driving_log.csv'
	with open(csvname, 'r') as csvfile: 
		csvreader =csv.reader(csvfile)
		for line in csvreader:
			lines.append(line)
images =[]
measurements =[]
for line in lines:
	for i in range(3):
		sourcepath = line[i]
		token = sourcepath.split('\\')
		dirname = token[-3]
		filename = token[-1]
		localpath = dirname + "/IMG/" + filename
		image= cv2.imread(localpath)
		images.append(image)
	measurement = float(line[3])
	correction =0.2
	measurements.append(measurement)
	measurements.append(measurement+correction) #left
	measurements.append(measurement-correction) #right

X_train=np.array(images)
y_train=np.array(measurements)

augmented_images=[]
augmented_measurements=[]
for image, measurement in zip(images,measurements):
	flipped_image = cv2.flip(image,1) #flip horizontally
	flipped_measurement = measurement * -1.0
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train_augmented= np.array(augmented_images)
y_train_augmented= np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
keep=0.5
model = Sequential()
model.add(Cropping2D(cropping = ((70,25),(0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: x/255.0 -0.5))
model.add(Convolution2D(24,5,5, subsample =(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample =(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample =(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(keep))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam',loss='mse')
model.fit(X_train_augmented, y_train_augmented, validation_split=0.2,shuffle=True,nb_epoch=2)
#model.fit(X_train, y_train, validation_split=0.2,shuffle=True,nb_epoch=3)

model.save("model22.h5")
