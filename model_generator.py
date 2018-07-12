import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#For dataset out of memory limit
#Used for track2
#Read in dataset of both track1 and track2 (clock wise, counter clock wise, and one recovery section for track2)
lines=[]
dirsrc =['slow','lap2','lap2_n1','data','run_n2','lap2_special'] 
for i in range(len(dirsrc)):
	csvname = dirsrc[i]+'/driving_log.csv'
	with open(csvname, 'r') as csvfile: 
		csvreader =csv.reader(csvfile)
		for line in csvreader:
			lines.append(line)

#Divide traing and validation set 80/20
train_samples,validation_samples = train_test_split(lines,test_size=0.2)

#Define generator to read in images(with center/left/right images and horizontal flip augmentation) 
#and steering angles(with left/right/flip corrections) 
def generator(samples, batch_size):
	num_samples =len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size] 
			images =[]
			angles =[]
			for batch_sample in batch_samples:
				name= batch_sample[0].split("\\")[-3] +'/IMG/'+batch_sample[0].split('\\')[-1]
				center_image =cv2.imread(name)
				center_angle =float(batch_sample[3])
				center_image_flip =cv2.flip(center_image,1)
				center_angle_flip =center_angle* -1.0

				left_correction = 0.2#0.03
				right_correction = 0.2#0.02

				left_name= batch_sample[1].split("\\")[-3] +'/IMG/'+batch_sample[1].split('\\')[-1]
				left_image =cv2.imread(left_name)
				left_angle =float(batch_sample[3]) + left_correction
				left_image_flip =cv2.flip(left_image,1)
				left_angle_flip =left_angle* -1.0

				right_name= batch_sample[2].split("\\")[-3] +'/IMG/'+batch_sample[2].split('\\')[-1]
				right_image =cv2.imread(right_name)
				right_angle =float(batch_sample[3]) - right_correction
				right_image_flip =cv2.flip(right_image,1)
				right_angle_flip =right_angle* -1.0

				images.append(center_image)
				angles.append(center_angle)
				images.append(center_image_flip)
				angles.append(center_angle_flip)
				
				images.append(left_image)
				angles.append(left_angle)
				images.append(left_image_flip)
				angles.append(left_angle_flip)

				images.append(right_image)
				angles.append(right_angle)
				images.append(right_image_flip)
				angles.append(right_angle_flip)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train,y_train)

#Define generators with default batch_size 
train_generator =generator(train_samples,batch_size =32)
validation_generator = generator(validation_samples, batch_size=32)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#Define CNN from Nvidia model with cropping and normaliztion 
#Dropout not needed 
keep=0.5
model = Sequential()
model.add(Cropping2D(cropping = ((70,25),(0,0)), input_shape= (160,320,3)))
model.add(Lambda(lambda x: x/255.0 -0.5))
model.add(Convolution2D(24,5,5, subsample =(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample =(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample =(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(keep))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

#Define optimizer with Adam default learning rate 
#optimizer= keras.optimizers.Adam(lr = 0.003)
model.compile(optimizer='adam', loss='mse')
ratio=6 #data augmented is 6 times of the center images #
model.fit_generator(train_generator, samples_per_epoch = ratio*len(train_samples), validation_data=validation_generator, nb_val_samples =len(validation_samples), nb_epoch=2)
model.save("model_track2.h5")
