# **SCND Term1 Project3: Behavioral Cloning**
---

**Behavioral Cloning Project**

The steps of this project are the following:
* Used the simulator to collect data of good driving behavior of both **clockwise** and **counter clockwise** runs with **recovery** runs from bad positions.
* Augmented data with horizontal **flip** and took advantage of all **center/left/right** images with proper steering angle correction.
* Built a convolution neural network (**CNN**) in **Keras** with **cropping** and **normalization** that predicted steering angles from images, use more data or dropout to reduce **overfitting**  
* Train and validate the model with a training and validation set with **80/20** percent split, use **generator** to generate data for training rather than storing the training data in memory.
* Test that the model successfully drives around **track #1 and #2** without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **model_generator.py** is the script with generator, used in training track #2
* **drive.py** for driving the car in autonomous mode
* I wasn't able to achieve one unified model for both tracks. **model_track1.h5** containing a trained convolution neural network for track #1, and "**model_track2**" contains model for "track2"
* **writeup_report_submit.md** summarized the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_track1.h5
python drive.py model_track2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains **comments** to explain how the code works.

* **model.py** is the CNN model for track1
* **model_generator.py** is the **similar** model with generator and one more "Dropout" layer, with more datasets used for track #2
---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on Nivida end-to-end deep learning network.
(Reference: https://devblogs.nvidia.com/deep-learning-self-driving-cars/).  It consists:
* A **Cropping** layer to remove the top and bottom of the images outside road (sky, tree, car hood, etc)
* A **Normalization** layer to normalize data to -0.5 to 0.5, which helps conditioning and speedup of following CNN layers
* 5 **Convolutional** layers to extract image features. First three convolutional layers with a 2×2 stride and a 5×5 kernel, and last two layers with non-strided convolution with a 3×3 kernel size. Depths are between 24 o 64. **RELU** layers are used to introduce nonlinearity.
* 4 full connected **Dense** layers, with **Dropout** is used after 1st dense layer. Last layer is **output** layer to provide steering control.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The final model used an Adam optimizer with default starting learning rate of 0.001, Dropout keep probability 0.5, and batch_size of 32. I tried other parameters and didn't find much improvement, so the default values are used.

Epochs number is 2-3, because the model converged quite quickly after 1-2 epochs. With more epochs, the loss errors will reduce, but in actual driving it may perform worse. I found loss error values does not correlate with actual driving performance.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with corrections.

For details about how I created the training data, see the next section.

---
### Detailed Model Architecture and Training Strategy

#### 1. Solution Design Approach

**- Track1**

The overall strategy for deriving a model architecture was to use an end-to-end convolution neural network to train images from the cameras directly without explicit features identified.

My first step was to use a convolution neural network model similar to LeNet. I thought this model might be appropriate because it proves to be good to image feature extraction in the traffic light classification project.

I added cropping layers to exclude sky, tree, and car hood on the top and bottom of the images, leaving only the middle road portion with lane lines. Because driving only replies on that portion.

Then I normalized the cropped RGB image to -0.5 to 0.5, so that the data is better conditioned for CNN. The first two steps added greatly improved the driving performance.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set. This implied that the model was underfitting. We need more sophisticated model.

The modified model is based on Nvidia's paper described in part1 last section, with larger CNN with more convolutional layers.

I found that the modified model had a low mean squared error on the training set but a high mean squared error on the validation set after 1-2 epochs. This implied that the model was overfitting.

To combat the overfitting, I added more data with a few more runs. I also added dropout layers and reduce dropout keep probability to reduce overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle hit the left borders, so I add more runs in the reversed direction to balance the biased left turn tendency in one direction. Then there are also some spots the car fell off the track with the other side lane line is missing or the curvature is too high, I added a few recovery runs to start from the bad positions.

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

**Track 2**

The track #1 model does not run well on track #2. I added a couple runs from track2 in both directions in addition to track1 datasets. The car ran much better expect on spot hitting left to the hill. I added a few recovery images from that spot, and the model started to run the whole track without leaving the road.

However, when I applied the model back to track1, it hit the border after the bridge on the left. I then tried to add a few recovery runs for track1, more runs from both tracks, and also tried a few model parameters tuning, dropout layer add/removal, convolutional layer add/removal, etc. The loss error reduced in both training and validation sets by more than 40% (0.05 -> 0.03 even 0.01), the model can't drive both tracks without leaving the road.

The final models and videos I uploaded are for each track separately.

#### 2. Final Model Architecture

The final model architecture as described in section 1 consisted of a convolution neural network with the following layers and layer sizes

Layer (type) | Output Shape    |      Param #     |Connected to|
-------------|-----------------|------------------|------------|
cropping2d_1 (Cropping2D)|(None, 65, 320, 3)|0|cropping2d_input_1[0][0]
lambda_1 (Lambda)                |(None, 65, 320, 3)|    0          | cropping2d_1[0][0]
convolution2d_1 (Convolution2D)|  (None, 31, 158, 24) |  1824  |      lambda_1[0][0]
convolution2d_2 (Convolution2D) | (None, 14, 77, 36) |   21636    |   convolution2d_1[0][0]
convolution2d_3 (Convolution2D)  |(None, 5, 37, 48)|     43248|       convolution2d_2[0][0]
convolution2d_4 (Convolution2D) | (None, 3, 35, 64)  |   27712  |     convolution2d_3[0][0]
convolution2d_5 (Convolution2D)|  (None, 1, 33, 64)  |   36928     |  convolution2d_4[0][0]
flatten_1 (Flatten)    |          (None, 2112) |         0 |          convolution2d_5[0][0]
dense_1 (Dense)           |       (None, 100) |          211300  |    flatten_1[0][0]
dropout_1 (Dropout)   |           (None, 100) |          0 |          dense_1[0][0]
dense_2 (Dense)       |           (None, 50)     |       5050    |    dense_1[0][0]
dense_3 (Dense)    |              (None, 10)  |          510   |      dense_2[0][0]
dense_4 (Dense)    |              (None, 1)    |         11       |   dense_3[0][0]
**Total params: 348,219**
Trainable params: 348,219
Non-trainable params: 0

Here is a visualization of the architecture:

![cnn-architecture](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/cnn-architecture.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here are example images of center lane driving from cetner/left/right cameras:

![center](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/center_2018_07_08_20_32_18_716.jpg)
![left](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/left_2018_07_08_20_32_18_716.jpg)
![right](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/right_2018_07_08_20_32_18_716.jpg)

The left camera image is used with steering angle correction of +0.2, and right camera image with correction of -0.2. Correction 0.2 is chosen based on fine tuning around small angles.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from both biased positions. These images show what a recovery looks like starting from left and right respectively:

![recovery-left](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/center_2018_07_08_21_54_49_066.jpg)
![recovery-right](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/center_2018_07_09_19_56_07_744.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would augment the dataset without additional runs. For example, here is an image that has then been flipped from original image:

![original](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/center_2018_07_08_20_32_05_789.jpg)
![flipped](https://github.com/zoespot/CarND-Behavioral-Cloning-P3/blob/master/examples/center_2018_07_08_20_32_05_789_flipped.jpg)

The correction coefficient is -1 for flipped images.

After the collection process, I had 23153 data points (3 cameras total 63459 images), and with flipping totally 126918 data points. I then preprocessed this data by cropping the 160x320x3 size image to 65x320x3


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2-3 as evidenced by balanced loss error between training and validation dataset. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

For track1, the epochs and training/validation losses are: (although it's overfitting with this model, it drives better than the model with more dropout added and less validation errors)
```
100958/100958 [==============================] - 279s - loss: 0.0231 - val_loss: 0.0383
Epoch 2/3
100958/100958 [==============================] - 267s - loss: 0.0204 - val_loss: 0.0385
Epoch 3/3
100958/100958 [==============================] - 271s - loss: 0.0196 - val_loss: 0.0393
```
For track2, the epochs and training/validation losses are:
```
129630/129630 [==============================] - 280s - loss: 0.0675 - val_loss: 0.0660
Epoch 2/3
129630/129630 [==============================] - 273s - loss: 0.0565 - val_loss: 0.0685
Epoch 3/3
129630/129630 [==============================] - 271s - loss: 0.0536 - val_loss: 0.0579

```
---
## Final Comments
1. A few observations:
  * **Current model is not very error tolerant, so training dataset quality is vital.** Some runs will be better than other runs, even with the same model. Human driving may drift excessively in the collecting the training dataset. We need to either drive carefully in training mode, or exclude the bad moves afterwards. Drive.py models driving consistently around 10-15mph. It also helps to train at this steady speed.
  * **Recovery can be very helpful to correct one spot error**, as I add in track2 training. But recovery won't help much if multiple spots need correction, as I try to correct track1 issues with several recovery runs added to track2 model.
  * Model architecture is important. With one less convolutional layer for example, the error rate stuck above 0.06, no matter how to tune other training parameters or datasets.
  * **Cropping and data augmentation are very helpful**. It greatly increase the model accuracy and driving performance.
  * **MSE (mean squared error) estimation is NOT a good indication of driving performance**, when error rates go below 0.05. Because with low error level, it's difficult to say if it's actually a model error to fit, or a human judgement mistake from the dataset. Generally high error rate mode (>0.08) won't drive well. However, when the error rate goes below 0.05, then model with error rate 0.04 won't necessarily drive better than model with error rate 0.03.
  * **Epochs number don't need to be large**. More epochs with more dropout will result in lower training and validation error. However, the model doesn't drive better with less validation errors.
  * Dropout layer will reduce the validation error and overfitting, but it's not necessary generating better model to drive. No more than one Dropout layer used in final model.
  * **Dataset doesn't need to be large**. As in the above points, data quality is more important. Adding more training runs and recovery runs with worse quality will deteriorate final model, even with less training/validation errors.  

2. Further improvement ideas:
  * In order to achieve a unified model for both tracks or even unknown tracks, high quality training dataset might be needed. It might be done by writing a program to calibrate the steering angle from the images and knowledge of the known tracks, instead of relying on one human judgement at one scene. Or we need to manually analyze every error image to figure out the cause.
  * Alternatively construct a more robust neural network architecture to tolerate human mistakes

---
### Content added based on project submission feedback
* Add activation for Dense layer too
  * Runs great on track2 with barely touching the right border at the dead-trun
  * Hit right on track1, probably because it's trained in track2 data to ride on top of the middle lane line
* Add Batch Normalization before activation
  * even worse for track1, hit right immediately, also fail on track2
* Use Nadam (Nesterov momentum based Adam) optimizer instead of Adam
* Use ELU (exponential linear unit) instead of Relu
  * may converge faster, need to upgrade Keras from v1.2.1 to 2.2
  * keras v2.2 runs slow with fit generator and it doesn't run with drive.py
* Use comma ai architecture instead of Nivida architecture
