# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.JPG "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/center_recovery.jpg "Center Recovery Image"
[image4]: ./examples/left_recovery.jpg "Left Recovery Image"
[image5]: ./examples/right_recovery.jpg "Right Recovery Image"
[image6]: ./examples/center_lane_driving.jpg "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I started with a simple neural network to ensure that my data pipeline was functioning correctly, then I built it out after some testing. First I added a lambda function to normalize my data and scale it between -1 and 1. Then I added the cropping 2D function to crop the parts of the image data that are not useful for training the model. The model then has 3 convolutional layers using a 5x5 kernel size, and a convolutional layer with a 3x3 kernel. These RELU layers introduce nonlinearity. Then I have a flattening layer followed by fully connected layers. 

#### 2. Attempts to reduce overfitting in the model

I employed dropout layers (model.py lines 102-110) to try to reduce overfitting in my model. I also created separate training and validation data sets(model.py lines 62-68). 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

I collected a variety of data to train my model to drive the car and stay on the road. I collected two laps laps of driving facing one way, then turned the car around and collected one lap driving the other way. I also included driving recovering the car from the road edge and returning to the center of the lane. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Using the resources and suggestions provided in the lesson I decided on what features to include in my model architecture. The first step I did was to take one of the simple neural networks from the Keras lessons and try to apply it to the project. Then I did a bit more research and perused some of the supplemental material provided in the course before settling on my final architecture. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layers to my model. I also ran the model many times while adjusting the number of epochs. Ultimately I felt that using 7 epochs worked well with my model. 

It took several attempts but at the end of the process, the vehicle was able to drive autonomously around the track without leaving the road. 

#### 2. Final Model Architecture


Here is a visualization of the architecture:

![Architecture][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to the center of the lane if it drifted towards the edge. These images show what a recovery looks like starting from center, left, then right:

![Center Recovery][image3]
![Left Recovery][image4]
![Right Recovery][image5]


After the collection process, I had 13773 data points. I then preprocessed this data by normalizing and cropping the images to only include the relevant (road) area. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as determined after testing my model using different numbers of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
