import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda, Cropping2D, Dropout




lines = []

with open('data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader: 
        lines.append(line)
del lines[0]

#generate data using left, right, and center images
def get_data(lines, batch_size = 32, image_path = 'data/IMG/'):
    num_lines = len(lines)
    
    #loop continuously 
    while 1: 
        #shuffle the data 
        shuffle(lines)
        for offset in range(0, num_lines, batch_size): 
            batch_lines = lines[offset:offset + batch_size]
            
            images = []
            measurements = []
            for sample in batch_lines:
                
                #center image
                name = image_path + sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_measurement = float(sample[3])
                images.append(center_image)
                measurements.append(center_measurement)
                
                #left image
                name = image_path + sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                images.append(left_image)
                measurements.append(center_measurement + 0.2)
                
                #right image
                name = image_path + sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                images.append(right_image)
                measurements.append(center_measurement - 0.2)
                
            #convert to numpy arrays
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            #return shuffled training sets
            yield shuffle(X_train, y_train)

#split training and validation data
training_data, validation_data = train_test_split(lines, test_size = 0.2)

#variable for changing the batch size
batch_size = 32

get_training = get_data(training_data, batch_size = batch_size)

get_validation = get_data(validation_data, batch_size = batch_size)
            
    
    
###model architecture###
            
model = Sequential()

#normalize the data and scale between -1 and 1
model.add(Lambda(lambda x: x / 127.5 - 0.5, input_shape = (160, 320, 3)))

#crop the data to show the region of interest
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))

#conv2D layer with filter 16, ksize 5x5, stride 2x2, and 'relu' activation
model.add(Conv2D(24, (5, 5), strides = (2, 2), activation = 'relu'))

#conv2D layer with filter 24, ksize 5x5, stride 2x2, and 'relu' activation
model.add(Conv2D(32, (5, 5), strides = (2, 2), activation = 'relu'))

#conv2D layer with filter 32, ksize 5x5, stride 2x2, and 'relu' activation
model.add(Conv2D(48, (5, 5), strides = (2, 2), activation = 'relu'))

#conv2D layer with filter 48, ksize 3x3, stride 1x1, and 'relu' activation
model.add(Conv2D(64, (3, 3), strides = (1, 1), activation = 'relu'))

#flatten the input
model.add(Flatten())

#dense layer size 100
model.add(Dense(100))
model.add(Dropout(0.05))

#dense layer size 50
model.add(Dense(50))
model.add(Dropout(0.05))

#dense layer size 10
model.add(Dense(10))
model.add(Dropout(0.05))

#dense layer size 1
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

model.fit_generator(training_generator, steps_per_epoch = np.ceil(len(training_data)/batch_size), 
                   validation_data = validation_generator, 
                   validation_steps = np.ceil(len(validation_data)/batch_size), 
                   epochs = 7, 
                   verbose = 1)

model.summary()

model.save('model.h5')






