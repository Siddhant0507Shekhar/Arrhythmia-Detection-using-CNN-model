# Arrhythmia Detection using CNN and MIT-BIH Database

## Introduction
This project focuses on the development of a Convolutional Neural Network (CNN) model for the detection of arrhythmia using the MIT-BIH Arrhythmia Database. The MIT-BIH Arrhythmia Database is a widely used dataset for studying heart rhythm abnormalities. It consists of electrocardiogram (ECG) recordings of patients with various types of arrhythmias.

## MIT-BIH Arrhythmia Database
The MIT-BIH Arrhythmia Database contains ECG recordings of 48 patients, with each recording varying in length. The database includes annotations for different classes of heartbeats, including both healthy and unhealthy rhythms. The classes include normal (N), right bundle branch block (RBBB), ventricular ectopic beat (V), left bundle branch block (LBBB), and atrial premature beat (A).

## Data Preprocessing
The data preprocessing involves segmenting the ECG signals based on the provided R-peak indices. Each heartbeat is converted into a 2D image using the matplotlib library, with the class label associated with it. The images are then resized to 225x150 pixels.

## Data Preparation
The preprocessed images are further processed using the PIL library to convert them to grayscale. These grayscale images are loaded as numpy arrays of size 150x224 pixels. The input images and corresponding labels are stored in separate arrays for training and testing.

## Model Architecture
The CNN model architecture used in this project is inspired by a research paper. The model is implemented using the Keras library. The architecture consists of convolutional layers, batch normalization, max-pooling layers, and fully connected layers. The final layer uses the softmax activation function for multi-class classification into the five classes: N, R, V, L, and A.

```python
# Model Architecture
from keras.models import Sequential
import keras
from keras.layers import Conv2D,BatchNormalization,MaxPool2D,Flatten,Dense,Dropout

model = Sequential()
model.add(Conv2D(64, (3,3),strides = (1,1), input_shape = img_size,kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Flatten())
model.add(Dense(2048))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


## Conclusion
This project demonstrates the use of a CNN model for arrhythmia detection using the MIT-BIH Arrhythmia Database. The model shows promising results in classifying different types of heartbeats accurately. We have reached 97.2% accuracy with only 35000 training points and due to our system hindrance we can't increase the traing data more but would belive that with this architecture we can accuracy of above 99%. The trained model can be further evaluated and deployed for real-world applications in the field of cardiac health monitoring. 

# Predicting output from the model
The main problem which you can face while predicting output from this model is how to convert your linear data into images such that R peak will be in middle of the image. In MIT BIH database we have been provided R peaks in txt file for each patient , but in real world case we can find it using Biosppy module. So in biosppy module , I have used Christov Segmenter like this 

```python
r_peaks = biosppy.signals.ecg.christov_segmenter(signal=np_arr, sampling_rate=200.0)[0]
```
Here np_arr is patient heart ecg data as an array , You can adjust sampling rate according to the device from which it records. Then after finding R peaks, you can segment the whole array into each heartbeat equal to number of heartbeats in that array.
