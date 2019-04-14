#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:56:36 2019

@author: rickyputra
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.layers.convolutional import AveragePooling2D
from keras.layers import LSTM

def data_normalize(data_temp):
    data_temp2=np.array(data_temp.T, dtype=np.float32)
    data_temp2 -=np.mean(data_temp2,axis=0) # mean in column
    data_temp2 /=np.std(data_temp2,axis=0) # std in column
    data_temp=data_temp2.T
    return data_temp

traindf = pd.read_csv("dataset\Walk1_C_100_noise.csv")
#traindf = traindf.append(pd.read_csv("dataset\Walk1_C_noise.csv"),sort=False, ignore_index=False)
#traindf = traindf.append(pd.read_csv("dataset\Walk2_C.csv"),sort=False, ignore_index=False)
#traindf = traindf.append(pd.read_csv("dataset\Walk1_CLR.csv"),sort=False, ignore_index=False)

train = traindf.values
#train = train[:,1:-6510] # take subject + 100 measurement

# normalize sensor data
trainX = data_normalize(train[:,2:])

# take 1st column i.e. Subject ID
trainY = train[:,-601]

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)


trainX, testX, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=38)

#trainX = trainX.reshape(-1,100,2,3).astype("float32")
x_train = trainX.reshape(-1,1,100,6).astype("float32")

x_test = testX.reshape(-1,1,100,6).astype("float32")

# Model 1 CNN
model = Sequential()
K.set_image_dim_ordering('th')
#model.add(Conv2D(240,(1,10), input_shape=(100,2,3), activation="relu", padding="same"))
model.add(Conv2D(240,(10,10), input_shape=(1,100,6), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(300,(5,5), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AveragePooling2D(pool_size=(1,5), strides=None, padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(496, activation= 'softmax' ))

# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'Adam' , metrics=[ 'accuracy' ])

#
model.fit(x_train, y_train,epochs=10, batch_size= 100)

#
score, acc = model.evaluate(x_test, y_test, batch_size=100, verbose=1)

print('val score:', score)
print('val accuracy:', acc)

# Model 2 LSTM
verbose, epochs, batch_size = 0, 15, 100
x_train = trainX.reshape(-1,100,6).astype("float32")
x_test = testX.reshape(-1,100,6).astype("float32")

model = Sequential()
model.add(LSTM(100, input_shape=(100,6)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(496, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
	
print('val score:', score)
print('val accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_1.h5")
print("Saved model to disk")

#model.summary()
# load json and create model
json_file = open('model_1.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
# load weights into new model
model.load_weights("model_1.h5")
print("Loaded model from disk")

# test data
testdf = pd.read_csv("dataset\Walk2_C.csv")
test = testdf.values
test = test[:,1:-4512] #Walk2_C
x_test = data_normalize(test[:,1:])
x_test = x_test.reshape(-1,1,100,6).astype("float32")
#
testY = test[:,-601]
y_test = lb.fit_transform(testY)

model.compile(loss= 'categorical_crossentropy' , optimizer= 'Adam' , metrics=[ 'accuracy' ])

score, acc = model.evaluate(x_test, y_test, batch_size=100, verbose=1)

print('Test score:', score)
print('Test accuracy:', acc)

#
#y_pred = model.predict(x_test, verbose=1)
#
#print(testY[np.argmax(y_pred[0], axis=1)])
#
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
#
#from sklearn.metrics import classification_report
#print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=testY))


