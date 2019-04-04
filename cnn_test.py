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
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

def data_normalize(data_temp):
    data_temp2=np.array(data_temp.T, dtype=np.float32)
    data_temp2 -=np.mean(data_temp2,axis=0)
    data_temp2 /=np.std(data_temp2,axis=0)
    data_temp=data_temp2.T
    return data_temp

#train = pd.read_csv("result_imuzcenter.csv").values
traindf = pd.read_csv("result_imuzcenter.csv")
traindf = traindf.append(pd.read_csv("result_imuzleft.csv"))
traindf = traindf.append(pd.read_csv("result_imuzright.csv"))

traindf = traindf[traindf['Act'].isin(['Walk1','Walk2'])]
train = traindf.values

trainX = data_normalize(train[:,1:-2])
trainX = trainX.reshape(-1,1,2,3).astype("float32")
x_train = trainX

trainY = train[:,-2]

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(trainY)

# split train and test
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=78)

# new model
model = Sequential()
K.set_image_dim_ordering('th')
model.add(Conv2D(30,(2,2), input_shape=(1,2,3), activation="relu", padding="valid"))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(15,(1,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation= 'relu' ))
model.add(Dense(496, activation= 'softmax' ))

# load json and create model
#json_file = open('model.json', 'r')
#model = json_file.read()
#json_file.close()
#model = model_from_json(model)
## load weights into new model
#model.load_weights("model_1.h5")
#print("Loaded model from disk")

# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

model.fit(x_train, y_train,epochs=20,batch_size= 100)

# serialize model to JSON
model_json = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_1.h5")
print("Saved model to disk")

score = model.evaluate(x_test, y_test, batch_size=100)