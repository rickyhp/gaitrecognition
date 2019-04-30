# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:30:51 2019

@author: rp
"""
import pandas as pd
import numpy as np
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import backend as K
import os
from sklearn.model_selection import train_test_split

# load features
DATAPATH = r'F:\\CODE\\GaitChallenge2019\\features'
TRAIN_DATAFILE = 'train_all_in_one_features_50.npy'
TEST_DATAFILE = 'test_all_in_one_features_50.npy'

train_dataset = np.load(DATAPATH + os.sep + TRAIN_DATAFILE)
test_dataset = np.load(DATAPATH + os.sep + TEST_DATAFILE)
all_dataset = np.concatenate((train_dataset, test_dataset),axis=0)

X_all = all_dataset[:,1:-2]
#X_train = train_dataset[:,1:-2]

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_all = lb.fit_transform(all_dataset[:,0])
#y_train = lb.fit_transform(train_dataset[:,0])

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=68)

X_train_input = X_train.reshape(-1,1,8,12).astype("float32")
X_test_input = X_test.reshape(-1,1,8,12).astype("float32")

# new model
model = Sequential()
K.set_image_dim_ordering('th')
model.add(Conv2D(240,(1,10), input_shape=(1,8,12), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(300,(1,7), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(360,(1,5), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Conv2D(420,(1,3), activation="relu", padding="same"))
model.add(AveragePooling2D(pool_size=(1,5), strides=None, padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(496, activation= 'softmax' ))

# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'SGD' , metrics=[ 'accuracy' ])

model.fit(X_train_input, y_train, epochs=20, batch_size= 100)

score, acc = model.evaluate(X_test_input, y_test, batch_size=100, verbose=1)

print('Test score:', score)
print('Test accuracy:', acc)

model.summary()

