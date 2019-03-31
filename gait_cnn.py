# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:14:37 2019

@author: rp
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K


train2 = pd.read_csv("fashion-mnist_train.csv").values
train2 = train2[:, 1:]
train2X = train2.reshape(train2.shape[0],1,28,28).astype( 'float32' )

train = pd.read_csv("result_imuzcenter.csv").values
train = train[:-124774, 1:-2]
trainX = train.reshape(train.shape[0],2,3,1).astype( 'float32' )
