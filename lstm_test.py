# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:43:41 2019

@author: rp
"""

# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from matplotlib import pyplot
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers import TimeDistributed
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import AveragePooling2D
from sklearn.model_selection import train_test_split

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)    
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/subject_'+group+'.txt')
    return X, y

def load_merged_dataset():
    trainX, trainy = load_dataset_group('train','HARDataset/')
    testX, testy = load_dataset_group('test','HARDataset/')
    
    allX = np.append(trainX, testX, axis=0)
    ally = np.append(trainy, testy, axis=0)
    
    print(allX.shape, ally.shape)
    
    ally = ally - 1
    ally = to_categorical(ally)
    
    print(allX.shape, ally.shape)
    
    trainX, testX, trainy, testy = train_test_split(allX, ally, test_size=0.2, random_state=38)
        
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)

    return trainX, trainy, testX, testy
#
#def load_merge_dataset():
#    X = pd.DataFrame()
#    y = pd.DataFrame(data={'Subject':[]})
#    
#    train_group = 'train'
#    filepath = 'HARDataset/' + train_group + '/'
#    
#    X = X.append(pd.read_csv(filepath + 'X_train.txt',header=None, delim_whitespace=True),sort=False, ignore_index=False)        
#    
#    # load class output
#    y = y.append(pd.read_csv('HARDataset/' + train_group + '/subject_train_header.txt' ),sort=False, ignore_index=False)
#    
#    test_group = 'test'
#    filepath = 'HARDataset/' + test_group + '/'
#    
#    X = X.append(pd.read_csv(filepath + 'X_test.txt',header=None, delim_whitespace=True),sort=False, ignore_index=False)        
#    
#    # load class output
#    y = y.append(pd.read_csv('HARDataset/' + test_group + '/subject_test_header.txt' ),sort=False, ignore_index=False)
#    
#    print(X.shape, y.shape)
#    y = y - 1
#    y = to_categorical(y)
#    print(X.shape, y.shape)
#    
#    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=38)
#    
#    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
#    
#    trainX = trainX.values.reshape(-1,187,3).astype("float32")
#    
#    testX = testX.values.reshape(-1,187,3).astype("float32")
#    
#    return trainX, trainy, testX, testy

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_lstm_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def evaluate_cnn_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv2D(240,(10,10), input_shape=(1,n_timesteps,n_features), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(300,(5,5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(AveragePooling2D(pool_size=(1,5), strides=None, padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation= 'softmax' ))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    trainX = trainX.reshape(-1,n_features).astype("float32")
    trainX = trainX.reshape(-1,1,n_timesteps,n_features).astype("float32")
    testX = testX.reshape(-1,n_features).astype("float32")
    testX= testX.reshape(-1,1,n_timesteps,n_features).astype("float32")
    
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

def evaluate_cnnlstm_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# fit and evaluate a model
def evaluate_convlstm_model(trainX, trainy, testX, testy):
	# define model
	verbose, epochs, batch_size = 0, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape into subsequences (samples, time steps, rows, cols, channels)
	n_steps, n_length = 4, 32
	trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
	testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
    # load data
    #trainX, trainy, testX, testy = load_dataset() # for HAR identification
    trainX, trainy, testX, testy = load_merged_dataset() # for Subject identification
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_lstm_model(trainX, trainy, testX, testy)
        #score = evaluate_cnn_model(trainX, trainy, testX, testy)
        #score = evaluate_cnnlstm_model(trainX, trainy, testX, testy)
        #score = evaluate_convlstm_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment()
