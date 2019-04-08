#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 01:54:08 2019

@author: rickyputra
"""
#%matplotlib inline
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import os
import glob
import errno
import tensorflow as tf
from scipy.stats import kurtosis, skew
import math

## OU-InertialGaitData
ou_isir_data = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/Android/*.csv'
ou_isir_data_out = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/Android/out/'
ou_isir_data_out_files = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/Android/out/*.csv'

ou_isir_imuzcenter = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/IMUZCenter/*.csv'
ou_isir_imuzcenter_out = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/IMUZCenter/out/'

ou_isir_imuzleft = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/IMUZLeft/*.csv'
ou_isir_imuzleft_out = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/IMUZLeft/out/'

ou_isir_imuzright = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/IMUZRight/*.csv'
ou_isir_imuzright_out = '/Users/rickyputra/CODE/OU-IneritialGaitData/ManualExtractionData/IMUZRight/out/'
#

# Remove first and second rows, and create new files in out subdir
OU_ISIR_DATA = ou_isir_imuzright
OU_ISIR_DATA_OUT = ou_isir_imuzright_out

WINDOW_LENGTH = 100

files = glob.glob(OU_ISIR_DATA)
for name in files:
    try:
        with open(name) as f:
            data = f.read().splitlines(True)
        arrFileName = name.split('/')
        fileName = arrFileName[len(arrFileName)-1]
        with open(OU_ISIR_DATA_OUT + fileName, 'w+') as fout:
            fout.writelines(['Gx,','Gy,','Gz,','Ax,','Ay,','Az,','Subject,','Act\n'])
            fout.writelines(data[2:])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

# Extract and create subject id and activity features from filename           
def process_file(name):
    try:
        print(name)
        ou_isir = pd.read_csv(name)
        arrFileName = name.split('_')
        activity = arrFileName[len(arrFileName)-1].split('.')[0]
        subject = arrFileName[len(arrFileName)-2]
        df = pd.DataFrame(ou_isir)
        df['Subject'] = subject
        df['Act'] = activity
        return df
    except IOError as exc:
        print('Error : ' + exc.errno)
        if exc.errno != errno.EISDIR:
            raise
            
# features engineering
def get_summary(data, start, end):
    # print('In summary: ', data.shape)
    data_window = data[start:end]
    acf = np.correlate(data_window, data_window, mode='full')
    acv = np.cov(data_window.T, data_window.T)
    sq_err = (data_window - np.mean(data_window)) ** 2
    return [
        np.mean(data_window),
        np.std(data_window),
        np.var(data_window),
        np.min(data_window),
        np.max(data_window),
        np.mean(acf),
        np.std(acf),
        np.mean(acv),
        np.std(acv),
        skew(data_window),
        kurtosis(data_window),
        math.sqrt(np.mean(sq_err))
    ]


def get_magnitude(data):
    x_2 = data[:, 0] * data[:, 0]
    y_2 = data[:, 1] * data[:, 1]
    z_2 = data[:, 2] * data[:, 2]
    m_2 = x_2 + y_2 + z_2
    m = np.sqrt(m_2)
    return np.reshape(m, (m.shape[0], 1))


def get_window(data):
    start = 0
    size = data.shape[0]
    while start < size:
        end = start + WINDOW_LENGTH
        yield start, end
        start += int(WINDOW_LENGTH // 2)  # 50% overlap, hence divide by two


def get_features(raw_data):
    data_parts = [None] * 2
    data_parts[0], data_parts[1] = raw_data[:, :3], raw_data[:, 3:]
    final_features = None
    for data in data_parts:
        data = np.concatenate((data, get_magnitude(data)), axis=1)
        features = None
        for (start, end) in get_window(data):
            window_features = []
            for j in range(data.shape[1]):
                window_features += get_summary(data[:, j], start, end)
            if features is None:
                features = np.array(window_features)
            else:
                features = np.vstack((features, np.array(window_features)))
        if final_features is None:
            final_features = np.array(features)
        else:
            final_features = np.hstack((final_features, features))

    if (len(final_features.shape)) < 2:
        final_features = final_features.reshape(1, -1)
    return final_features

def data_normalize(data_temp):
    data_temp2=np.array(data_temp.T, dtype=np.float32)
    data_temp2 -=np.mean(data_temp2,axis=0) # mean in column
    data_temp2 /=np.std(data_temp2,axis=0) # std in column
    data_temp=data_temp2.T
    return data_temp

#files = glob.glob(ou_isir_data_out_files)
files = glob.glob(OU_ISIR_DATA_OUT + '*.csv')

df = None

df = [process_file(name) for name in files]
result = pd.concat(df, axis=0, sort=False)

#result.to_csv('result_imuzcenter.csv')
result.to_csv('result_imuzright.csv')

#
result = pd.read_csv('result_imuzcenter.csv')
#result = pd.read_csv('result_imuzcenter.csv')
#result = result.append(pd.read_csv('result_imuzleft.csv'))
#result = result.append(pd.read_csv('result_imuzright.csv'))

# data understanding
#result.groupby('Subject').count().min()
#result.groupby('Subject').count().max()

X = result[['Subject','Gx','Gy','Gz','Ax','Ay','Az','Act']]

Y = result[['Subject']]

X = X[X['Act'].isin(['Walk1'])]

data = X[['Gx','Gy','Gz','Ax','Ay','Az']].values

get_summary(data[:,1],1,100)

#X.groupby('Subject').count().min()

curSubject = ''
path = "F:\\CODE\\gaitrecognition\\"
#WINDOW = 20
#r = 0
#i = 0
#j = 1
#X1 = pd.DataFrame(columns=['Gx','Gy','Gz','Ax','Ay','Az'])
#
#for index, row in X.iterrows():
#    i += 1
#    print(i, ' : ',row['Subject'], row['Gx'], row['Gy'], row['Gz'], row['Ax'], row['Ay'], row['Az'])
#    if(curSubject != row['Subject']):        
#        curSubject = row['Subject']                
#        X1 = X1.append({'Gx':row['Gx'],'Gy':row['Gy'],
#                        'Gz':row['Gz'], 'Ax':row['Ax'], 'Ay':row['Ay'],
#                        'Az':row['Az']}, ignore_index='True')
#        try:
#            os.mkdir(path + "dataset\\" + row['Subject'])
#        except:
#            print('folder exists, skipping')                
#    else:
#        X1 = X1.append({'Gx':row['Gx'],'Gy':row['Gy'],
#                        'Gz':row['Gz'], 'Ax':row['Ax'], 'Ay':row['Ay'],
#                        'Az':row['Az']}, ignore_index='True')
#
#    if(i % WINDOW == 0):            
#            X1.to_csv(path + "dataset\\" + row['Subject'] + "\\" + row['Subject'] + "." + str(j) + ".csv")
#            j += 1
#            X1 = X1[0:0]
#            print('===== saved to csv =====')



d = {'Subject':[]}
X1 = pd.DataFrame(data=d)
r = 0
for index, row in X.iterrows():
    print(r, ' : ',row['Subject'], row['Gx'], row['Gy'], row['Gz'], row['Ax'], row['Ay'], row['Az'])
    if(curSubject != row['Subject']):
        i = 1
        r += 1
        curSubject = row['Subject']
        X1 = X1.append({'Subject':row['Subject'],'Gx'+str(i):row['Gx'],'Gy'+str(i):row['Gy'],
                        'Gz'+str(i):row['Gz'], 'Ax'+str(i):row['Ax'], 'Ay'+str(i):row['Ay'],
                        'Az'+str(i):row['Az']},ignore_index=True)
    else:
        i += 1
        X1.loc[r-1,'Gx'+str(i)] = row['Gx']
        X1.loc[r-1,'Gy'+str(i)] = row['Gy']
        X1.loc[r-1,'Gz'+str(i)] = row['Gz']
        X1.loc[r-1,'Ax'+str(i)] = row['Ax']
        X1.loc[r-1,'Ay'+str(i)] = row['Ay']
        X1.loc[r-1,'Az'+str(i)] = row['Az']

X1.to_csv(path + "dataset\\Walk1_C.csv")

X1 = pd.read_csv("Walk1_C.csv")
X1_arr = X1.values

def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise
 
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

X12 = X1_arr
# generate gaussian noise sequences for each subject
for i in range(1,10):
    X2_arr = X1_arr
    X2_arr[:,2:] = DA_Jitter(X2_arr[:,2:])
    X12 = np.concatenate((X12, X2_arr))

for i in range(1,10):
    X2_arr = X1_arr
    X2_arr[:,2:] = DA_Scaling(X2_arr[:,2:])
    X12 = np.concatenate((X12, X2_arr))

X12 = pd.DataFrame(X12)
X12.to_csv("Walk1_C_noise.csv")
