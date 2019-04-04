#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 01:54:08 2019

@author: rickyputra
"""
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import os
import glob
import errno
import tensorflow as tf

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

#files = glob.glob(ou_isir_data_out_files)
files = glob.glob(OU_ISIR_DATA_OUT + '*.csv')

df = None

df = [process_file(name) for name in files]
result = pd.concat(df, axis=0, sort=False)

#result.to_csv('result_imuzcenter.csv')
result.to_csv('result_imuzright.csv')

#
result = pd.read_csv('result_imuzcenter.csv')

# data understanding
result.groupby('Subject').count().min()
result.groupby('Subject').count().max()

X = result[['Subject','Gx','Gy','Gz','Ax','Ay','Az','Act']]
Y = result[['Subject']]

X = X[X.Act=='Walk1']
X.groupby('Subject').count().min()

curSubject = ''
path = "F:\\CODE\\gaitrecognition\\"
WINDOW = 20
r = 0
i = 0
j = 1
X1 = pd.DataFrame(columns=['Gx','Gy','Gz','Ax','Ay','Az'])

for index, row in X.iterrows():
    i += 1
    print(i, ' : ',row['Subject'], row['Gx'], row['Gy'], row['Gz'], row['Ax'], row['Ay'], row['Az'])
    if(curSubject != row['Subject']):        
        curSubject = row['Subject']                
        X1 = X1.append({'Gx':row['Gx'],'Gy':row['Gy'],
                        'Gz':row['Gz'], 'Ax':row['Ax'], 'Ay':row['Ay'],
                        'Az':row['Az']}, ignore_index='True')
        try:
            os.mkdir(path + "dataset\\" + row['Subject'])
        except:
            print('folder exists, skipping')                
    else:
        X1 = X1.append({'Gx':row['Gx'],'Gy':row['Gy'],
                        'Gz':row['Gz'], 'Ax':row['Ax'], 'Ay':row['Ay'],
                        'Az':row['Az']}, ignore_index='True')

    if(i % WINDOW == 0):            
            X1.to_csv(path + "dataset\\" + row['Subject'] + "\\" + row['Subject'] + "." + str(j) + ".csv")
            j += 1
            X1 = X1[0:0]
            print('===== saved to csv =====')



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

# CNN
####################Set parameters###############################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 128*6])  
x_image = tf.reshape(x, [-1,16,8,6]) #reshape
sess = tf.InteractiveSession()

"""
# first layer
"""
W_conv1 = weight_variable([5, 5, 6, 32])  
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""
# second layer
"""
W_conv2 = weight_variable([5, 5, 32, 64]) 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


"""
# third layer, the fully connected layer with an input dimension of 4*2*64 and an output dimension of 1024
"""
W_fc1 = weight_variable([4*2*64, 1024])  
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
"""
# fourth layer，with an input dimension of 1024 and an output dimension of 21，corresponding to 21 categories
"""
W_fc2 = weight_variable([1024, 21])
b_fc2 = bias_variable([21])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #Use softmax as a multi-class activation function
y_ = tf.placeholder(tf.float32, [None, 21])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # Loss function，cross entropy
tf.summary.scalar('loss', cross_entropy)
train_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Use Adam optimizer
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
log_dir='/Users/rickyputra/CODE/gaitrec/tensorboard/'  # Use tensorboard
train_writer = tf.summary.FileWriter(log_dir + 'train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'test/')
sess.run(tf.initialize_all_variables())  # Variable initialization
optimal_accuracy = 0.0
#train
for i in range(10000):  
    batch_x,batch_y = next_batch(50,train_np,ytrain_label)  
    if i%100 == 0:
        loss,test_accuracy,test_summary=sess.run([cross_entropy,accuracy,merged],feed_dict={
            x:test_np, y_: ytest_label, keep_prob: 1.0})
        test_writer.add_summary(test_summary, i)
        print("step %d, test accuracy: %g, loss: %g"%(i, test_accuracy,loss))
        optimal_accuracy = max(optimal_accuracy, test_accuracy)
    summary_, _ = sess.run([merged,train_optimizer],feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})   
    train_writer.add_summary(summary_, i)
train_writer.close()
test_writer.close()

print("best test accuracy: %g"%(optimal_accuracy))
print("final test accuracy %g"%accuracy.eval(feed_dict={x: test_np, y_: ytest_label, keep_prob: 1.0}))


