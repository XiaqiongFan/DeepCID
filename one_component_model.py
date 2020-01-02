# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:29:07 2017

@author: admin
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import os

tf.reset_default_graph()

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
def weights_variables(shape):
    weight = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return weight
    
def bias_variables(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,s):
    return tf.nn.conv2d(x,W,strides=[1,1,2,1],padding='SAME') 
    
def max_pool_1x2(x):
    return tf.nn.max_pool(x,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')    
 

xs= tf.placeholder(tf.float32,[None,881],name='xs') 
ys= tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
x_image = tf.reshape(xs,[-1,1,881,1])

##conv1 layer##
W_conv1 = weights_variables([1,5,1,32])
b_conv1 = bias_variables([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1,1) + b_conv1)
h_pool1 = max_pool_1x2(h_conv1)                        
h_W1_drop = tf.nn.dropout(h_pool1,keep_prob)

##conv2 layer##
W_conv2 = weights_variables([1,5,32,64])
b_conv2 = bias_variables([64])
h_conv2 = tf.nn.relu(conv2d(h_W1_drop,W_conv2,1) + b_conv2)
h_pool2 = max_pool_1x2(h_conv2)                       
h_W2_drop = tf.nn.dropout(h_pool2,keep_prob)

##func1 layer##
W_fc1 = weights_variables([1*56*64,1024])
b_fc1 = bias_variables([1024])
h_pool2_flat = tf.reshape(h_W2_drop,[-1,1*56*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

##func2 layer##
W_fc2 = weights_variables([1024,2])
b_fc2 = bias_variables([2])

pred = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
prediction = tf.nn.softmax(pred, name= 'prediction')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = ys))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
saver = tf.train.Saver()


path = u'E:/DeepCID'
compound = 0

datafile1 = path+'/augmented_data/'+str(compound)+'component.npy'
X = np.load(datafile1)
Xtrain0 = X[0:15000]
Xvalid0 = X[15000:17500]
Xtest0 = X[17500:20000]
scaler = preprocessing.StandardScaler().fit(Xtrain0)
Xtrain = scaler.transform(Xtrain0)
Xvalid = scaler.transform(Xvalid0)
Xtest = scaler.transform(Xtest0)

datafile2 = path+'/augmented_data/'+str(compound)+'label.npy'
Y1 = np.load(datafile2) 
Y2 = np.ones((Y1.shape)) - Y1
Y = np.concatenate((Y1,Y2),axis=1)
Ytrain = Y[0:15000]
Yvalid = Y[15000:17500]
Ytest = Y[17500:20000]


accuracy_valid = []
loss_valid  = []
save_file = './model/compoent_'+str(compound)+'/compoent.ckpt'

batch_size = 100
epochs = 300
accuracy_valid.clear()
loss_valid.clear()


num_steps = Xtrain.shape[0] // batch_size
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(epochs):
        for step in range(num_steps):
            offset = (step * batch_size) % (Ytrain.shape[0] - batch_size)   
            batch_xs = Xtrain[offset:(offset + batch_size), :]
            batch_ys = Ytrain[offset:(offset + batch_size), :]
            feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5}
            _, loss, predictions = sess.run([train_step, cross_entropy, prediction], feed_dict=feed_dict)       
        
            feed_dict_val={xs:Xvalid,ys:Yvalid,keep_prob:1.0}       
            
            
        loss_val,pred_val = sess.run([cross_entropy,prediction], feed_dict=feed_dict_val)
        acc_val = accuracy(pred_val,Yvalid)
        print('The epoch', epoch, 'finished. The accuracy %.1f%%' % (acc_val),'loss =',loss_val)
        accuracy_valid.append(acc_val/100)
        loss_valid.append(loss_val)
    
    TIMES = [(i + 1) for i in range(0, epochs)]    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(TIMES,accuracy_valid,'r')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')    
    ax2 = ax1.twinx()  
    ax2.plot(TIMES, loss_valid, 'g')
    ax2.set_ylabel('Loss')

    saver.save(sess, save_file)
    print('Trained Model Saved.')


with tf.Session() as sess:
    saver.restore(sess, save_file)
    valid_ypred = sess.run(prediction,feed_dict={ xs: Xtest, ys: Ytest, keep_prob : 1.0})
print('The test accuracy %.1f%%' % accuracy(valid_ypred,Ytest))



