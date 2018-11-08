# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:03:39 2018

@author: admin
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import os
import csv

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])  

if __name__ == '__main__':
    #Load the mixture spectrum, its' labels and components information 
    datafile1 =u'C:/Users/admin/Desktop/DeepMA/mixture.npy'
    Xtest = np.load(datafile1) 

    datafile2 =u'C:/Users/admin/Desktop/DeepMA/label.npy'
    label = np.load(datafile2)   

    n = label.shape[0]
    
    csv_reader = csv.reader(open('C:/Users/admin/Desktop/DeepMA/namedata.csv', encoding='utf-8'))
    names = [row for row in csv_reader]    
    ypred = np.zeros((n*Xtest.shape[0],2))
    
  
    # Set the root directory of models and reload the models one by one 
    root = "C:/Users/admin/Desktop/DeepMA/model"
    list_dirs = os.walk(root) 
    i=0
    for root, dirs, files in list_dirs: 
        for d in dirs: 
            tf.reset_default_graph()
            print (os.path.join(root, d),i)   
            tf.reset_default_graph()
            
            Y1 = label[i,:].reshape([Xtest.shape[0],1])
            Y2 = np.ones((Y1.shape)) - Y1
            Ytest = np.concatenate((Y1,Y2),axis=1)
       
            os.chdir(os.path.join(root, d))
            datafile =u'./X_scale.npy'
            X_scale = np.load(datafile)
            Xtest_scale = (Xtest - X_scale[0])/X_scale[1]
             
            with tf.Session() as sess:
                new_saver=tf.train.import_meta_graph('./compoent.ckpt.meta')
                new_saver.restore(sess,"./compoent.ckpt")
                graph = tf.get_default_graph()
                xs=graph.get_operation_by_name('xs').outputs[0]
                #ys=graph.get_operation_by_name('ys').outputs[0]
                keep_prob=graph.get_operation_by_name('keep_prob').outputs[0]
     
                prediction = graph.get_tensor_by_name('prediction:0')
                test_ypred = sess.run(prediction,feed_dict={xs: Xtest_scale, keep_prob : 1.0})  
            ypred[i*Xtest.shape[0]:(i+1)*Xtest.shape[0],:] = test_ypred
            print('compoent', i, 'finished.','The test accuracy %.1f%%' % 
                  accuracy(ypred[i*Xtest.shape[0]:(i+1)*Xtest.shape[0],:],Ytest))   
    
            i+=1
    
  
    # Print the components' name that are exited in the miatures
    for j in range(Xtest.shape[0]):
        print('The', j ,'th sample contains: ')
                
        # Yreal is design for calculate the accuracy.         
        y_real1 = label[:,j-Xtest.shape[0]].reshape([n,1])
        y_real2 = np.ones((y_real1.shape)) - y_real1
        y_real = np.concatenate((y_real1,y_real2),axis=1)   
                
        ypre = np.zeros((n,2))
        for k in range(n):            
            ypre[k,:] = ypred[j,:]
            j = j+Xtest.shape[0]
        
        for h in range(n):
            if (ypre[h,0]>=0.5):
                print(names[h])                   
        print('The prediction finished')
      

    
    
    
    
    
    
    
    
    
    
    
