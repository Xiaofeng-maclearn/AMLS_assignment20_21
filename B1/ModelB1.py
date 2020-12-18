#!/usr/bin/env python
# coding: utf-8

# In[2]:

# Import libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Preprocessing
def preProcess():
    df = pd.read_csv('./Datasets/cartoon_set/labels.csv',sep='\t')
    y = np.array(df['face_shape'])
    
    X = np.zeros((10000,15625))
    for i in range(10000):
        img = cv2.imread('./Datasets/cartoon_set/img/{}.png'.format(i),cv2.IMREAD_GRAYSCALE)
        img_compressed1 = img[img.shape[0]//2:3*img.shape[0]//4,img.shape[1]//4:img.shape[1]//2]
        X[i,:] = img_compressed1.reshape(1,img_compressed1.size)
    
    x_train, x_val, y_train, y_val = train_test_split(X, y,random_state=0)
    
    df = pd.read_csv('./Datasets/cartoon_set_test/labels.csv',sep='\t')
    y_test = np.array(df['face_shape'])
    
    x_test = np.zeros((2500, 15625))
    for i in range(2500):
        img = cv2.imread('./Datasets/cartoon_set_test/img/{}.png'.format(i),cv2.IMREAD_GRAYSCALE)
        img_compressed1 = img[img.shape[0]//2:3*img.shape[0]//4,img.shape[1]//4:img.shape[1]//2]
        x_test[i,:] = img_compressed1.reshape(1,img_compressed1.size)
    
    return x_train, x_val, y_train, y_val, x_test, y_test

#Model Training
def train(xTrain, yTrain, xVal, yVal):
    logreg = LogisticRegression(solver='lbfgs',max_iter=500)
    logreg.fit(xTrain, yTrain)
    y_pred= logreg.predict(xVal)
    accuracy = accuracy_score(y_pred, yVal)
    return accuracy,logreg

#Model Testing
def test(xTest, yTest, logreg):
    y_pred = logreg.predict(xTest)
    return accuracy_score(y_pred, yTest)


# In[ ]:




