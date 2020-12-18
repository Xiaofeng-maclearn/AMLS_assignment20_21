#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import cv2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#Data Preprocessing
def preProcess():
    df = pd.read_csv('./Datasets/cartoon_set/labels.csv',sep='\t')
    y = np.array(df['eye_color'])
    
    X = np.zeros((10000,18*18))
    
    for i in range(10000):
        img = cv2.imread('./Datasets/cartoon_set/img/{}.png'.format(i),cv2.IMREAD_GRAYSCALE)
        img_compressed1 = img[257:275,195:213]
        x = img_compressed1.reshape(1,img_compressed1.size)
        if not(np.ones(x.size)*x[0]-x.T).any():
            y[i] = 5
        X[i,:] = x
        
    x_train, x_val, y_train, y_val = train_test_split(X, y,random_state=0)
    
    df = pd.read_csv('./Datasets/cartoon_set_test/labels.csv',sep='\t')
    y_test = np.array(df['eye_color'])
    
    x_test = np.zeros((2500,18*18))
    
    for i in range(2500):
        img = cv2.imread('./Datasets/cartoon_set_test/img/{}.png'.format(i),cv2.IMREAD_GRAYSCALE)
        img_compressed1 = img[257:275,195:213]
        x = img_compressed1.reshape(1,img_compressed1.size)
        if not(np.ones(x.size)*x[0]-x.T).any():
            y_test[i] = 5
        x_test[i,:] = x
        
    return x_train, x_val, y_train, y_val, x_test, y_test

#Model training
def train(xTrain, yTrain, xVal, yVal):
    logreg = LogisticRegression(solver='lbfgs',max_iter=500)
    logreg.fit(xTrain, yTrain)
    y_pred = logreg.predict(xVal)
    accuracy = accuracy_score(y_pred, yVal)
    return accuracy,logreg

#Model testing
def test(xTest, yTest, logreg):
    y_pred = logreg.predict(xTest)
    return accuracy_score(y_pred, yTest)

