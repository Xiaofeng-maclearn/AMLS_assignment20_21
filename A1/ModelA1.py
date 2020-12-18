#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data Preprocessing
def preProcess():
    df = pd.read_csv('./Datasets/celeba/labels.csv',sep='\t')
    y = np.array(df['gender'])
        
    X = np.zeros((5000,9701))
    for i in range(5000):
        img = cv2.imread('./Datasets/celeba/img/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE)
        img_Rcompressed = img[np.arange(1,img.shape[0]+1,2)]
        img_compressed = img_Rcompressed[:,np.arange(1,img.shape[1]+1,2)]
        X[i,:] = img_compressed.reshape(1,img_compressed.size)
            
    x_train, x_val, y_train, y_val = train_test_split(X, y,random_state=0)
        
    df = pd.read_csv('./Datasets/celeba_test/labels.csv',sep='\t')
    y_test = np.array(df['gender'])
        
    x_test = np.zeros((1000,9701))
    for i in range(1000):
        img = cv2.imread('./Datasets/celeba_test/img/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE)
        img_Rcompressed = img[np.arange(1,img.shape[0]+1,2)]
        img_compressed = img_Rcompressed[:,np.arange(1,img.shape[1]+1,2)]
        x_test[i,:] = img_compressed.reshape(1,img_compressed.size)
        
    return x_train, x_val, y_train, y_val, x_test, y_test

# Define sigmoid function
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Function which makes a prediction based on feature vector learnt
def logRegrNEWRegrPredict(xTest, theta):
    intercept = np.ones((xTest.shape[0], 1))
    xTest = np.concatenate((intercept, xTest), axis=1)
    y_pred = sigmoid(np.dot(xTest, theta))
    y_pred[y_pred >= 0.5]=1
    y_pred[y_pred < 0.5]=-1
    return y_pred

# Training function
def train(xTrain,yTrain,xVal,yVal):
    intercept = np.ones((xTrain.shape[0], 1))
    n = xTrain.shape[0]
    xTrain = np.concatenate((intercept, xTrain), axis=1)
    yTrain[yTrain < 1] = 0
    theta = np.zeros(xTrain.shape[1])
    lr = 0.0001
    prev_acc = 0
    max_theta = np.zeros(xTrain.shape[1])
    prev_theta = np.zeros(xTrain.shape[1])
    while True:
        max_acc = 0
        for maxIt in range(1000):
            z = np.dot(xTrain, theta)
            h = sigmoid(z)
            gradient = np.dot(xTrain.T, np.add(h, -yTrain))/n
            theta = theta - lr*gradient
            y_pred = logRegrNEWRegrPredict(xVal,theta)
            acc = accuracy_score(y_pred, yVal)
            if acc > max_acc:
                max_acc=acc
                max_theta = theta
        if prev_acc > max_acc:
            break
        prev_acc = max_acc
        prev_theta = max_theta
        
    return prev_acc, prev_theta

#Testing function
def test(xTest, yTest, theta):
    y_pred = logRegrNEWRegrPredict(xTest, theta)
    return accuracy_score(y_pred, yTest)
        


# In[ ]:




