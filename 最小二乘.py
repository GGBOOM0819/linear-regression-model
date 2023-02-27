#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
 
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
 
df = pd.read_csv("C:/Users/10230/Desktop/forestfires.csv", header=0)
 
 
X = df.iloc[:, 4:12]
Y = df.iloc[:, 12]
xMat = np.mat(X)
yMat = np.mat(Y).reshape(-1, 1)
 
m, n = X.shape[0], X.shape[1]
xMat = np.concatenate((np.ones((m, 1)), xMat), axis=1)
#theta = np.dot(np.dot(np.linalg.inv(np.dot(xMat.T, xMat)), xMat.T), yMat)
theta_n = (xMat.T * xMat).I * xMat.T * yMat
# print(theta_n)
#pred = np.dot(xMat, theta)
#Loss = 0.5 * np.mean(np.square(pred - yMat))
 
 
def costFunc(xMat, yMat, theta):
   inner = np.power((xMat * theta.T) - yMat, 2)
   return np.sum(inner) / (2 * len(xMat))
 
 
def gradientDescent(xMat, yMat, theta, alpha, iters):
   temp = np.mat(np.zeros(theta.shape))
   cost = np.zeros(iters)
   thetaNums = int(theta.shape[1])
   print(thetaNums)
  for i in range(iters):
       error = (xMat * theta.T - yMat)
       for j in range(thetaNums):
           derivativeInner = np.multiply(error, xMat[:, j])
           temp[0, j] = theta[0, j] -                (alpha * np.sum(derivativeInner) / len(xMat))
 
       theta = temp
       cost[i] = costFunc(xMat, yMat, theta)
 

