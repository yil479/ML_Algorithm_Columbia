#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt


# In[42]:


#load data
X_train_df = pd.read_csv('X_train.csv',header=None)
X_test_df = pd.read_csv('X_test.csv',header=None)
y_train_df = pd.read_csv('y_train.csv',header=None)
y_test_df = pd.read_csv('y_test.csv',header=None)

X_train = np.array(X_train_df)
X_test = np.array(X_test_df)
y_train = np.array(y_train_df)
y_test = np.array(y_test_df)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[43]:


#3 a)
def kernel(x, xi, b):
    dist = np.linalg.norm(x - xi)
    return np.exp(-dist/b)


# In[44]:


def Gaussian_Process(X_train, X_test, y_train, sigma2, b):
    n = len(X_train)
    Kn = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Kn[i,j] = kernel(X_train[i], X_train[j], b)

    K = np.zeros((1,n))
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        for j in range(n):
            K[0,j] = kernel(X_test[i], X_train[j], b)
        y_pred[i] = multi_dot([K, np.linalg.inv(sigma2*np.identity(n) + Kn),y_train])
    return y_pred    


# In[46]:


#3 b)
b_list = [5,7,9,11,13,15]
sigma2_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
y_pred = np.zeros((len(b_list), len(sigma2_list), len(y_test)))

for i in range(len(b_list)):
    for j in range(len(sigma2_list)):     
        y_pred[i,j,:] = Gaussian_Process(X_train, X_test, y_train, sigma2_list[j], b_list[i])


# In[47]:


RMSE = np.zeros((len(b_list), len(sigma2_list)))
for i in range(len(b_list)):
    for j in range(len(sigma2_list)):
        RMSE[i,j] = np.sqrt(1/X_test.shape[0]*np.sum([(y_t - y_p)**2 for y_t, y_p in zip(y_test, y_pred[i,j,:])]))


# In[48]:


data = {'b = 5': RMSE[0,:], 'b = 7': RMSE[1,:],'b = 9': RMSE[2,:],'b = 11': RMSE[3,:],'b = 13': RMSE[4,:],'b = 15': RMSE[5,:]}
pd.DataFrame.from_dict(data, orient='index', columns=['$σ^2$ = 0.1', '$σ^2$ = 0.2', '$σ^2$ = 0.3', '$σ^2$ = 0.4', '$σ^2$ = 0.5', '$σ^2$ = 0.6', '$σ^2$ = 0.7', '$σ^2$ = 0.8', '$σ^2$ = 0.9', '$σ^2$ = 1'])


# 3 c) When b = 5 and sigma^2 = 0.4, RMSE = 1.930311, which is the best result of all. It is better than the result of the first homework. The drawback would be that GP may lead to overfit, since it doesn't have a penalization function like ridge regression.

# In[49]:


#3 d)
X_train_4 = X_train[:,3]
y_pred = Gaussian_Process(X_train_4, X_train_4, y_train, sigma2 = 2, b = 5)


# In[66]:


X_train_4_sort = np.sort(X_train_4)
idx = np.argsort(X_train_4)
y_pred_sort = y_pred[idx]

plt.scatter(X_train_4, y_train)
plt.plot(X_train_4_sort, y_pred_sorted, color='red')
plt.xlabel('car weight')
plt.ylabel('miles per gallon')
plt.title('miles per gallon vs car weight')
plt.show()

