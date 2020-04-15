#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x_train = pd.read_csv('X_train.csv', header = None)
y_train = pd.read_csv('y_train.csv', header = None)
x_test = pd.read_csv('X_test.csv', header = None)
y_test = pd.read_csv('y_test.csv', header = None)


# In[5]:


size_n = len(x_train.columns)
I = np.identity(size_n)
#SVD
u, s, vh = np.linalg.svd(x_train,full_matrices = False)
xtx=vh.transpose().dot(np.power(np.diag(s), 2).dot(vh))
#create empty list
l_list = np.zeros(shape=(5001,size_n))
dof = np.zeros(shape=(5001,))


# In[6]:


for lamda in range(5001):
    
    l_list[lamda] = ((np.linalg.inv(lamda*I+xtx)).dot(x_train.transpose()).dot(y_train)).transpose()
    dof[lamda]=np.trace(x_train.dot(np.linalg.inv(lamda*I+xtx).dot(x_train.transpose())))


# In[7]:


labels = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year made', '1']
for i in range(0,7):
    plt.plot(dof, l_list[:,i], label=labels[i])
plt.legend()
plt.xlabel('dof')
plt.ylabel('w_rr')
plt.show()


# In[8]:


RMSE = []
for lamda in range(100):
    y_predict = x_test.dot(l_list[lamda].transpose())
    RMSE.append(np.sqrt(np.sum((np.array(y_predict)-y_test.transpose().values)**2)/x_test.shape[0]))
plt.plot(np.arange(51), RMSE[0:51])
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()


# In[9]:


#p=2
x_train_2 = x_train.copy()
x_test_2 = x_test.copy()


# In[10]:


#add p=2
for i in range(size_n):
    x_train_2[size_n+i] = (x_train_2[i])**2
    x_test_2[size_n+i] = (x_test_2[i])**2


# In[11]:


for i in range(size_n-1):
    train_mean = np.mean(x_train_2.iloc[:,size_n+i])
    train_std = np.std(x_train_2.iloc[:,size_n+i])
    #standardize
    x_train_2.iloc[:,size_n+i] = (x_train_2.iloc[:,size_n+i] - train_mean) / train_std
    x_test_2.iloc[:,size_n+i] = (x_test_2.iloc[:,size_n+i] - train_mean) / train_std


# In[12]:


size_n_2=x_train_2.shape[1]
row_n_2=x_train_2.shape[0] 
I_2 = np.identity(size_n_2) 
#svd
u, s, vh = np.linalg.svd(x_train_2,full_matrices = True)
xtx=vh.transpose().dot(np.power(np.diag(s), 2).dot(vh))
#create empty matrix
l_list_2 = np.zeros(shape=(row_n_2,size_n_2))


# In[13]:


for lamda in range(row_n_2):
    w_rr = (np.linalg.inv(lamda*I_2+xtx)).dot(x_train_2.transpose()).dot(y_train)
    l_list_2[lamda] = w_rr.transpose()


# In[14]:


#rmse for p=2
RMSE_2 = []
for lamda in range(100):
    y_predict_2 = x_test_2.dot(l_list_2[lamda].transpose())
    RMSE_2.append(np.sqrt(np.sum((np.array(y_predict_2)-y_test.transpose().values)**2)/x_test.shape[0]))
plt.plot(np.arange(51), RMSE_2[0:51])
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()


# In[15]:


#for p=3
x_train_3 = x_train_2.copy()
x_test_3 = x_test_2.copy()


# In[16]:


#add p=3
for i in range(size_n):
    x_train_3[size_n_2+i] = x_train[i]**3
    x_test_3[size_n_2+i] = x_test[i]**3


# In[17]:


for i in range(size_n-1):
    train_mean = np.mean(x_train_3.iloc[:,size_n_2+i])
    train_std = np.std(x_train_3.iloc[:,size_n_2+i])
    #standardize
    x_train_3.iloc[:,size_n_2+i] = (x_train_3.iloc[:,size_n_2+i] - train_mean) / train_std
    x_test_3.iloc[:,size_n_2+i] = (x_test_3.iloc[:,size_n_2+i] - train_mean) / train_std


# In[20]:


#p=3
size_n_3=x_train_3.shape[1]
row_n_3=x_train_3.shape[0] 
I_3 = np.identity(size_n_3) 
#svd
u, s, vh = np.linalg.svd(x_train_3,full_matrices = True)
xtx=vh.transpose().dot(np.power(np.diag(s), 2)).dot(vh)
#create empty list
l_list_3 = np.zeros(shape=(row_n_3,size_n_3))


# In[21]:


for lamda in range(row_n_3):
    w_rr = (np.linalg.inv(lamda*I_3+xtx)).dot(x_train_3.transpose()).dot(y_train)
    l_list_3[lamda] = w_rr.transpose()


# In[22]:


#RMSE when p=3
RMSE_3 = []
for lamda in range(100):
    y_pred_3 = x_test_3 .dot(l_list_3[lamda].transpose())
    RMSE_3.append(np.sqrt(np.sum((np.array(y_pred_3)-y_test.transpose().values)**2)/x_test.shape[0]))
plt.plot(np.arange(51), RMSE_3[0:51])
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()


# In[24]:


plt.plot(np.arange(1,100), RMSE[1:100], label='p=1')
plt.plot(np.arange(1,100), RMSE_2[1:100], label='p=2')
plt.plot(np.arange(1,100), RMSE_3[1:100],label='p=3') 
plt.legend() 
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()


# In[ ]:





# In[ ]:




