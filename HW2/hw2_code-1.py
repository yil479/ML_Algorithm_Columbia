#!/usr/bin/env python
# coding: utf-8

# In[263]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from sklearn.model_selection import KFold
import math


# In[264]:


X = pd.read_csv('X.csv', header = None)
y = pd.read_csv('y.csv', header = None)
kf = KFold(n_splits =10, random_state=42)


# In[265]:


from scipy.stats import poisson
true_positive=0
true_negative=0
false_positive=0
false_negative=0

for train_index, test_index in kf.split(X):
    sum_0 = 0
    sum_1 = 0
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
    #initiate prediction array for test
    prediction = np.zeros(y_test.shape[0])
    #spam rate pi
    pi = sum(y_train[0])/X_train.shape[0]
   
    for i in range(X_train.shape[0]):
        if y_train.iloc[i,0] == 0:
            sum_0 += X_train.iloc[i,:]
        else:
            sum_1 += X_train.iloc[i,:]
    #lambda esimator of dimension for spam
    lambda_1 = (sum_1+1)/(1+sum(y_train[0]))
    #lambda estimator of dimension for normal
    lambda_0 = (sum_0+1)/(1+X_train.shape[0]-sum(y_train[0]))
    for m in range(y_test.shape[0]):
        #bernoulli trial
        p_0 = 1-pi
        p_1 = pi
        for n in range(X_test.shape[1]):
            #naive bayes estimation for spam
            #p_1=p_1*(lambda_1[n]**X_test.iloc[m,n]*(math.exp(-lambda_1[n])))
            p_1=p_1*(poisson.pmf(X_test.iloc[m,n],lambda_1[n]))
            #naive bayes estimation for normal
            #p_0=p_0*(lambda_0[n]**X_test.iloc[m,n]*(math.exp(-lambda_0[n])))
            p_0=p_0*(poisson.pmf(X_test.iloc[m,n],lambda_0[n]))
        prediction[m] += (p_1> p_0)
        
        if y_test.iloc[m,0] == 1 and prediction[m] ==True:
            true_positive =true_positive+1
        elif y_test.iloc[m,0] == 0 and prediction[m] ==False: 
            true_negative =true_negative+1
        elif y_test.iloc[m,0] == 0 and prediction[m] ==True: 
            false_positive =false_positive+1
        else:
            false_negative =false_negative+1

accuracy = (true_positive + true_negative)/4600            


# In[266]:


print("The accuracy is", accuracy, "The true positive is", true_positive,"The true negative is", true_negative,"The false positive is", false_positive,"The false negative is", false_negative) 


# In[267]:


estimator_0 = 0
estimator_1 = 0
estimator_0 += lambda_0
estimator_1 += lambda_1
plt.figure(figsize=(8, 5))
plt.stem(np.arange(54), estimator_1, label='y=1',markerfmt='red', linefmt='red')
plt.stem(np.arange(54), estimator_0, label='y=0',markerfmt='blue',linefmt='blue' )
plt.xlabel('dimensions')
plt.ylabel('lambda value')
plt.legend()
plt.show()


# In[268]:


accuracy_list =0
for train_index, test_index in kf.split(X):
    accuracy = 0
    total_error = 0
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
    prediction = np.zeros((y_test.shape[0],20))
    
    X_train=X_train.values
    X_test=X_test.values

    
    ytrain_arry=np.asarray(y_train)
    #print(ytrain_arry)
    ytest_arry= np.asarray(y_test)
    #print(ytest_arry)
    y_train = np.squeeze(ytrain_arry,axis=None)
    y_test = np.squeeze(ytest_arry,axis=None)
    y_test_2 = y_test.reshape(-1,1)
    y_test_3 = y_test.shape[0]
    #numpy.argpartition
    for i in range(X_test.shape[0]):
        #distance
        distance = np.sum(np.absolute(X_train-X_test[i,:]),axis=1)
        
        for j in range(1,21):
            #get indices of nearest feature vectors and return their classes
            k_nearest= y_train[np.argpartition(distance, j-1,axis=-1,order=None)[:j]]
            #count occurence for the classes
            prediction[i,j-1] = np.argmax(np.bincount(k_nearest,weights=None, minlength=2))
    
    error =prediction - y_test_2
    accuracy = 1-np.sum(np.absolute(error),axis = 0)/y_test_3
    accuracy_list += np.array(accuracy)


# In[269]:


distance


# In[270]:


accuracy_list =accuracy_list/10
accuracy_list


# In[271]:


plt.plot(np.arange(1,21), accuracy_list)
plt.xlabel('# of k')
plt.xticks(range(1,21))
plt.ylabel('accuracy')


# In[272]:


X_train = np.array(pd.read_csv('X_train.csv',header=None))
X_test = np.array(pd.read_csv('X_test.csv',header=None))
y_train = np.array(pd.read_csv('y_train.csv',header=None))
y_test = np.array(pd.read_csv('y_test.csv',header=None))


# In[ ]:


#Q1
#Guassian process when b=1 and sigma =0
b = 5 
sigma2 = 2
N=sigma2*np.identity(len(X_train))
K_n = np.zeros((len(X_train),len(X_train)))
miu = np.zeros((1,len(X_train)))
prediction = np.zeros(len(y_test))
for i in range(len(X_train)):
    for j in range(len(X_train)):
        #Kenel update
        K_n[i,j] = np.exp((-1/b)*np.linalg.norm(X_train[i] - X_train[j]))
        


for i in range(len(X_test)):
    for j in range(len(X_train)):
        #kernel update
        miu[0,j] = np.exp((-1/b)*np.linalg.norm(X_train[i] - X_train[j]))
    prediction[i] = multi_dot([miu, np.linalg.inv( K_n+N),y_train])
prediction  


# In[ ]:


#part 4
forth_d = X_train[:,3]
S=sigma2*np.identity(len(forth_d))
#set up predictin velue
y_pred = np.zeros(len(forth_d))
K = np.zeros((1,len(forth_d)))
K_n = np.zeros((len(forth_d),len(forth_d)))
for i in range(len(forth_d)):
    for j in range(len(forth_d)):
        #Kernel update
        K_n[i,j]=np.exp((-1/b)*np.linalg.norm(forth_d[i] - forth_d[j]))
        

for i in range(len(forth_d)):
    for j in range(len(forth_d)):
        #kenel update        
        K[0,j] = np.exp((-1/b)*np.linalg.norm(forth_d[i] - forth_d[j]))
        
    y_pred[i] = multi_dot([K, np.linalg.inv(K_n+S),y_train])
#plot the graph
plt.scatter(y_train, forth_d)
plt.plot(y_pred, forth_d, c='black')
plt.show()


# In[ ]:




