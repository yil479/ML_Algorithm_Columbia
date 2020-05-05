#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#1(a)
scores = pd.read_csv("CFB2019_scores.csv", header = None)
scores.columns = ['Team_A_index','Team_A_points',
                'Team_B_index', 'Team_B_points']
scores.head(10)


# In[4]:


scores['Team_A_index'].nunique()


# In[5]:


#construct random walk matrix
RW_M = np.zeros((scores['Team_A_index'].nunique(), scores['Team_A_index'].nunique()))
#team rankings
for i in range(0, scores.shape[0]):
    
    j1=scores.iloc[i,:].Team_A_index-1
    j2=scores.iloc[i,:].Team_B_index-1
    points_j1=scores.iloc[i,:].Team_A_points
    points_j2=scores.iloc[i,:].Team_B_points
    #teamA WIN
    if points_j1 > points_j2:
        RW_M[int(j1), int(j1)] += 1+points_j1/(points_j1+points_j2)
        RW_M[int(j2), int(j2)] += points_j2/(points_j1+points_j2)
        RW_M[int(j1), int(j2)] += points_j2/(points_j1+points_j2)
        RW_M[int(j2), int(j1)] += 1+points_j1/(points_j1+points_j2)
    #teamB WIN
    elif points_j1 < points_j2:
        RW_M[int(j1), int(j1)] += points_j1/(points_j1+points_j2)
        RW_M[int(j2), int(j2)] += 1+points_j2/(points_j1+points_j2)
        RW_M[int(j1), int(j2)] += 1+points_j2/(points_j1+points_j2)
        RW_M[int(j2), int(j1)] += points_j1/(points_j1+points_j2)


# In[6]:


RW_M


# In[21]:


#team name list
team_name = pd.read_csv('TeamNames.txt',sep="\n", header=None)


# In[49]:


def rank_top25_team(t):
    M = RW_M/np.sum(RW_M, axis=1).reshape(-1,1)
    w0 = np.random.uniform(size=(1,769))
    w0 = w0/np.sum(w0, axis=1)
    W_t = np.matmul(w0,M)
    for i in range(t-1):
        W_t = np.matmul(W_t,M)
        
    df = pd.DataFrame({'wt':W_t.tolist()[0]}).sort_values(by=['wt'],ascending=False)
    top25 = pd.concat([df, team_name], axis=1, join='inner').head(25)
    top25.columns=['wt','team_name']
    return top25


# In[50]:


t_10 = rank_top25_team(10)
t_100 = rank_top25_team(100)
t_1000 = rank_top25_team(1000)
t_10000 = rank_top25_team(10000)


# In[51]:


multi_table([t_10,t_100,t_1000,t_10000])


# In[48]:


#1(b)
from scipy.sparse.linalg import eigs
M_transpose = (RW_M/np.sum(RW_M, axis=1).reshape(-1,1)).transpose()
#u1 is the first eigenvecot of M transpose
lambda1, u1 = eigs(M_transpose,k=1)

#w_infinite
W_inf = u1.flatten()/np.sum(u1.flatten())
#normalized random walk matrix
M = RW_M/np.sum(RW_M, axis=1).reshape(-1,1)
#w_0
w0 = np.random.uniform(size=(1,769))
w0 = w0/np.sum(w0, axis=1)
#w_1
W_t = np.matmul(w0,M)

diff = []
diff.append(np.sum(abs(W_t - W_inf))) 
for i in range(10000-1):
    #w_t
    W_t = np.matmul(W_t,M) 
    diff.append(np.sum(abs(W_t - W_inf)))  
    
#plot    
plt.figure(figsize=(8,8))
plt.plot(np.arange(1,10001), diff)
plt.xlabel('t')
plt.ylabel('difference')


# In[121]:


#2(a)
document = pd.read_csv('nyt_data.txt',sep='\n',header=None)
document.head(10)


# In[101]:


X = np.zeros([3012,8447])


# In[144]:


j = 0
for k in range(0, document.shape[0]):
    each_document = document.iloc[k,:].tolist()[0].split(',')
    for words in each_document:

        i = int(words.split(':')[0])-1
        X[i, j] = int(words.split(':')[1])
    j =j+1


# In[145]:


X


# In[157]:


#NMF implementation
warning_space = 1e-16
objective_function = []
W= np.random.uniform(1,2,(3012,25))
H = np.random.uniform(1,2,(25,8447))
for i in range(0,100):
    #data matrix dot devided by its approximation
    purple = np.divide(X,np.matmul(W,H)+warning_space) 
    #normalized the transpose
    pink = W.transpose()/np.sum(W.transpose(), axis=1).reshape(-1,1)
    #update H
    H = np.multiply(H, np.matmul(pink, purple))
    
    
    purple = np.divide(X,np.matmul(W,H)+warning_space) 
    #normalized the transpose
    light_blue = H.transpose()/np.sum(H.transpose(), axis=0)
    #update W
    W = np.multiply(W, np.matmul(purple, light_blue))


    objective = np.sum(np.multiply(np.log(1/(np.matmul(W,H)+warning_space)),X) + np.matmul(W,H))
    objective_function.append(objective)


# In[160]:


plt.figure(figsize=(8,8))
plt.plot(np.arange(0,100), objective_function)
plt.xlabel('iteration')
plt.ylabel('objective function')


# In[224]:


#start 2b

vocab = pd.read_csv('nyt_vocab.dat', header=None)
vocab = pd.DataFrame({'vocab':vocab.iloc[:,0].tolist()})
vocab.head(10)


# In[226]:


W_norm=W/np.sum(W, axis=0)
weight_column = pd.DataFrame(W_norm)
final_table = []
for i in range(weight_column.shape[1]):
    df =pd.DataFrame({'weight':weight_column.iloc[:,i].tolist()})
    df =df.sort_values(by='weight', ascending=False).iloc[0:10,:]
    topic = pd.concat([df, vocab], axis=1, join='inner')
    final_table.append(topic)  


# In[27]:


#display tables
#reference: https://github.com/epmoyer/ipy_table/issues/24
from IPython.core.display import HTML

def multi_table(table_list):
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )


# In[253]:


table1= final_table[0]
table2=final_table[1]
table3 = final_table[2]
table4 = final_table[3]
table5 = final_table[4]
table6 = final_table[5]
table7 = final_table[6]
table8 = final_table[7]
table9 = final_table[8]
table10 = final_table[9]
table11 = final_table[10]
table12 = final_table[11]
table13 = final_table[12]
table14 = final_table[13]
table15 = final_table[14]
table16 = final_table[15]
table17 = final_table[16]
table18 = final_table[17]
table19 = final_table[18]
table20 = final_table[19]
table21= final_table[20]
table22=final_table[21]
table23 = final_table[22]
table24 = final_table[23]
table25 = final_table[24]


# In[251]:


multi_table([table1,table2,table3,table4,table5])


# In[252]:


multi_table([table6,table7,table8,table9,table10])


# In[254]:


multi_table([table11,table12,table13,table14,table15])


# In[255]:


multi_table([table16,table17,table18,table19,table20])


# In[257]:


multi_table([table21,table22,table23,table24,table25])


# In[ ]:




