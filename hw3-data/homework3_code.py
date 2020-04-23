#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA
from scipy.stats import multivariate_normal


# In[27]:


#implement Kmeans
def Kmeans(centroids_list, K ,data):
    cost_list = []
    for rounds in range(0,20):
        #initializations
        new_clusters_list=[]
        k_cluster=[]
        total_cost = 0
        cluster_centroid = []
        #find the closest centroid for each xi
        for i in range(0,500):
            distance_list = []
            for centroid in centroids_list:
                distance = np.sum((data[i]-centroid)**2)
                distance_list.append(distance)
                nearest_cluster = min(distance_list)
                new_cluster = distance_list.index(nearest_cluster)
            new_clusters_list.append(new_cluster)
        #new K clusters
        
        for index in range(0,K):
            cluster = data[np.array(new_clusters_list)==index,:]
            k_cluster.append(cluster)
            
        #new Centroids
        for item in range(0,K):
            cluster_mean = np.average(k_cluster[item],axis=0)
            cluster_centroid.append(cluster_mean)
            
            cost = np.sum((k_cluster[item]-centroids_list[item])**2)
            total_cost = total_cost + cost
            

        cost_list.append(total_cost)
        centroids_list = cluster_centroid
    return cost_list, k_cluster
        


# In[28]:


#generate 500 observations with mixing weights
start = np.random.randint(0,501,size=5)
data1 = np.random.multivariate_normal(np.array([0,0]),np.matrix([[1,0],[0,1]]),500)
data2 = np.random.multivariate_normal(np.array([3,0]),np.matrix([[1,0],[0,1]]),500)
data3 = np.random.multivariate_normal(np.array([0,3]),np.matrix([[1,0],[0,1]]),500)
data_pi = np.random.choice(range(3),500,p=[0.2,0.5,0.3])
dataset = np.concatenate((data1[data_pi==0,:],
                       data2[data_pi==1,:],
                       data3[data_pi==2,:] ))


# In[29]:



K_2 = (dataset[start[0]],dataset[start[1]])
K_3 = (dataset[start[0]],dataset[start[1]],dataset[start[2]])
K_4 = (dataset[start[0]],dataset[start[1]],dataset[start[2]],dataset[start[3]])
K_5 = (dataset[start[0]],dataset[start[1]],dataset[start[2]],dataset[start[3]],dataset[start[4]])


# In[30]:


cost_2, Kcluster_2 = Kmeans(K_2,2,dataset)
cost_3, Kcluster_3 = Kmeans(K_3,3,dataset)
cost_4, Kcluster_4 = Kmeans(K_4,4,dataset)
cost_5, Kcluster_5 = Kmeans(K_5,5,dataset)
plt.figure(figsize=(6,6))
plt.xticks(np.arange(1,21))
plt.xlabel('i_th iteration')
plt.ylabel('cost')
plt.plot(np.arange(1,21),cost_2,label='K=2')
plt.plot(np.arange(1,21),cost_3,label='K=3')
plt.plot(np.arange(1,21),cost_4,label='K=3')
plt.plot(np.arange(1,21),cost_5,label='K=3')
plt.legend()


# In[31]:


cluster1=pd.DataFrame(Kcluster_3[0])
cluster2=pd.DataFrame(Kcluster_3[1])
cluster3=pd.DataFrame(Kcluster_3[2])
x1,y1 = cluster1[0].tolist(),cluster1[1].tolist()
x2,y2 = cluster2[0].tolist(),cluster2[1].tolist()
x3,y3 = cluster3[0].tolist(),cluster3[1].tolist()
plt.scatter(x1,y1,label='cluster1')
plt.scatter(x2,y2,label='cluster2')
plt.scatter(x3,y3,label='cluster3')
plt.legend()


# In[32]:


cluster1=pd.DataFrame(Kcluster_5[0])
cluster2=pd.DataFrame(Kcluster_5[1])
cluster3=pd.DataFrame(Kcluster_5[2])
cluster4=pd.DataFrame(Kcluster_5[3])
cluster5=pd.DataFrame(Kcluster_5[4])
x1,y1 = cluster1[0].tolist(),cluster1[1].tolist()
x2,y2 = cluster2[0].tolist(),cluster2[1].tolist()
x3,y3 = cluster3[0].tolist(),cluster3[1].tolist()
x4,y4 = cluster4[0].tolist(),cluster4[1].tolist()
x5,y5 = cluster5[0].tolist(),cluster5[1].tolist()
plt.scatter(x1,y1,label='cluster1')
plt.scatter(x2,y2,label='cluster2')
plt.scatter(x3,y3,label='cluster3')
plt.scatter(x4,y4,label='cluster4')
plt.scatter(x5,y5,label='cluster5')
plt.legend()


# In[33]:


#Q2
X_train = pd.read_csv('Prob2_Xtrain.csv', header=None)
X_test = pd.read_csv('Prob2_Xtest.csv', header=None)
y_train = pd.read_csv('Prob2_ytrain.csv', header=None)
y_test = pd.read_csv('Prob2_ytest.csv', header=None)
#combine train data and sperate spam and non-spam
total_train = pd.concat([X_train,y_train],axis=1)
train_1 = total_train[total_train.iloc[:,-1]==1]
train_0 = total_train[total_train.iloc[:,-1]==0]
#split the X and y
X_train_1 = train_1.iloc[:,0:10]
X_train_0 = train_0.iloc[:,0:10]
y_train_1 = train_1.iloc[:,-1]
y_train_0 = train_0.iloc[:,-1]


# In[34]:


def EM(K,data):
    
    #initialization
    empirical_cov = np.array(K*[np.cov(data.transpose())])
    nk_shape = data.shape[0]
    mu = np.random.multivariate_normal(np.array(data.describe().loc['mean',:]),empirical_cov[0],K)
    pi = np.ones(K)*np.array(1/K)
   
    
    
    #E Step
    
    objective_function = []
    for j in range(0,30):
        phi = [0]*K
        nk = np.zeros(K)
        pin = 0
        for j in range(0, K):
            pin += pi[j]*multivariate_normal.pdf(data,mu[j],empirical_cov[j],allow_singular=True)
        for j in range(0, K): 
            phi[j] = pi[j]*multivariate_normal.pdf(data,mu[j],empirical_cov[j], allow_singular=True)/pin
        
        #M Step
        for j in range(0,K):
            nk[j] = np.sum(phi[j])
            pi[j] = nk[j]/nk_shape
        
         #update
        for j in range(0,K):
            mu[j] = (1/nk[j])*np.matmul(np.matrix(phi[j].reshape(1,-1)), np.matrix(data))
        empirical_cov[j] = np.matmul(np.multiply(phi[j].reshape(-1,1),(np.array(data)-mu[j])).transpose(),(np.array(data)-mu[j]))/nk[j]

        objective_function.append(np.sum(np.log(pin)))
    
    return objective_function, pi, mu, empirical_cov


# In[35]:


def plottingGMM(data, K):
    plt.figure(figsize=(8,6))
    rounds=0
    x_axis = np.arange(5,31)
    obj_k = []
    pi_k = []
    mu_k = []
    cov_k = []
    for i in range(0, 10):
        objective_function, pi, mu, empirical_cov = EM(K, data)
        obj_k.append(objective_function)
        pi_k.append(pi)
        mu_k.append(mu)
        cov_k.append(empirical_cov)

    for i in obj_k:
        i = i[4:]
        plt.plot(x_axis, i, label="round"+str(rounds))
        rounds+=1
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('objective function')
    return obj_k, pi_k, mu_k,cov_k
    


# In[36]:


#class 1
obj_1, pi_1, mu_1,cov_1 = plottingGMM(X_train_1, 3)
plt.title("class1")


# In[37]:


#class 0
obj_0, pi_0, mu_0,cov_0 = plottingGMM(X_train_0, 3)
plt.title("class0")


# In[38]:


#b
def best_objective_index(obj_k):
    obj_lst = []
    n = 0
    largest=0
    for i in obj_k:
        obj_lst.append(i[-1])
        largest= max(obj_lst)
        index_lst = [n for n, m in enumerate(obj_lst) if m == largest]
        best_index = index_lst[0]
    return best_index     
def GMM_val(data, K):
    obj_k = []
    pi_k = []
    mu_k = []
    cov_k = []
    for i in range(0, 10):
        objective_function, pi, mu, empirical_cov = EM(K, data)
        obj_k.append(objective_function)
        pi_k.append(pi)
        mu_k.append(mu)
        cov_k.append(empirical_cov)
    return obj_k, pi_k, mu_k,cov_k
#best values
obj_0, pi_0, mu_0,cov_0 = GMM_val(X_train_0, 3)
obj_1, pi_1, mu_1,cov_1 = GMM_val(X_train_1, 3)
index_0 = best_objective_index(obj_0) 
index_1 = best_objective_index(obj_1)
best_pi_0 = pi_0[index_0]
best_mu_0 = mu_0[index_0]
best_cov_0 = cov_0[index_0]
best_pi_1 = pi_1[index_1]
best_mu_1 = mu_1[index_1]
best_cov_1 = cov_1[index_1]


# In[39]:


def Bayes_GMM(data, K, best_pi_1, best_mu_1, best_cov_1, best_pi_0, best_mu_0, best_cov_0, compare_list):
    class0 = 0
    class1 = 0
    prediction = []
    TruePositive = 0
    FalsePositive = 0
    TrueNegative = 0
    FalseNegative = 0
    
    for i in range(0,K):
        class0 = best_pi_0[i]*multivariate_normal.pdf(data,best_mu_0[i],best_cov_0[i],allow_singular=True)+class0
    class0 = class0.tolist()
    for i in range(0,K):
        class1 += best_pi_1[i]*multivariate_normal.pdf(data,best_mu_1[i],best_cov_1[i],allow_singular=True)+class1
    class1 = class1.tolist() 

    for i in range(0, len(class1)):
        if class1[i]>=class0[i]:
            prediction.append(1)
        else:
            prediction.append(0)

    for i in range(0,len(compare_list)):
        if prediction[i]==1 and compare_list[i]==1:
            TruePositive += 1
        elif prediction[i]==0 and compare_list[i]==1:
            FalseNegative += 1
        elif prediction[i]==1 and compare_list[i]==0:
            FalsePositive += 1
        elif prediction[i]==0 and compare_list[i]==0:
            TrueNegative += 1
            
    matrix = [('FalseNegative:'+str(FalseNegative), 'TrueNegative:'+str(TrueNegative)),('TruePostive:'+str(TruePositive), 'FalseNegative:'+str(FalsePositive))]
    df = pd.DataFrame(matrix)
    df = df.rename({0: 'Truelist +', 1: 'Truelist -'}, axis='columns')
    df = df.rename({0: 'Predictitonlist -', 1: 'Predictionlist +'}, axis='index')
    accuracy = (TruePositive+TrueNegative)/(TruePositive+TrueNegative+FalsePositive+FalseNegative)
    display(df)
    print('The Accuracy is:'+ str(accuracy))


# In[40]:


#guassian 3
Bayes_GMM(X_test, 3, best_pi_1, best_mu_1, best_cov_1, best_pi_0, best_mu_0, best_cov_0,y_test.iloc[:,0].tolist())


# In[41]:


#gaussian 1
obj_0, pi_0, mu_0,cov_0 = GMM_val(X_train_0, 1)
obj_1, pi_1, mu_1,cov_1 = GMM_val(X_train_1, 1)
index_0 = best_objective_index(obj_0) 
index_1 = best_objective_index(obj_1)
best_pi_0 = pi_0[index_0]
best_mu_0 = mu_0[index_0]
best_cov_0 = cov_0[index_0]
best_pi_1 = pi_1[index_1]
best_mu_1 = mu_1[index_1]
best_cov_1 = cov_1[index_1]
Bayes_GMM(X_test, 1, best_pi_1, best_mu_1, best_cov_1, best_pi_0, best_mu_0, best_cov_0,y_test.iloc[:,0].tolist())


# In[42]:


#guassian 2
obj_0, pi_0, mu_0,cov_0 = GMM_val(X_train_0, 2)
obj_1, pi_1, mu_1,cov_1 = GMM_val(X_train_1, 2)
index_0 = best_objective_index(obj_0) 
index_1 = best_objective_index(obj_1)
best_pi_0 = pi_0[index_0]
best_mu_0 = mu_0[index_0]
best_cov_0 = cov_0[index_0]
best_pi_1 = pi_1[index_1]
best_mu_1 = mu_1[index_1]
best_cov_1 = cov_1[index_1]
Bayes_GMM(X_test, 2, best_pi_1, best_mu_1, best_cov_1, best_pi_0, best_mu_0, best_cov_0,y_test.iloc[:,0].tolist())


# In[43]:


#guassian 4
obj_0, pi_0, mu_0,cov_0 = GMM_val(X_train_0, 4)
obj_1, pi_1, mu_1,cov_1 = GMM_val(X_train_1, 4)
index_0 = best_objective_index(obj_0) 
index_1 = best_objective_index(obj_1)
best_pi_0 = pi_0[index_0]
best_mu_0 = mu_0[index_0]
best_cov_0 = cov_0[index_0]
best_pi_1 = pi_1[index_1]
best_mu_1 = mu_1[index_1]
best_cov_1 = cov_1[index_1]
Bayes_GMM(X_test, 4, best_pi_1, best_mu_1, best_cov_1, best_pi_0, best_mu_0, best_cov_0,y_test.iloc[:,0].tolist())


# In[53]:


def matrix_factorization(dataset, sigma_squared, rank, Lambda, rounds, total_user, total_movie):

    obj_function = []
    u = np.repeat(np.NaN,total_user*10).reshape(total_user, 10) 
    v = np.random.multivariate_normal([0]*total_movie, np.identity(total_movie), size = 10) 
    
    for iteration in range(0,rounds):
        # update u
        for i in range(0,total_user):
            vu_i = abs(np.sign(dataset[i])) 
            v_j = np.multiply(vu_i,v) 
            optimal_u = np.matmul(v_j,v_j.transpose()) 
            term1 = inv(np.identity(rank)*sigma_squared + optimal_u) 

            term2 = np.matmul(dataset[i].reshape(-1,1).transpose(),v.transpose()) 
            solved_u = np.matmul(term2, term1) 
            u[i] = solved_u 
        #update V
        for j in range(0,1682):
            vu_j = abs(np.sign(dataset[:,j])) 
            u_i = np.multiply(vu_j, u.transpose()) 
            optimal_v = np.matmul(u_i,u_i.transpose()) 
            temp1 = inv(np.identity(rank)*sigma_squared + optimal_v) 
            
            temp2 = np.matmul(dataset[:,j].reshape(-1,1).transpose(),u) 
            solved_v = np.matmul(temp2,temp1)         
            v[:,j] = solved_v
        #prior u
        prior_u = 0
        for i in range(0, u.shape[0]):
            loss_u = u[i]
            prior_u+=Lambda/2*np.sum(np.square(loss_u))
        #prior v
        prior_v = 0
        for j in range(0, v.shape[0]):
            loss_v = v[j]
            prior_v+=Lambda/2*np.sum(np.square(loss_v))
        #update first term   
        vu_ij = abs(np.sign(dataset))
        prediction = np.matmul(u, v)
        square_error = np.multiply(dataset - prediction,vu_ij)
        likelihood = np.square(LA.norm(square_error))/(2*sigma_squared)
        
        #loss function
        L = -likelihood-prior_u-prior_v
        obj_function.append(L)
        
    return obj_function, u, v 


# In[71]:


#number of users unique
train_ratings = pd.read_csv('Prob3_ratings.csv', header=None) 
train_ratings.columns = ['user_id','item_id','rating']
total_user=len(train_ratings.loc[:,'user_id'].unique())


# In[72]:


#number of movies unique
movies_list = pd.read_csv('Prob3_movies.txt',sep="\n", header=None)
total_movie=movies_list.shape[0]


# In[73]:


M_ij = np.zeros((total_user, total_movie))
for i in range(0, train_ratings.shape[0]):
    item = train_ratings.iloc[i,:]
    M_ij[int(item[0]-1),int(item[1]-1)] = item[2]


# In[78]:


def MF_run(dataset, runs):
    
    obj_list = []
    u_list = []
    v_list = []
    for i in range(0, runs):
        objectives, u, v = matrix_factorization(dataset,0.25,10, 1,100,total_user,total_movie)
        obj_list.append(objectives)
        u_list.append(u)
        v_list.append(v)
    return obj_list, u_list, v_list
obj_list, u_list, v_list = matrix_fac_ntimes(M_ij, 10)


# In[85]:


def RMSE(prediction, actual):
    """
    Return RMSE of prediction result that obtained by matrix factorization and true data
    
    @pred: obtained by uT*v
    @true: true matrix
    """
    #subtract = np.multiply(actual - prediction, abs(np.sign(actual)))
    #N = np.sum(abs(np.sign(actual)))
    #MSE = np.sum(np.square(np.multiply(actual - prediction, abs(np.sign(actual)))))/(np.sum(abs(np.sign(actual))))
    RMSE = np.sqrt(np.sum(np.square(np.multiply(actual - prediction, abs(np.sign(actual)))))/(np.sum(abs(np.sign(actual)))))
    return RMSE


# In[146]:


iteration = 1
plt.figure(figsize=(10,8))
for i in obj_list:
    plt.plot(np.arange(2,101), i[1:], label = 'runs'+str(iteration))
    plt.legend()
    plt.xlabel('iterations')
    plt.title('objective function')
    iteration+=1


# In[84]:


test_rating = pd.read_csv('Prob3_ratings_test.csv', header=None)
test_rating.columns = ['user_id','item_id','rating']
M_ij_test = np.zeros((total_user, total_movie))
for i in range(0, test_rating.shape[0]):
    item = test_rating.iloc[i,:]
    M_ij_test[int(item[0]-1),int(item[1]-1)] = item[2]


# In[88]:


#plot the table
final_value= []
for i in range(0, len(obj_list)):
    final_value.append(obj_list[i][-1])
rmse_list = []
for i in range(0, len(u_list)):
    u = u_list[i]
    v = v_list[i]
    pred_list = np.matmul(u, v)
    rmse = RMSE(pred_list, M_ij_test)
    rmse_list.append(rmse)
table = {'run': np.arange(1,11),'RMSE':rmse_list,'final_value':final_value}
df=pd.DataFrame(table)
df.sort_values(by=['final_value'],ascending=False)


# In[109]:


#b
def ten_closest_movies(id_movie, v_transpose):
    target_movie = v_transpose[id_movie-1]
    
    distances = []
    for movies in v_transpose: #best_v_T.shape[0]=1682
        euclidean_distance  = np.sqrt(np.sum(np.square(target_movie - movies)))
        distances.append(euclidean_distance)
    table = {'movie_id':np.arange(1,1683).tolist(),'distance': distances}
    return table


# In[136]:


#for star war(movie id-50)
ten_closest = pd.DataFrame(ten_closest_movies(50,v_list[0].transpose())).sort_values(by=['distance']).head(11)
ten_closest = ten_closest.drop(['movie_id'], axis=1)
starwar_10 = pd.concat([movies_list,ten_closest], axis=1, join='inner')
starwar_10 = starwar_10.sort_values(by=['distance'])
starwar_10.columns=['movie name','distance']
starwar_10.set_index('movie name')


# In[143]:


#for my fair lady(movie id-485)
ten_closest_2 = pd.DataFrame(ten_closest_movies(485,v_list[0].transpose())).sort_values(by=['distance']).head(11)
ten_closest_2 = ten_closest_2.drop(['movie_id'], axis=1)
My_Fair_Lady_10 = pd.concat([movies_list,ten_closest_2], axis=1, join='inner')
My_Fair_Lady_10 = My_Fair_Lady_10.sort_values(by=['distance'])
My_Fair_Lady_10.columns=['movie name','distance']
My_Fair_Lady_10.set_index('movie name')


# In[144]:


#for GoodFellas(movie id- 182)
ten_closest_3 = pd.DataFrame(ten_closest_movies(182,v_list[0].transpose())).sort_values(by=['distance']).head(11)
ten_closest_3 = ten_closest_3.drop(['movie_id'], axis=1)
GoodFellas_10 = pd.concat([movies_list,ten_closest_3], axis=1, join='inner')
GoodFellas_10 = GoodFellas_10.sort_values(by=['distance'])
GoodFellas_10.columns=['movie name','distance']
GoodFellas_10.set_index('movie name')


# In[ ]:




