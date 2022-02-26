#!/usr/bin/env python
# coding: utf-8

# * Analyzing a complex set of variables into its principal components
# * Reduce number of dimensionsin data. Then train ML algorithm on smaller data set.
# * Show which features explain the most variance in the data.
# * Principal component is a linear combination of original features. More variance more influence
# * EigenVector: Directional information
# 
#     **PCA Steps**
# 1. Get original data
# 2. Calculate Covariance Matrix
# 3. Calculate EigenVectors
# 4. Sort EigenVectors by EigenValues
# 5. Choose N Largest EigenValues
# 6. Project original data onto EigenVectors

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('cancer_tumor_data_features.csv')


# In[3]:


df.head()


# In[5]:


sns.heatmap(df)


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


scaled_X = scaler.fit_transform(df)


# In[9]:


scaled_X.mean()


# In[10]:


covariance_matrix = np.cov(scaled_X,rowvar=False)


# In[11]:


eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)


# In[13]:


eigen_vectors.shape


# In[15]:


eigen_values.shape


# In[16]:


# PCA N Feature Space ---> N PC Space ---> 2
# How many PC to choose? N=2


# In[17]:


num_components = 2


# In[23]:


sorted_key = np.argsort(eigen_values)[::-1][:num_components]


# In[24]:


eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:,sorted_key]


# In[25]:


principal_components = np.dot(scaled_X, eigen_vectors)

# Original Data ---> project ---> eigen_vectors


# In[26]:


principal_components


# In[28]:


plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1])


# In[29]:


from sklearn.datasets import load_breast_cancer


# In[30]:


cancer_dict = load_breast_cancer()


# In[31]:


type(cancer_dict)


# In[33]:


cancer_dict.keys()


# In[35]:


print(cancer_dict['DESCR'])


# In[36]:


cancer_dict['target']


# In[37]:


plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1],c=cancer_dict['target'])


# ## PCA

# In[38]:


from sklearn.decomposition import PCA


# In[39]:


model = PCA(n_components=2)


# In[41]:


pc_results = model.fit_transform(scaled_X)


# In[44]:


plt.figure(figsize=(8,6))
plt.scatter(pc_results[:,0],pc_results[:,1],c=cancer_dict['target'])


# In[46]:


model.components_.shape


# In[47]:


df_comp = pd.DataFrame(model.components_,index=['PC1','PC2'],columns=df.columns)


# In[48]:


df_comp


# In[54]:


plt.figure(figsize=(20,5))
sns.heatmap(df_comp,annot=True)


# In[55]:


model.explained_variance_ratio_


# In[56]:


np.sum(model.explained_variance_ratio_)


# In[57]:


pca_30 = PCA(n_components=30)


# In[58]:


pca_30.fit(scaled_X)


# In[59]:


pca_30.explained_variance_ratio_


# In[60]:


np.sum(pca_30.explained_variance_ratio_)


# In[61]:


explined_variance =[]

for n in range(1,30):
    pca = PCA(n_components=n)
    pca.fit(scaled_X)
    
    explined_variance.append(np.sum(pca.explained_variance_ratio_))


# In[64]:


plt.plot(range(1,30),explined_variance)
plt.xlabel('Number of Features')
plt.ylabel('Variance')


# In[ ]:




