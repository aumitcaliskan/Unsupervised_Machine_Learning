#!/usr/bin/env python
# coding: utf-8

# * Very common in biology
# * Similarity by choosing a distance metric
# * **Agglomerative Approach:** Each point begins as its own cluster, then clusters are joined.
# * **Divisive Approach:** All points begin in the same cluster, then clusters are split.
# * 
# **Pros**
# 
# * Easy to understand
# * Helps users decide how many clusters to choose
# * Not necessary to choose cluster amount before running the algorithm

# ## Similarity Metric
# 
# * Measures distance between two points. 
# * **affinity** default= euclidean.   
#     Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed".
# * Need to use scaler.  

# ## Dendrogram
# 
# * Plot displaying all potential clusters
# * Very computationally expensive to compute and display for larger data sets.
# * **Very useful for deciding on number of clusters**
# 
# ![dendrogram](dendrogram.png)
# 

# ## Linkage Matrix
# 
# Which linkage criterion to use. The linkage criterion determines which
#     distance to use between sets of observation. The algorithm will merge
#     the pairs of cluster that minimize this criterion.  
#     
# **linkage :** {'ward', 'complete', 'average', 'single'}, default='ward'
# 
#    **'ward'** minimizes the variance of the clusters being merged.    
#    **'average'** uses the average of the distances of each observation of the two sets.    
#    **'complete' or 'maximum'** linkage uses the maximum distances between all observations of the two sets.    
#    **'single'** uses the minimum of the distances between all observations of the two sets.   

# ## Attributes
# 
# **No predict** Algorithm needs to cluster again when predicting a value 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('cluster_mpg.csv')


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df['origin'].value_counts()


# In[10]:


df_dummies = pd.get_dummies(df.drop('name', axis=1))


# In[11]:


df_dummies


# In[12]:


from sklearn.preprocessing import MinMaxScaler

# MinMax is quite well for dendrogram


# In[13]:


scaler = MinMaxScaler()


# In[14]:


scaled_data = scaler.fit_transform(df_dummies)


# In[15]:


scaled_data


# In[17]:


scaled_df = pd.DataFrame(scaled_data, columns=df_dummies.columns)


# In[18]:


scaled_df


# In[19]:


plt.figure(figsize=(15,8))
sns.heatmap(scaled_df, cmap='magma')


# In[24]:


plt.figure(figsize=(15,8))
sns.clustermap(scaled_df, row_cluster=False)

# origin_japan, origin_europe, origin_usa very far from each other. No similarity
# Displacement and weight are very close to each other. Very high similarity
# This relationship is between columns.


# In[23]:


scaled_df.corr()


# In[26]:


# We need to cluster cars not features.

plt.figure(figsize=(15,8))
sns.clustermap(scaled_df, col_cluster=False)


# In[4]:


from sklearn.cluster import AgglomerativeClustering


# In[28]:


model = AgglomerativeClustering(n_clusters=4)


# In[31]:


cluster_labels = model.fit_predict(scaled_df)


# In[32]:


cluster_labels


# In[37]:


plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(data=df,x='mpg',y='horsepower',hue=cluster_labels,palette='viridis')


# ## Exploring Number of Clusters with Dendrograms
# 
# Make sure to read the documentation online!
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# 
# #### Assuming every point starts as its own cluster

# In[61]:


model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)


# In[39]:


cluster_labels = model.fit_predict(scaled_df)


# In[40]:


cluster_labels


# In[46]:


from scipy.cluster import hierarchy


# In[43]:


linkage_matrix = hierarchy.linkage(model.children_)


# In[45]:


linkage_matrix

# [point,point,distance,number of points in this cluster]


# In[48]:


plt.figure(figsize=(20,10),dpi=200)
dendro = hierarchy.dendrogram(linkage_matrix, truncate_mode='lastp',p=10)


# In[49]:


plt.figure(figsize=(20,10),dpi=200)
dendro = hierarchy.dendrogram(linkage_matrix, truncate_mode='level',p=3)


# In[50]:


scaled_df.describe()


# In[56]:


np.sqrt(len(scaled_df.columns))

# max distance theoretically


# In[54]:


scaled_df['mpg'].idxmax()


# In[55]:


scaled_df['mpg'].idxmin()


# In[57]:


car_a = scaled_df.iloc[320]
car_b = scaled_df.iloc[28]


# In[58]:


distance = np.linalg.norm(car_a-car_b)


# In[60]:


distance

# this is not the distance between clusters


# ### Creating a Model Based on Distance Threshold
# 
# * distance_threshold
#     * The linkage distance threshold above which, clusters will not be merged.

# In[62]:


model = AgglomerativeClustering(n_clusters=None, distance_threshold=2)


# In[63]:


cluster_labels = model.fit_predict(scaled_df)


# In[64]:


cluster_labels


# In[65]:


plt.figure(figsize=(20,10),dpi=200)
dendro = hierarchy.dendrogram(linkage_matrix, truncate_mode='lastp',p=11)


# In[ ]:




