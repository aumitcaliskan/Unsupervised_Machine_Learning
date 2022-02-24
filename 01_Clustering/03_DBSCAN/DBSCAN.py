#!/usr/bin/env python
# coding: utf-8

# # DBSCAN (Density Based Spatial Clustering of Applications with Noise)

# Using **density** of points as its main factor for assigning cluster labels. This creates the ability to find cluster segmentations that other algorithms have difficulty with.
# 
# Two key hyperparameters (epsilon and minimum number of points)
# * **epsilon** distance extended from a point.
# * **minimum number of points** minimum number of points in an epsilon distance.  
# 
# Point Types: Core, Border, Outlier
# * **Core** point with min. points in epsilon range (including itself)
# * **Border** in epsilon range of core point, but does not contain min number or points.
# * **Outlier** cannot be reached by points in a cluster assignment
# 
# How it works:
# * Pick a random point. 
# * Determine the point type.
# * Find the core point.
# * Then assign all points to a cluster or as an outlier.

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# In[24]:


# X,y = datasets.make_blobs(random_state=41, n_samples=300)


# In[25]:


# plt.scatter(*X.T, edgecolors='black')


# In[2]:


blobs = pd.read_csv('cluster_blobs.csv')


# In[3]:


blobs.head()


# In[4]:


sns.scatterplot(data=blobs,x='X1',y='X2')


# In[39]:


# X,y = datasets.make_moons(n_samples=500,noise=0.05)


# In[40]:


# plt.scatter(*X.T, edgecolors='black')


# In[5]:


moons = pd.read_csv('cluster_moons.csv')


# In[6]:


moons.head()


# In[7]:


sns.scatterplot(data=moons,x='X1',y='X2')


# In[67]:


# X,y = datasets.make_circles(n_samples=800,factor=0.3,noise=0.05)


# In[68]:


# plt.scatter(*X.T, edgecolors='black')


# In[41]:


circles = pd.read_csv('cluster_circles.csv')


# In[42]:


circles.head()


# In[44]:


sns.scatterplot(data=circles,x='X1',y='X2')


# In[69]:


def display_categories(model,data):
    labels = model.fit_predict(data)
    sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')


# In[70]:


from sklearn import cluster


# ## KMeans

# In[74]:


model = cluster.KMeans(n_clusters=2)


# In[73]:


display_categories(model,blobs)

# n_clusters = 3


# In[76]:


display_categories(model,moons)

# n_clusters = 2


# In[77]:


display_categories(model,circles)

# n_clusters = 2


# ## DBSCAN

# In[89]:


model = cluster.DBSCAN()
display_categories(model,blobs)


# In[90]:


model = cluster.DBSCAN(eps=0.2)
display_categories(model,moons)


# In[91]:


model = cluster.DBSCAN(eps=0.2)
display_categories(model,circles)


# In[92]:


two_blobs = pd.read_csv('cluster_two_blobs.csv')
two_blobs_outliers = pd.read_csv('cluster_two_blobs_outliers.csv')


# In[93]:


sns.scatterplot(data=two_blobs,x='X1',y='X2')


# In[94]:


sns.scatterplot(data=two_blobs_outliers,x='X1',y='X2')


# # Epsilon
# 
#     eps : float, default=0.5
#      |      The maximum distance between two samples for one to be considered
#      |      as in the neighborhood of the other. This is not a maximum bound
#      |      on the distances of points within a cluster. This is the most
#      |      important DBSCAN parameter to choose appropriately for your data set
#      |      and distance function.

# In[96]:


model = cluster.DBSCAN()
display_categories(model,two_blobs)


# In[98]:


model = cluster.DBSCAN(eps=0.001)
display_categories(model,two_blobs)


# In[99]:


model = cluster.DBSCAN(eps=10)
display_categories(model,two_blobs)


# In[101]:


model = cluster.DBSCAN(eps=1)
display_categories(model,two_blobs_outliers)


# In[104]:


np.sum(model.labels_ == -1)

# outlier numbers


# In[106]:


100 * np.sum(model.labels_ == -1) / len(model.labels_)

# percent of points  classified as outliers


# In[111]:


outlier_percents = []
number_of_outliers = []

for eps in np.linspace(0.001,7,200):
    model = cluster.DBSCAN(eps=eps)
    model.fit(two_blobs_outliers)
    number_of_outliers.append(np.sum(model.labels_ == -1))
    percent_outliers = 100 * np.sum(model.labels_ == -1) / len(model.labels_)
    outlier_percents.append(percent_outliers)


# In[118]:


sns.lineplot(x=np.linspace(0.001,7,200),y=number_of_outliers)
plt.xlim(0,2)
plt.ylim(0,10)
plt.ylabel('Number of Outliers')
plt.hlines(y=3,xmin=0,xmax=2,colors='red')


# In[121]:


sns.lineplot(x=np.linspace(0.001,7,200),y=outlier_percents)
plt.xlim(0,2)
plt.ylim(0,10)
plt.ylabel('Percent of Outliers')
plt.hlines(y=1,xmin=0,xmax=2,colors='red')


# ## Minimum Samples
# 
#      |  min_samples : int, default=5
#      |      The number of samples (or total weight) in a neighborhood for a point
#      |      to be considered as a core point. This includes the point itself.
#      

# In[122]:


outlier_percents = []
number_of_outliers = []

for n in np.arange(1,100):
    model = cluster.DBSCAN(min_samples=n)
    model.fit(two_blobs_outliers)
    number_of_outliers.append(np.sum(model.labels_ == -1))
    percent_outliers = 100 * np.sum(model.labels_ == -1) / len(model.labels_)
    outlier_percents.append(percent_outliers)


# In[124]:


sns.lineplot(x=np.arange(1,100), y=outlier_percents)
plt.xlabel('Min. Number of Samples')
plt.ylabel('Percentage of Outliers')


# In[127]:


model = cluster.DBSCAN(eps=1,min_samples=1)
display_categories(model,two_blobs_outliers)

# You will never get outlier if min_samples=1


# In[129]:


num_dim = two_blobs_outliers.shape[1]
model = cluster.DBSCAN(eps=1,min_samples=2*num_dim)
display_categories(model,two_blobs_outliers)


# In[ ]:




