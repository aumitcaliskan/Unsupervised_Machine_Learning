#!/usr/bin/env python
# coding: utf-8

# Find reasonable clusters of customers for marketing segmentation and study. What we end up doing with those clusters would depend **heavily** on the domain itself, in this case, marketing.
# 
# https://archive.ics.uci.edu/ml/datasets/bank+marketing

# ### bank client data:
#     1 - age (numeric)
#     2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
#     3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
#     4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
#     5 - default: has credit in default? (categorical: 'no','yes','unknown')
#     6 - housing: has housing loan? (categorical: 'no','yes','unknown')
#     7 - loan: has personal loan? (categorical: 'no','yes','unknown')
#     # related with the last contact of the current campaign:
#     8 - contact: contact communication type (categorical: 'cellular','telephone')
#     9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
#     10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
#     11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#     # other attributes:
#     12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#     13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#     14 - previous: number of contacts performed before this campaign and for this client (numeric)
#     15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
#     # social and economic context attributes
#     16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#     17 - cons.price.idx: consumer price index - monthly indicator (numeric)
#     18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
#     19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
#     20 - nr.employed: number of employees - quarterly indicator (numeric)
#     21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('bank-full.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


# Domain experience for further explaination about data. Do your own search or contact and get information from someone.
# Try to perform visualization as much as you can.


# In[16]:


plt.figure(figsize=(12,6))
sns.histplot(data=df, x='age',bins=40, kde=True, hue='loan')


# In[24]:


plt.figure(figsize=(12,6))
sns.histplot(data=df[df['pdays'] != 999], x='pdays')

# 999 means client was not previously contacted


# In[25]:


df['contact'].unique()


# In[26]:


df['duration'].value_counts()


# In[32]:


plt.figure(figsize=(12,6))
sns.histplot(data=df, x='duration',kde=True, hue='contact')
plt.xlim(0,1000)


# In[33]:


sns.countplot(data=df,x='contact')


# In[41]:


df['job'].value_counts().index


# In[43]:


plt.figure(figsize=(12,6))

sns.countplot(data=df,x='job',order=df['job'].value_counts().index)
plt.xticks(rotation=90);


# In[46]:


plt.figure(figsize=(12,6))

sns.countplot(data=df,x='education',order=df['education'].value_counts().index, hue='default')
plt.xticks(rotation=90);


# In[48]:


sns.countplot(data=df, x='default')


# In[49]:


df['default'].value_counts()


# In[50]:


df['loan'].value_counts()


# In[51]:


X = pd.get_dummies(df)


# In[52]:


X


# In[53]:


from sklearn.preprocessing import StandardScaler


# In[55]:


scaler = StandardScaler()


# In[56]:


scaled_X = scaler.fit_transform(X)

# there is no label, so no there is  no test. That is why we fit_transform all data


# In[57]:


from sklearn.cluster import KMeans


# In[62]:


model = KMeans(n_clusters=2)


# In[60]:


model.get_params()


# **n_clusters :** int, default=8
#     The number of clusters to form as well as the number of
#     centroids to generate.  
#     
# **init :** {'k-means++', 'random'}, callable or array-like of shape (n_clusters, n_features), default='k-means++'
# 
# **n_init :** int, default=10
#     Number of time the k-means algorithm will be run with different centroid seeds.  
#     The final results will be the best output of n_init consecutive runs in terms of inertia.
#     
# **tol :** float, default=1e-4
#     Relative tolerance with regards to Frobenius norm of the difference
#     in the cluster centers of two consecutive iterations to declare
#     convergence.

# ### Creating and Fitting a KMeans Model
# 
# Note of our method choices here:
# 
# * fit(X[, y, sample_weight])
#     * Compute k-means clustering.
# 
# * fit_predict(X[, y, sample_weight])
#     * Compute cluster centers and predict cluster index for each sample.
# 
# * fit_transform(X[, y, sample_weight])
#     * Compute clustering and transform X to cluster-distance space.
# 
# * predict(X[, sample_weight])
#     * Predict the closest cluster each sample in X belongs to.

# In[89]:


cluster_labels = model.fit_predict(scaled_X)


# In[90]:


cluster_labels


# In[91]:


X['Cluster'] = cluster_labels


# In[92]:


# X


# In[93]:


plt.figure(figsize=(16,6))
X.corr()['Cluster'].iloc[:-1].sort_values().plot(kind='bar')


# ### Attributes
# 
# **cluster_centers_ :** ndarray of shape (n_clusters, n_features)
#     Coordinates of cluster centers.  
#     
# **labels_ :** ndarray of shape (n_samples,)
#     Labels of each point
#     
# **inertia_ :** float
#     Sum of squared distances of samples to their closest cluster center,
#     weighted by the sample weights if provided
# 

# In[94]:


model.inertia_


# In[102]:


ssd = []

for k in range(2,10):
    model = KMeans(n_clusters=k)
    model.fit(scaled_X)
    ssd.append(model.inertia_)


# In[103]:


ssd


# In[105]:


plt.plot(range(2,10), ssd, 'o--')


# In[107]:


pd.Series(ssd).diff()


# ### MiniBatchKMeans :
# 
# Alternative online implementation that does incremental updates of the centers positions using mini-batches. For large scale learning (say n_samples > 10k) MiniBatchKMeans is probably much faster than the default batch implementation.

# **In practice, the k-means algorithm is very fast (one of the fastest
# clustering algorithms available), but it falls in local minima. That's why
# it can be useful to restart it several times.**

# In[ ]:




