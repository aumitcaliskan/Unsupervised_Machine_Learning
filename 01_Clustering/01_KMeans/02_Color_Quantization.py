#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# .jpg .png ---> numpy array with matplotlib.image


# In[2]:


# Grayscale : Color range goes from black to white.
# Color images can be represented as a combination of Red, Green, Blue. (RGB) In range(0-255). (213,111,56)
# Shape of the color array has 3 dimensions. Height, Width, Color Channels. (1280,720,3)
# We can reshape the image to an X array feature set with features R,G,B. 


# In[5]:


image_as_array = mpimg.imread('palm_trees.jpg')


# In[8]:


image_as_array.shape

# (height, width, color)


# In[10]:


plt.figure(dpi=200)
plt.imshow(image_as_array)


# In[11]:


# (H,W,C) --- > 2D (H*W,C)


# In[12]:


(h,w,c) = image_as_array.shape


# In[13]:


h


# In[14]:


image_as_array2d = image_as_array.reshape(h*w,c)


# In[16]:


image_as_array2d.shape


# In[17]:


from sklearn.cluster import KMeans


# In[18]:


model = KMeans(n_clusters=6)


# In[19]:


labels = model.fit_predict(image_as_array2d)


# In[20]:


labels


# In[22]:


rgb_codes = model.cluster_centers_.round(0).astype(int)


# In[23]:


rgb_codes


# In[24]:


rgb_codes[labels]


# In[25]:


image_as_array2d


# In[27]:


quantized_image = np.reshape(rgb_codes[labels],(h,w,c))


# In[30]:


plt.figure(dpi=200)
plt.imshow(quantized_image)


# In[ ]:




