#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the library
import numpy as np
import pandas as pd
import missingno as msno

import plotly.express as px

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#load dataset

data=pd.read_csv('C:\\Users\\hprab\\Downloads\\archive\\water_potability.csv')
data.head()


# In[3]:


#data processing
data.info()


# In[4]:


data.describe()


# In[5]:


data.describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='viridis')


# In[6]:


data.isna().sum()


# In[15]:


data[10:25]


# In[7]:


msno.heatmap(data,cmap='plasma')


# In[8]:


msno.bar(data)


# In[9]:


sns.clustermap(data.corr(), cmap = "vlag", dendrogram_ratio = (0.1, 0.2), annot = True, linewidths = .8, figsize = (9,10))
plt.show()


# In[10]:


#handdling missing value

data["ph"].fillna(value = data["ph"].mean(), inplace = True),
data["Sulfate"].fillna(value = data["Sulfate"].mean(), inplace = True),
data["Trihalomethanes"].fillna(value = data["Trihalomethanes"].mean(), inplace = True)


# In[12]:


data.isna().sum()


# In[14]:


data[10:25]
 


# In[16]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
standardized_features = scaler.fit_transform(data)

print(standardized_features)


# In[ ]:




