#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,20))
#libraries


# In[4]:


#load dataset and save as db
db=pd.read_csv('datasets/USA_Housing.csv')
#user input dataset(the column value to be predicted must be the last column)


# In[2]:


name=input()
#specify the name of the dataset


# In[5]:


import os
path='F:/Study Material/College Study/Sem 6/DWM/Mini Project/DWM_mini_project/Linear regression/'+name
parent_file='F:/Study Material/College Study/Sem 6/DWM/Mini Project/DWM_mini_project/Linear regression'
os.mkdir(path)
#creates a new folder with the given name


# In[6]:


os.chdir(path)
#moves to that file for saving the outputs


# In[7]:


db.head()
#gives the top 5 values of all coumns (needs to be there on as output)
db.to_csv('top5_rows.csv')


# In[8]:


db.describe().columns #gives all the interger columns


# In[9]:


newdb=db[db.describe().columns] 
#creates a new database with only columns having integer values


# In[10]:


newdb.head()


# In[11]:


sns.pairplot(newdb)
#this needs to be out as data analysis of the dataset along with a few more graphs
plt.savefig('pair_plot_dataset')


# In[12]:


newdb.hist()
#another graph
plt.savefig('histogram_dataset')


# In[13]:


newdb.plot.area()
#another graph
plt.savefig('area_plot_dataset')


# In[14]:


newdb.plot.line()
#graph4
plt.savefig('line_plot_dataset')


# In[15]:


newdb.columns


# In[16]:


y=newdb[newdb.columns[-1]] #assigns last columm of the dataset to y variable for prediction


# In[17]:


y.head()


# In[18]:


x=newdb[newdb.columns[:-1]]
#assigns all columns except the last one to the x variable


# In[19]:


x.head()


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[22]:


model=LinearRegression()


# In[23]:


model.fit(x_train,y_train)


# In[24]:


pred=model.predict(x_test)


# In[25]:


pred[:6]
#predictions of top 6 values in table


# In[26]:


from sklearn import metrics


# In[27]:


sns.lineplot(y_test,pred)
#to play the relation between the prediction and the actual values
plt.savefig('lineplot_result')


# In[28]:


sns.regplot(y_test,pred)
plt.savefig('regression_plot_result')


# In[29]:


model.score(x,y)*100
#this is the accuracy percentage for this datatset accoring to our model


# In[30]:


import joblib


# In[31]:


joblib.dump(model,'general_model.sav')
#saves in general model file


# In[32]:


#model complete


# In[33]:


os.chdir(parent_file)
#comes back to parent directory


# In[ ]:




