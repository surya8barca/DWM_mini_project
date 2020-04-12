#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,20))
#libraries


# In[2]:


#load dataset and save as db
db=pd.read_csv('datasets/USA_Housing.csv')
#user input dataset(the column value to be predicted must be the last column)


# In[3]:


db.head()
#gives the top 5 values of all coumns (needs to be there on as output)


# In[4]:


db.describe().columns #gives all the interger columns


# In[5]:


newdb=db[db.describe().columns] 
#creates a new database with only columns having integer values


# In[6]:


newdb.head()


# In[7]:


sns.pairplot(newdb)
#this needs to be out as data analysis of the dataset along with a few more graphs


# In[8]:


newdb.hist()
#another graph


# In[9]:


newdb.plot.area()
#another graph


# In[10]:


newdb.plot.line()
#graph4


# In[11]:


newdb.columns


# In[12]:


y=newdb[newdb.columns[-1]] #assigns last columm of the dataset to y variable for prediction


# In[13]:


y.head()


# In[14]:


x=newdb[newdb.columns[:-1]]
#assigns all columns except the last one to the x variable


# In[15]:


x.head()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[18]:


model=LinearRegression()


# In[19]:


model.fit(x_train,y_train)


# In[20]:


pred=model.predict(x_test)


# In[21]:


pred[:6]
#predictions of top 6 values in table


# In[22]:


from sklearn import metrics


# In[24]:


sns.lineplot(y_test,pred)
#to play the relation between the prediction and the actual values


# In[37]:


plt.plot(y_test,pred,'.')


# In[48]:


fig=sns.regplot(y_test,pred)
plt.savefig('')


# In[44]:


model.score(x,y)*100
#this is the accuracy percentage for this datatset accoring to our model


# In[26]:


import joblib


# In[27]:


joblib.dump(model,'general_model.sav')
#saves in general model file


# In[ ]:


#model complete

