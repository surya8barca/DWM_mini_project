#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,20))
#libraries


# In[237]:


#load dataset and save as db
db=pd.read_csv('datasets/Real estate.csv')
#user input dataset(the column value to be predicted must be the last column)


# In[239]:


db.head()
#gives the top 5 values of all coumns (needs to be there on as output)


# In[232]:


db.describe().columns #gives all the interger columns


# In[240]:


newdb=db[db.describe().columns] 
#creates a new database with only columns having integer values


# In[242]:


newdb.head()


# In[244]:


sns.pairplot(newdb)
#this needs to be out as data analysis of the dataset along with a few more graphs


# In[246]:


newdb.hist()
#another graph


# In[249]:


newdb.plot.area()
#another graph


# In[250]:


newdb.plot.line()
#graph4


# In[252]:


newdb.columns


# In[253]:


y=newdb[newdb.columns[-1]] #assigns last columm of the dataset to y variable for prediction


# In[254]:


y.head()


# In[256]:


x=newdb[newdb.columns[:-1]]
#assigns all columns except the last one to the x variable


# In[257]:


x.head()


# In[258]:


from sklearn.model_selection import train_test_split


# In[259]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[260]:


model=LinearRegression()


# In[261]:


model.fit(x_train,y_train)


# In[263]:


pred=model.predict(x_test)


# In[266]:


pred[:6]
#predictions of top 6 values in table


# In[267]:


from sklearn import metrics


# In[268]:


sns.lineplot(y_test,pred)
#to play the relation between the prediction and the actual values


# In[269]:


model.score(x,y)*100
#this is the accuracy percentage for this datatset accoring to our model


# In[270]:


import joblib


# In[271]:


joblib.dump(model,'general_model.sav')
#saves in general model file


# In[ ]:


#model complete

