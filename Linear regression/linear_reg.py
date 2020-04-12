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


# In[80]:


#load dataset and save as db
db=pd.read_csv('datasets/USA_Housing.csv')
#user input dataset(the column value to be predicted must be the last column)

print('finish')
# In[83]:


name='USA_Housing'
#specify the name of the dataset


# In[88]:


import os
path='F:/Study Material/College Study/Sem 6/DWM/Mini Project/DWM_mini_project/Linear regression/'+name
os.mkdir(path)


# In[89]:


os.chdir(path)


# In[90]:


db.head()
#gives the top 5 values of all coumns (needs to be there on as output)
db.to_csv('top5_rows.csv')


# In[91]:


db.describe().columns #gives all the interger columns


# In[92]:


newdb=db[db.describe().columns] 
#creates a new database with only columns having integer values


# In[93]:


newdb.head()


# In[94]:


sns.pairplot(newdb)
#this needs to be out as data analysis of the dataset along with a few more graphs
plt.savefig('pair_plot_dataset')


# In[95]:


newdb.hist()
#another graph
plt.savefig('histogram_dataset')


# In[96]:


newdb.plot.area()
#another graph
plt.savefig('area_plot_dataset')


# In[97]:


newdb.plot.line()
#graph4
plt.savefig('line_plot_dataset')


# In[98]:


newdb.columns


# In[99]:


y=newdb[newdb.columns[-1]] #assigns last columm of the dataset to y variable for prediction


# In[100]:


y.head()


# In[101]:


x=newdb[newdb.columns[:-1]]
#assigns all columns except the last one to the x variable


# In[102]:


x.head()


# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[105]:


model=LinearRegression()


# In[106]:


model.fit(x_train,y_train)


# In[107]:


pred=model.predict(x_test)


# In[108]:


pred[:6]
#predictions of top 6 values in table


# In[109]:


from sklearn import metrics


# In[110]:


sns.lineplot(y_test,pred)
#to play the relation between the prediction and the actual values
plt.savefig('lineplot_result')


# In[111]:


sns.regplot(y_test,pred)
plt.savefig('regression_plot_result')


# In[112]:


model.score(x,y)*100
#this is the accuracy percentage for this datatset accoring to our model


# In[113]:


import joblib


# In[114]:


joblib.dump(model,'general_model.sav')
#saves in general model file


# In[ ]:


#model complete

