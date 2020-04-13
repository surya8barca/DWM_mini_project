#!/usr/bin/env python
# coding: utf-8

# In[15]:


def dwm(dataset,filename):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    plt.figure(figsize=(20,20))
    db=pd.read_csv('datasets/'+dataset+'.csv')
    import os
    path='F:/Study Material/College Study/Sem 6/DWM/Mini Project/DWM_mini_project/Linear regression/'+filename
    parent_file='F:/Study Material/College Study/Sem 6/DWM/Mini Project/DWM_mini_project/Linear regression'
    os.mkdir(path)
    os.chdir(path)
    db.describe().columns
    newdb=db[db.describe().columns] 
    newdb.head()
    sns.pairplot(newdb)
    plt.savefig('pair_plot_dataset')
    newdb.hist()
    plt.savefig('histogram_dataset')
    newdb.plot.area()
    plt.savefig('area_plot_dataset')
    newdb.plot.line()
    plt.savefig('line_plot_dataset')
    newdb.columns
    y=newdb[newdb.columns[-1]] 
    x=newdb[newdb.columns[:-1]]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y)
    model=LinearRegression()
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    from sklearn import metrics
    sns.lineplot(y_test,pred)
    plt.savefig('lineplot_result')
    sns.regplot(y_test,pred,'.')
    plt.savefig('regression_plot_result')
    score=model.score(x,y)*100
    import joblib
    joblib.dump(model,'general_model.sav')
    os.chdir(parent_file)

    return score;


# In[16]:


dataset=input("Enter the name of dataset: ")
filename=input("Enter the name of file you want to add the files in: ")


# In[17]:


accuracy=dwm(dataset,filename)
print("Accuracy :{} ".format(accuracy) )


# In[18]:


accuracy


# In[ ]:




