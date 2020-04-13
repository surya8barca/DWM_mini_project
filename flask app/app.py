#! C:\Users\HP\AppData\Local\Programs\Python\Python37\python.exe
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
plt.figure(figsize=(20,20))
from sklearn.model_selection import train_test_split

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib




app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')
# @app.route('/')
# def index():
# 	db=pd.read_csv('datasets/Fish.csv')
# 	newdb=db[db.describe().columns]
# 	y=newdb[newdb.columns[-1]]
# 	x=newdb[newdb.columns[:-1]]
    
# 	x_train,x_test,y_train,y_test=train_test_split(x,y)
	
# 	model=LinearRegression()
# 	#clf=joblib.load(model)
# 	model.fit(x_train,y_train)
# 	pred=model.predict(x_test)
# 	score=model.score(x,y)*100
# 	plt=sns.pairplot(newdb)
# 	plt.savefig('static/images/new_plot.png')
# 	plx=sns.regplot(y_test,pred)
# 	fig=plx.get_figure()
# 	fig.savefig("static/images/output.png")
# 	return render_template('results1.html',prediction = pred,plot='/static/images/output.png',score=score)

@app.route('/predict', methods=['POST'])
def predict():
	db=pd.read_csv('datasets/Fish.csv')
	newdb=db[db.describe().columns]
	y=newdb[newdb.columns[-1]]
	x=newdb[newdb.columns[:-1]]
    
	x_train,x_test,y_train,y_test=train_test_split(x,y)
	
	model=LinearRegression()
	#clf=joblib.load(model)
	model.fit(x_train,y_train)
	pred=model.predict(x_test)
	score=model.score(x,y)*100
	plt=sns.pairplot(newdb)
	plt.savefig('static/images/new_plot.png')
	plx=sns.regplot(y_test,pred)
	fig=plx.get_figure()
	fig.savefig("static/images/output.png")
	return render_template('results1.html',prediction = pred,plot='/static/images/output.png',score=score)


if __name__ == '__main__':
	app.run(debug=True)