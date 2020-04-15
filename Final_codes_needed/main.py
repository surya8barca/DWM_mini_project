from flask import *  
import os
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap 
import pathlib


app = Flask(__name__)  
Bootstrap(app)

ALLOWED_EXTENSIONS = ['csv']
def allowed_file(filename):
            return '.' in filename and \
                filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def mlcode(dataset):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    #from sklearn import tree
    from sklearn.linear_model import LinearRegression
    plt.figure(figsize=(20,20))
    db=pd.read_csv('datasets/'+dataset)
    import os
    parent_file=os.path.join(pathlib.Path().absolute(),"static")
    os.makedirs(os.path.join(parent_file, dataset), exist_ok=True)
    os.chdir(os.path.join(parent_file, dataset))
    db.describe().columns
    newdb=db[db.describe().columns] 
    newdb.head()
    sns.pairplot(newdb)
    plt.savefig('pair_plot_dataset')
    newdb.hist()
    plt.tight_layout()
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
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])  
    
    
    pred=model.predict(x_test)
    
    from sklearn import metrics
    mae=metrics.mean_absolute_error(y_test, pred)
    mse=metrics.mean_squared_error(y_test,pred)
    rmse=metrics.mean_squared_error(y_test,pred)
    sns.lineplot(y_test,pred)
    
    plt.savefig('lineplot_result')
    sns.regplot(y_test,pred,'.')
    plt.savefig('regression_plot_result')
    score=model.score(x,y)*100
   
    import joblib
    joblib.dump(model,'model.sav')
    #for saving the weights of sttributes
    from pandas.plotting import table 
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, coeff_df, loc='center')  # where df is your data frame

    plt.savefig('mytable.png',bbox_inches='tight')

    #Decision tree classifier
    #from sklearn.tree import DecisionTreeClassifier
    #clf = DecisionTreeClassifier(max_depth=3)

# Train Decision Tree Classifer
    #clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
    #y_pred = clf.predict(x_test)
    #tree.plot_tree(clf.fit(x, y))
    #plt.savefig('tree')
    os.chdir(parent_file)


    return score,mae,mse,rmse


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])

def detect():
    
            file = request.files['dataset']

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # create the folders when setting up your app
                path = os.path.join(pathlib.Path().absolute(), 'datasets', filename)
                file.save(path)
                accuracy,mae,mse,rmse=mlcode(filename)
                #data analysis
                pairplot='/static/'+filename+'/pair_plot_dataset.png'
                areaplot='/static/'+filename+'/area_plot_dataset.png'
                histogram='/static/'+filename+'/histogram_dataset.png'
                lineplot='/static/'+filename+'/line_plot_dataset.png'
                weightsimg='/static/'+filename+'/mytable.png'
                #results
                reg_plot='/static/'+filename+'/regression_plot_result.png'
                lineplot_result='/static/'+filename+'/lineplot_result.png'
                #download
                model_file='/static/'+filename+'/model.sav'
                #tree='/static/'+filename+'/tree.png'
                return render_template('results.html',accuracy=accuracy,model_file=model_file,lineplot=lineplot,histogram=histogram,pairplot=pairplot,areaplot=areaplot,lineplot_result=lineplot_result,reg_plot=reg_plot,weightsimg=weightsimg,mae=mae,mse=mse,rmse=rmse)
                

    

  
if __name__ == '__main__':  
    app.run(debug = True)  