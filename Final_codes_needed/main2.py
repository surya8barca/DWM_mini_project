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

    return score


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
                accuracy=mlcode(filename)
                pathtoplt='/static/'+filename+'/regression_plot_result.png'

                return render_template('results.html',accuracy=accuracy,plot=pathtoplt)
                # return jsonify({"path":pathtoplt}),200

    

  
if __name__ == '__main__':  
    app.run(debug = True)  