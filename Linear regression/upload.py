from flask import *  
import os
from werkzeug.utils import secure_filename
import pathlib

app = Flask(__name__)  
 

ALLOWED_EXTENSIONS = ['csv']
def allowed_file(filename):
            return '.' in filename and \
                filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def index():
    try:
        return '''
            <form method = "POST" action="/upload" enctype="multipart/form-data">
                <input type="file" name='dataset'>
                <input type="submit">
            </form>
        '''
    except Exception as e:
        return f"An Error Occured: {e}"




@app.route('/upload', methods=['POST'])

def detect():
    try:
        file = request.files['dataset']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # create the folders when setting up your app
            #os.makedirs(os.path.join(app.instance_path, 'UPLOAD_FOLDER'), exist_ok=True)
            path = os.path.join(pathlib.Path().absolute(), 'datasets', filename)
            file.save(path)
            return jsonify({"success": filename}), 200
    except Exception as e:
        return f"An Error Occured: {e}"
    
  
if __name__ == '__main__':  
    app.run(debug = True)  