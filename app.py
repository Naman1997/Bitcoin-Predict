import os
# from flask_cors import CORS, cross_origin
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import send
import json

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/defaults')
def run_script():
    return render_template('defaults.html')
    # import send
    # return send.send_predictions(30) #30 for 30 days

@app.route('/configurable', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/RNN/<time>/', methods=['GET'])
def RNN(time):
    test_data = send.RNNtestdata(time)
    prediction_data = send.RNNpredictions(time)
    data = {}
    data['test'] = test_data.tolist()
    data['pred'] = prediction_data.tolist()
    return json.dumps(data, indent=4)

@app.route('/LR/<time>/', methods=['GET'])
def LR(time):
    total_data = send.LRpredictions(time)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(total_data)
    print()
    test_data = total_data[0]
    prediction_data = total_data[1]
    data = {}
    data['test'] = test_data
    data['pred'] = prediction_data
    return json.dumps(data, indent=4)


if __name__ == "__main__":
    app.run(debug=True, port=8080)  
