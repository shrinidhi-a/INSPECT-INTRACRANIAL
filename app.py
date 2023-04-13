import os
from flask import Flask, render_template, request
from predictor import check


author = 'TEAM DELTA'

app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
@app.route('/index')

def index():
    return render_template('index.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/serviceForward')
def serviceForward():
    return render_template('detection.html')

@app.route('/serviceForwardSecond')
def serviceForwardSecond():
    return render_template('classification.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        dest = '/'.join([target, filename])
        print(dest)
        file.save(dest)

    status = check(filename)

    if status != 'Healthy':
        status = 'Brain Tumour Exists'

    return render_template('complete.html', image_name=filename, predvalue=status)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        dest = '/'.join([target, filename])
        print(dest)
        file.save(dest)
        
    status = check(filename)

    return render_template('complete.html', image_name=filename, predvalue=status)

if __name__ == "main":
    app.run(port=4555, debug=True)
