from flask import Flask, render_template, request
import tensorflow as tf
import os
import cv2
import numpy as np
import random
import string


app = Flask('stock_pricer')
model = tf.keras.models.load_model('/3-cnn-0-dense-128-nodes-mask-detection-1598982774.model')


@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'templates/uploads')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for images in request.files.getlist("file"):
        print(images)
        print("{} is the file name".format(images.filename))
        filename = images.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png") or (ext == ".jpeg"):
            print("File format is supported...")
            letters = string.ascii_lowercase
            newfilename = ''.join(random.sample(letters, 12))
            newfilename = newfilename + ext
            destination = "/".join([target, newfilename])
            print("Accept incoming file:", newfilename)
            print("Save it to:", destination)
            images.save(destination)
            image = cv2.imread(destination)
            image = cv2.resize(image, (300, 300))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image).reshape(-1, 300, 300, 1)
            results = model.predict(image)
            if results[0] == 0:
                text = 'Mask Detected'
            else:
                text = 'No mask'
            return render_template('resultsform.html', link=destination, result=text)


app.run()
