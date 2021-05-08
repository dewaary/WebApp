from __future__ import division, print_function
# coding=utf-8
import sys
import os
import cv2
import numpy as np

import tensorflow as tf
# Keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image

sess = tf.Session()
graph = tf.get_default_graph()


# Flask utils
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/ResNet50_100_epochs.h5'

# Load your trained model
set_session(sess)
model = load_model(MODEL_PATH)
model._make_predict_function()

import pickle

with open('models/deskripsi.pkl', 'rb') as d:
    Deskripsi = pickle.load(d)

with open('models/categoriLabels.pkl', 'rb') as f:
    categories = sorted(pickle.load(f))
# # You can also use pretrained model from Keras
# # Check https://keras.io/applications/
# #from keras.applications.resnet50 import ResNet50
# #model = ResNet50(weights='imagenet')
# #model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')

def cvtRGB(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

def predict_one_image(file_path:str, model):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
    img = np.reshape(img, (1, 256, 256, 3))
    img = img/255.
    with graph.as_default():
        set_session(sess)
        pred = model.predict(img)
        class_num = np.argmax(pred)
    return class_num, np.max(pred)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        preds, _ = predict_one_image(file_path, model)
        content = {
            'class' : (
                str(categories[preds]),
            ),
            'class1': (
                str(Deskripsi[categories[preds]]),
            )
        }

        return jsonify(content)
    return None


if __name__ == '__main__':
    app.run(debug=True)

