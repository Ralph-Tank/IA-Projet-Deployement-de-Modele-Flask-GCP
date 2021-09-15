#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 05:33:46 2021

@author: rtankoua
"""

# ::: Import modules et packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Definir une application Flask 
app = Flask(__name__)

# ::: Préparer un Modèle Keras :::
# Fichiers Modèle 
MODEL_ARCHITECTURE = './model/cnn_model_adam_20210915.json'
MODEL_WEIGHTS = './model/cnn_model_100_epochs_adam_20210915.h5'

# Charger le modèle à partir des fichiers externes
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Recuperer les poids du modèle
model.load_weights(MODEL_WEIGHTS)
print('Modèle chargé. Consulter http://127.0.0.1:5000/')


# ::: MODELE FUNCTIONS :::
def model_predict(img_path, model):
    '''
        Args:
            -- img_path : un lien de chemin où est stocké l'image donnée en entrée.
            -- model : un modèle CNN (keras).
    '''

    IMG = image.load_img(img_path).convert('L')
    print(type(IMG))

    # Pre-processing de l'image
    
    IMG_ = IMG.resize((333, 250))
    print(type(IMG_))
    IMG_ = np.asarray(IMG_)
    print(IMG_.shape)
    IMG_ = np.true_divide(IMG_, 255)
    IMG_ = IMG_.reshape(1, 333, 250, 1)
    print(type(IMG_), IMG_.shape)

    print(model)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    prediction = model.predict(IMG_)
    predictions = np.argmax(prediction,axis=1) 
    
    return predictions


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
    # Page principale
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

    # Constantes:
    classes = {'TRAIN': ['BACTERIA', 'NORMAL', 'VIRUS'],
               'VALIDATION': ['BACTERIA', 'NORMAL'],
               'TEST': ['BACTERIA', 'NORMAL', 'VIRUS']}

    if request.method == 'POST':

        # Recuperer le fichier de la requête 
        f = request.files['file']

        # Sauvegarder le fichier dans ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Faire une prédiction
        prediction = model_predict(file_path, model)

        predicted_class = classes['TRAIN'][prediction[0]]
        print('Prédiction du modèle : {}'.format(predicted_class.lower()))

        return str(predicted_class).lower()

if __name__ == '__main__':
    app.run(debug = True)