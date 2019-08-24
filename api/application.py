from flask import Flask, request, g
from flask_cors import CORS
import json
import cv2
import numpy as np
import pickle
import keras
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
import tensorflow as tf

app = Flask(__name__)
CORS(app)


@app.route("/classify", methods=['POST'])
def classify():
  file = request.files['image']
  image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = cv2.resize(image,(299,299))
  image = np.reshape(image,[1,299,299,3])
  image = image / 255.

  data = { 'success': False }
  with graph.as_default():
    result = model.predict(image, steps = 1)
    preds = result.argsort()[0][-10:][::-1]
    probs = result[0][preds]
    label_map = decoder
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
    predictions = [label_map[k] for k in preds]
    data['success'] = True
    data['predictions'] = [{ 'category': pred[0], 'probability': float(pred[1]) } for pred in list(zip(predictions, probs))]

  return json.dumps(data)

@app.route("/add", methods=['POST'])
def addImageToDataset():
  file = request.files['image']
  category = request.form['category']
  return "Added!"

def uploadModel(filepath, num_classes = 257):
  base_xc_model = Xception(input_shape=(299, 299, 3), weights='imagenet', include_top=False)
  #Add your own integrated pooling layer, then your softmax output
  pooling = GlobalAveragePooling2D()(base_xc_model.output)
  predictions = Dense(num_classes, activation='softmax')(pooling)

  #Consolidate the xc_model
  xc_model = Model(inputs=base_xc_model.input,outputs=predictions)

  # Train only the top layers that we added
  # Freeze all layers of the based model that is already pre-trained.
  for layer in base_xc_model.layers:
      layer.trainable = False
  xc_model.load_weights(filepath + ".h5")

  #Prepare a compiler for the model to use to train
  xc_model.compile(optimizer=keras.optimizers.Nadam(),
                  loss='categorical_crossentropy',  # categorical_crossentropy for multi-class classifier
                  metrics=['accuracy'])
  return xc_model

def uploadDecoder(filepath):
  decoder = None
  with open( filepath + ".pickle", 'rb') as handle:
    decoder = pickle.load(handle)
  return decoder

global graph
graph = tf.get_default_graph()
model = uploadModel('../artifacts/top_weights_CalTech256_5.29_nadam_b512')
decoder = uploadDecoder('../artifacts/CalTech256_decoder')
image = None

if __name__ == '__main__':
    app.run(debug=True)
