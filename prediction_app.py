from flask import request 
from flask import jsonify 
from flask import Flask
import base64

import numpy as np
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.applications.resnet50 import preprocess_input as ResNet_preprocess_input, ResNet50
from keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input, VGG19
from keras.applications.inception_v3 import preprocess_input as Inception_preprocess_input, InceptionV3
from keras.applications.mobilenet import MobileNet, preprocess_input as MobileNet_preprocess_input

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.applications import ResNet50, InceptionV3, VGG19, MobileNet
from keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input, VGG19
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

import tensorflow as tf
import codecs, json 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
# initialize our Flask application and the Keras model

from flask_cors import CORS, cross_origin

app = Flask(__name__)

def load_model():
    global model, graph
    model = InceptionV3(weights="imagenet")
    print('--- Model loaded! --- \n')
    print ('='*50)
    graph = tf.get_default_graph()


def prepare_image(image, target_size):
    
    
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = Inception_preprocess_input(image) ## for pre-trained models, use preprocess input in line with  the model  used. 
                                              ## For custom model can use imagenet_utils.preprocess_input()

    # return the processed image
    return image

cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
@app.route('/predict', methods = ['GET', 'POST']) 
def predict():
    # initialize the data dictionary that will be returned
    message = request.get_json(force = True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    with  graph.as_default(): 
        image  = Image.open(io.BytesIO(decoded))
        if image is not None:
            processed_image = prepare_image(image, target_size = (224, 224))
            prediction = model.predict(processed_image)
            prediction = decode_predictions(prediction, top= 3)
        
            response = {
                'predictions':{
                    'probability_1': np.array(prediction[0][0][2]).tolist(),   
                    'label_1': np.array(prediction[0][0][1]).tolist(),
                
                    'probability_2': np.array(prediction[0][1][2]).tolist(),    
                    'label_2': np.array(prediction[0][1][1]).tolist(),
                
                    'probability_3': np.array(prediction[0][2][2]).tolist(),   
                    'label_3': np.array(prediction[0][2][1]).tolist()
                }
            }

         # return the data dictionary as a JSON response
            return jsonify(response) 
        else:
            return None, 'Error loading Image file'

if __name__ == "__main__":
    print ('='*50)
    print("\n*** Loading the Deep Learning Model ...... \n " )
    load_model()
    app.run(debug = True)


