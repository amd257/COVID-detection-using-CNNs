import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array 
from flask import Flask
from flask import jsonify
from flask import request,render_template
import numpy as np
import base64
from PIL import Image
import io
import re
import cv2


app = Flask(__name__) 

#function for loading the model(s)
def model_loader():
  model_path = 'model/Seq_Modelv3.h5'
  loaded_model = load_model(model_path)
  return loaded_model

#function for preprocessing the image
def image_preprocessing(image, target_size):

	img=np.array(image)
	if(img.ndim==3):
		gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray_img=img
	gray_img=gray_img/255
	resized_img=cv2.resize(gray_img,(target_size,target_size))
	print(resized_img.shape)
	reshaped_img=resized_img.reshape(list(resized_img.shape) + [1])
	print(reshaped_img.shape)
	return reshaped_img

	#image=image.resize((256,256)) #converted image to array
	#image = image / 255.0
	#image=image/255 #rescale the image by .1/255
	#image = img_to_array(image)
	#image = image.reshape(32, 256, 256)
	#image = np.expand_dims(image, axis=0)
	#image = tf.expand_dims(image, axis=0)# expand image dimensions to fit the model input dimensions
	#return image

print("Loading the Keras model...")
model = model_loader() #Calling model globally so that it can be used freely by other methods
print("Keras model loaded")
print("Keras Backend: "+ K.backend())
print("image_data_format: "+ K.image_data_format()) #default is channels_last
print("floatx: ", K.floatx())
print("epsilon: ", K.epsilon())
#K.set_image_data_format('channels_first')
#print("New image_data_format: "+ K.image_data_format())

@app.route("/")
def home():
	return(render_template("home.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('Inside Predict function')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image= image_preprocessing(image, target_size= 256)
	prediction = model.predict(processed_image).tolist()
	response = {
	'prediction': {
		'COVID': prediction[0][0],
		'Normal': prediction[0][1],
		'ViralPneumonia' : prediction[0][2]
		}
	}
	return jsonify(response)


app.run(debug=True)
