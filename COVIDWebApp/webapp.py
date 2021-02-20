from flask import Flask, render_template, request,jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image 
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size=256

app = Flask(__name__) 

#function for loading the model(s)
def model_loader():
  model_path = 'model/Seq_Modelv3.h5'
  loaded_model = load_model(model_path)
  return loaded_model

model = model_loader()

#label_dict={0:'COVID', 1:'Normal', 2:'ViralPneumonia'}

def preprocess(img):

	img=image.img_to_array(img) #converted image to array
	img=img/255 #rescale the image by .1/255
	preprocessed_img = np.expand_dims(img, axis=0)# expand image dimensions to fit the model input dimensions
	#reshaped=resized.reshape(3,img_size,img_size,1) #Input 0 of layer sequential_2 is incompatible with the layer: expected axis -1 of input shape to have value 3 but received input with shape [None, 256, 256, 1]
	return preprocessed_img

@app.route("/")
def home():
	return(render_template("home.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	hardik_img = image.load_img(temp_file.name, target_size=(256, 256)) #loading image using keras preprocessing image library
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">