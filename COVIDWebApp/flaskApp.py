import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array 
import numpy as np
import io
import re
import logging

UPLOAD_FOLDER = 'static//images//'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_filename(filename):

	if not "." in filename: #only filenames was a . will be considered
		return False

	ext=filename.rsplit(".",1)[1] #getting the file extension

	if ext.lower() in ALLOWED_EXTENSIONS:
		return True

	else:
		return False


#function for loading the model(s)
def model_loader():
  model_path = 'model/Seq_Modelv4.h5'
  loaded_model = load_model(model_path)
  return loaded_model

model = model_loader() #Calling model globally so that it can be used freely by other methods

#function for preprocessing the image
def preprocess_image(new_image):

	pp_img = image.img_to_array(new_image)
	pp_img = pp_img / 255.0 #rescale the image by .1/255
	pp_img = np.expand_dims(pp_img, axis=0)# expand image dimensions to fit the model input dimensions
	return pp_img


@app.route("/" , methods =["GET" , "POST"])
def index():

	return(render_template("index.html"))


@app.route("/predict" , methods =["GET" , "POST"])
def predict():

	if request.method == "POST":
		
		if 'imagefile' not in request.files:
			app.logger('No image file selected')
			return redirect(request.url)
	    
		imagefile = request.files["imagefile"]
		if imagefile.filename == "":
			app.logger('No image file selected')
			return redirect(request.url)
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			new_image = image.load_img(img_path, target_size=(256, 256)) #loading image using the keras image preprocessing library
			app.logger.info('keras preprocessing image library has loaded the image')
			preprocessed_image = preprocess_image(new_image)
			prediction = model.predict(preprocessed_image).tolist()
			app.logger.info('Prediction has been done: ')
			app.logger.info(prediction)
			
	
	return(render_template("index.html")) 






'''@app.route("/predict", methods=['GET' , 'POST'])
def predict():
	if request.method == 'POST':
		if 'file' not in request.files: #to check if the post request has the file
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		#if the user does not select a file, the browser might submit an empty file without a filename
		if file.filename == '' :
			flash('No file selected')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			imgPath = file.filename
			filename=secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('Predict function has saved the image file in the upload folder')
			flash('image path is: ', imgPath)
			
'''
if __name__ == '__main__':
	app.run(debug=True)
