import os
from flask import Flask, flash, request 
from flask import redirect, url_for,render_template,jsonify
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
SECRET_KEY = 'T\xee\x9d?\xf9\xcfMs\xd0\xf0\x01$\xc2\xd5\xd7XJ\xb9{\xfbd\xa6?_'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

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
def image_preprocessing(new_image):

	pp_img = image.img_to_array(new_image)
	pp_img = pp_img / 255.0 #rescale the image by .1/255
	pp_img = np.expand_dims(pp_img, axis=0)# expand image dimensions to fit the model input dimensions
	return pp_img


@app.route("/" , methods =["GET" , "POST"])
def index():
	return(render_template("index.html"))


@app.route("/predict" , methods =["GET" , "POST"])
def predict():

	prediction =[]
	if request.method == "POST":
		# check if the post request has the file part
		if 'imagefile' not in request.files:
			#app.logger.info('No image file selected')
			flash('No image file selected')
			return(redirect("index.html"))
			#return redirect(url_for('index'))
			#return redirect(request.url)
	    
		imagefile = request.files["imagefile"]
		# if user does not select a file, browser can also submit an empty file without filename
		if imagefile.filename == "":
			#app.logger.info('No image file selected')
			flash('No image file selected')
			return(redirect("index.html"))
			#return redirect(request.url)
			#return redirect(url_for('index'))
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			new_image = image.load_img(img_path, target_size=(256, 256)) #loading image using the keras image preprocessing library
			app.logger.info('keras preprocessing image library has loaded the image')
			preprocessed_image = image_preprocessing(new_image)
			prediction = model.predict(preprocessed_image).tolist()
			app.logger.info('Prediction has been done: ')
			app.logger.info(prediction[0][0])
			app.logger.info(prediction[0][1])
			app.logger.info(prediction[0][2])
			os.remove(img_path)
			app.logger.info('Image removed as prediction has been made and it was no longer needed')
	

	CovidPred = prediction[0][0]  
	NormalPred = prediction[0][1] 
	PneumoniaPred = prediction[0][2] 
	
	return render_template('predict.html', 
							CovidPred= CovidPred, 
							NormalPred = NormalPred, 
							PneumoniaPred = PneumoniaPred)


if __name__ == '__main__':
	app.run(debug=True)


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

