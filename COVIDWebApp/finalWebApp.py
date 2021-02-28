import os
import io, re
import logging
from flask import Flask, flash, request 
from flask import redirect, url_for,render_template,jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array 
import numpy as np

UPLOAD_FOLDER = 'static//images//'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}
SECRET_KEY = 'C?\xe7s\xb0\xe3\n\x87\\\xd8\xa3\xee\xdf\x13\x06\\\xee\x18\x03\xf2\xc0\x95x\xdd'

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

def clear_images_directory():
	dir = UPLOAD_FOLDER
	for file in os.listdir(dir):
		os.remove(os.path.join(dir, file))

#function for loading the binary model(s)
def binary_model_loader():
	model_path = 'model/SeqModel2_Binaryv2.h5'
	loaded_model = load_model(model_path)
	return loaded_model

#function for loading the three label model(s)
def threelabel_model_loader():
  model_path = 'model/SeqModel2_ThreeLabelv1.h5'
  loaded_model = load_model(model_path)
  return loaded_model


#function for preprocessing the image
def image_preprocessing(new_image):

	img = image.img_to_array(new_image)
	img = img / 255.0 #rescale the image by .1/255
	preprocessed_img = np.expand_dims(img, axis=0)# expands image dimensions to fit the model input dimensions
	return preprocessed_img



@app.route("/" , methods =["GET" , "POST"])
def homepage():
	return(render_template("homepage.html"))


#function for making the three label prediction between COVID, Normal and Pneumonia images
@app.route("/binaryPrediction" , methods =["GET" , "POST"])
def binaryPrediction():
	
	#Initialising the variables to be returned to the frontend using jinja2
	CovidPred = 0
	NonCovidPred = 0
	img_path=""

	if request.method == "POST":
		#remove previous images in the upload folder that are not required
		clear_images_directory()
		# check if the post request has the file part
		if 'imagefile' not in request.files:
			app.logger.info('No image file selected')
			flash('Please upload image file before clicking the predict button')
			return render_template('binaryPrediction.html')
	    
		imagefile = request.files["imagefile"]

		# if user does not select a file, browser can also submit an empty file without filename
		if imagefile.filename == "":
			app.logger.info('No image file selected')
			flash('No image file selected. Please select an image file before clicking predict')
			return render_template('binaryPrediction.html')
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			model = binary_model_loader() #Calling the binary classification Deep Learning model
			app.logger.info('Binary Classification Model has been loaded successfully')

			new_image = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale") #loading image using the keras image preprocessing library
			app.logger.info('Keras preprocessing image library has loaded the image')
			preprocessed_image = image_preprocessing(new_image)
			prediction = model.predict(preprocessed_image).tolist()
			app.logger.info('Prediction has been done:-')
			#if prediction[0][0] >=0.5:
				#app.logger.info('The model is {:.2%} percent confident that this is a Non-COVID case ', prediction[0][0]) #As Non-COVID is label 1 in the model
			#else:
				#app.logger.info('The model is {:.2%} percent confident that this is a COVID case ', 1-prediction[0][0]) #As COVID is label 0 in the model

			NonCovidPred = round(((prediction[0][0])*100), 2 )
			CovidPred = round(((1 - prediction[0][0])*100), 2 ) 
	

		else:
			flash('The file selected was not in the correct format. Please upload X-ray image in .png, .jpg or .jpeg format')

	

	return render_template('binaryPrediction.html', 
							CovidPred= CovidPred, 
							NonCovidPred = NonCovidPred,
							imgsrc = img_path)







#function for making the three label prediction between COVID, Normal and Pneumonia images
@app.route("/threeLabelPrediction" , methods =["GET" , "POST"])
def threeLabelPrediction():
	prediction =[[0,0,0]]
	img_path=""

	if request.method == "POST":
		#remove previous images in the upload folder that are not required
		clear_images_directory()
		# check if the post request has the file part
		if 'imagefile' not in request.files:
			app.logger.info('No image file selected')
			flash('Please upload image file before clicking the predict button')
			return render_template('threeLabelPrediction.html')
	    
		imagefile = request.files["imagefile"]

		# if user does not select a file, browser can also submit an empty file without filename
		if imagefile.filename == "":
			app.logger.info('No image file selected')
			flash('No image file selected. Please select an image file before clicking predict')
			return render_template('threeLabelPrediction.html')
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			model = threelabel_model_loader() #Calling the three label Deep Learning model
			app.logger.info('Three Label Model has been loaded successfully')

			new_image = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale") #loading image using the keras image preprocessing library
			app.logger.info('Keras preprocessing image library has loaded the image')
			preprocessed_image = image_preprocessing(new_image)
			prediction = model.predict(preprocessed_image).tolist()
			app.logger.info('Prediction has been done: ')
			app.logger.info(prediction[0][0])
			app.logger.info(prediction[0][1])
			app.logger.info(prediction[0][2])

		else:
			flash('The file selected was not in the correct format. Please upload X-ray image in png, jpg or jpeg format')

	CovidPred = round(((prediction[0][0])*100), 2 ) 
	NormalPred = round(((prediction[0][1])*100), 2 )
	PneumoniaPred = round(((prediction[0][2])*100), 2 )

	return render_template('threeLabelPrediction.html', 
							CovidPred= CovidPred, 
							NormalPred = NormalPred, 
							PneumoniaPred = PneumoniaPred,
							imgsrc = img_path)

if __name__ == '__main__':
	app.run(debug=True)



