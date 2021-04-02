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

	if not "." in filename: #Only filenames with a . will be considered
		return False

	ext=filename.rsplit(".",1)[1] #Getting the file extension

	if ext.lower() in ALLOWED_EXTENSIONS:
		return True

	else:
		return False

def clear_images_directory():
	dir = UPLOAD_FOLDER
	for file in os.listdir(dir):
		os.remove(os.path.join(dir, file))

#Function for loading the binary model
def binary_model_loader():
	model_path = 'model/Xception_Binary_Final.h5' 
	loaded_model = load_model(model_path) #Loading the most optimal model for Binary Classification
	return loaded_model

#Function for loading the multi-class model
def multiClass_model_loader():
  model_path = 'model/Xception_MultiClass_Final.h5' 
  loaded_model = load_model(model_path) #Loading the most optimal model for Multi-Class Classification
  return loaded_model


#Function for preprocessing the image
def image_preprocessing(new_image):

	img = image.img_to_array(new_image)
	img = img / 255.0 #Rescaling the image by .1/255
	preprocessed_img = np.expand_dims(img, axis=0) #Expanding image dimensions to fit the model input dimensions
	return preprocessed_img



@app.route("/" , methods =["GET" , "POST"])
def homepage():
	return(render_template("homepage.html"))


#Function for making the binary classification between COVID and Non-COVID images
@app.route("/binaryClassification" , methods =["GET" , "POST"])
def binaryClassification():
	
	#Initialising the prediction variables to be returned to the frontend using jinja2
	CovidPred = 0
	NonCovidPred = 0
	
	#Initialising the image path
	img_path=""

	if request.method == "POST":
		#Removing previous images in the upload folder that are not required
		clear_images_directory()
		#Checking if the post request has the file part
		if 'imagefile' not in request.files:
			app.logger.info('No image file selected')
			flash('Please upload image file before clicking the predict button')
			return render_template('binaryClassification.html')
	    
		imagefile = request.files["imagefile"]

		# If user does not select a file, browser can also submit an empty file without filename
		if imagefile.filename == "":
			app.logger.info('No image file selected')
			flash('No image file selected. Please select an image file before clicking predict')
			return render_template('binaryClassification.html')
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			model = binary_model_loader() #Calling the binary classification Deep Learning model
			app.logger.info('Binary Classification Model has been loaded successfully')

			new_image = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale") #Loading image using the keras image preprocessing library
			app.logger.info('Keras preprocessing image library has loaded the image')
			preprocessed_image = image_preprocessing(new_image)
			
			prediction = model.predict(preprocessed_image).tolist() #Making predictions on the X-ray using the predict function
			app.logger.info('Prediction has been done:'+ str(prediction))

            #Checking if the prediction result for the two classes have been calculated. 
			#If condition is true, values are assigned to prediction variables for the respective classes.
			#Since the activation function is sigmoid, in this case, prediction[0][0] represents prediction for label 1, i.e, Non-COVID.
			#On the other hand, the value (1-prediction[0][0]) represents prediction for label 0, i.e,COVID-19.
			if((prediction[0][0])!=0):
				NonCovidPred = (prediction[0][0])*100
				NonCovidPred = format((NonCovidPred - (NonCovidPred % 0.01)),'.2f') #Formatting prediction result to two decimal places without rounding the digits 
				
				CovidPred = (1 - prediction[0][0])*100
				CovidPred = format((CovidPred - (CovidPred % 0.01)),'.2f') #Formatting prediction result to two decimal places without rounding the digits
			

		else:
			flash('The file selected was not in the correct format. Please upload X-ray image in .png, .jpg or .jpeg format')

	

	return render_template('binaryClassification.html', 
							CovidPred= CovidPred, 
							NonCovidPred = NonCovidPred,
							imgsrc = img_path)



#Function for making the multi-class classification between 
#COVID, Normal, Viral Pneumonia and other Non-COVID Lung Infection images
@app.route("/multiClassClassification" , methods =["GET" , "POST"])
def multiClassClassification():
	#Initialising prediction results list, image path, and the prediction variables for each class
	prediction =[[0,0,0,0]]
	img_path=""
	CovidPred = 0
	LungInfectionPred = 0
	NormalPred = 0
	ViralPneumoniaPred = 0


	if request.method == "POST":
		#Removing previous images in the upload folder that are not required
		clear_images_directory()
		#Checking if the post request has the file part
		if 'imagefile' not in request.files:
			app.logger.info('No image file selected')
			flash('Please upload image file before clicking the predict button')
			return render_template('multiClassClassification.html')
	    
		imagefile = request.files["imagefile"]

		# If user does not select a file, browser can also submit an empty file without filename
		if imagefile.filename == "":
			app.logger.info('No image file selected')
			flash('No image file selected. Please select an image file before clicking predict')
			return render_template('multiClassClassification.html')
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			model = multiClass_model_loader() #Calling the Multi-Class Deep Learning model
			app.logger.info('Multi-Class Model has been loaded successfully')

			new_image = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale") #Loading image using the keras image preprocessing library
			app.logger.info('Keras preprocessing image library has loaded the image')
			preprocessed_image = image_preprocessing(new_image)
			
			prediction = model.predict(preprocessed_image).tolist() #Making predictions on the X-ray using the predict function
			app.logger.info('Prediction Results for each class:'+ str(prediction))

		else:
			flash('The file selected was not in the correct format. Please upload X-ray image in png, jpg or jpeg format')

	
	#Checking if prediction results for all the classes have been calculated. 
	#If condition is true, values are assigned to prediction variables for the respective classes.
	if((prediction[0][0])!=0):
		CovidPred = (prediction[0][0])*100
		CovidPred = format((CovidPred - (CovidPred % 0.01)),'.2f') #Formatting prediction result to two decimal places without rounding the digits 
	
	if((prediction[0][1])!=0):
		LungInfectionPred  = (prediction[0][1])*100
		LungInfectionPred  = format((LungInfectionPred - (LungInfectionPred % 0.01)),'.2f') #Formatting prediction result to two decimal places without rounding the digits 
	
	if((prediction[0][2])!=0):
		NormalPred = (prediction[0][2])*100
		NormalPred = format((NormalPred  - (NormalPred % 0.01)),'.2f') #Formatting prediction result to two decimal places without rounding the digits  
	
	if((prediction[0][3])!=0):
		ViralPneumoniaPred = (prediction[0][3])*100
		ViralPneumoniaPred = format((ViralPneumoniaPred  - (ViralPneumoniaPred % 0.01)),'.2f') #Formatting prediction result to two decimal places without rounding the digits 
	


	return render_template('multiClassClassification.html', 
							CovidPred= CovidPred,
							LungInfectionPred = LungInfectionPred, 
							NormalPred = NormalPred,
							ViralPneumoniaPred = ViralPneumoniaPred, 
							imgsrc = img_path)

if __name__ == '__main__':
	app.run(debug=True)


