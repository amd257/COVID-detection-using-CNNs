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
	preprocessed_img = np.expand_dims(img, axis=0)# expand image dimensions to fit the model input dimensions
	return preprocessed_img


@app.route("/" , methods =["GET" , "POST"])
def home():
	return(render_template("threeLabelPrediction.html")) #change this to homepage.html if you set that up


#function for making the three label prediction between COVID, Normal and Pneumonia images
@app.route("/threeLabelPrediction" , methods =["GET" , "POST"])
def threeLabelPrediction():
	prediction =[[0,0,0]]
	if request.method == "POST":
		# check if the post request has the file part
		if 'imagefile' not in request.files:
			app.logger.info('No image file selected')
			flash('No image file selected')
			return redirect(request.url)
	    
		imagefile = request.files["imagefile"]
		# if user does not select a file, browser can also submit an empty file without filename
		if imagefile.filename == "":
			app.logger.info('No image file selected')
			flash('No image file selected')
			return redirect(request.url)
		
		if imagefile and allowed_filename(imagefile.filename):
			filename=secure_filename(imagefile.filename)
			img_path= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			imagefile.save(img_path)
			app.logger.info(img_path)

			model = threelabel_model_loader() #Calling the three label model
			app.logger.info('Three Label Model has been loaded successfully')

			new_image = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale") #loading image using the keras image preprocessing library
			app.logger.info('Keras preprocessing image library has loaded the image')
			preprocessed_image = image_preprocessing(new_image)
			prediction = model.predict(preprocessed_image).tolist()
			app.logger.info('Prediction has been done: ')
			app.logger.info(prediction[0][0])
			app.logger.info(prediction[0][1])
			app.logger.info(prediction[0][2])
			os.remove(img_path)
			app.logger.info('Image removed as prediction has been made and it was no longer needed')
	CovidPred = round(((prediction[0][0])*100), 3 ) 
	NormalPred = round(((prediction[0][1])*100), 3 )
	PneumoniaPred = round(((prediction[0][2])*100), 3 )

	#response = {{}}

	#return jsonify(response)
	return render_template('threeLabelPrediction.html', 
							CovidPred= CovidPred, 
							NormalPred = NormalPred, 
							PneumoniaPred = PneumoniaPred)

if __name__ == '__main__':
	app.run(debug=True)