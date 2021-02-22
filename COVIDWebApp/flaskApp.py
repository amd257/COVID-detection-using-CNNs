import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and \
	       filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/" , methods =["GET" , "POST"])
def index():

	#predict()

	return(render_template("index.html"))


	

@app.route("/predict" , methods =["GET" , "POST"])
def predict():

	if request.method == "POST":
		
		if request.files:
			image = request.files["image"]
		
		if image and allowed_file(image.filename):
			filename=secure_filename(image.filename)
			imgPath= os.path.join(app.config['UPLOAD_FOLDER'], filename)
			image.save(imgPath)
	
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
