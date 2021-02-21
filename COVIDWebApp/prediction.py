import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array 
import numpy as np
from PIL import Image
import io
import re
import cv2


#function for loading the model(s)
def model_loader():
  model_path = 'model/Seq_Modelv3.h5'
  loaded_model = load_model(model_path)
  return loaded_model

#function for preprocessing the image
def image_preprocessing(image, target_size):

	#image=image.resize((256,256)) #converted image to array
	
	#image = image.reshape(32, 256, 256)
	image = image.img_to_array(image)
	image = image / 255.0 #rescale the image by .1/255
	image = np.expand_dims(image, axis=0)# expand image dimensions to fit the model input dimensions
	#image = tf.expand_dims(image, axis=0)
	return image
	'''img=np.array(image)
	if(img.ndim==3):
		gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray_img=img
	gray_img=gray_img/255
	resized_img=cv2.resize(gray_img,(target_size,target_size))
	print(resized_img.shape)
	reshaped_img=resized_img.reshape(list(resized_img.shape) + [1])
	print(reshaped_img.shape)
	return reshaped_img'''

def predict():
	imageFile_path ="A://COVID_CNN_MODEL//COVID_MODEL_2.0//Dataset//Test//COVID//COVID (3).png"
	nopp_image = image.load_img(imageFile_path, target_size=(256, 256))
	#image = Image.open("A://COVID_CNN_MODEL//COVID_MODEL_2.0//Dataset//Test//COVID//COVID (3).png")
	#processed_image= image_preprocessing(img, target_size= 256)
	nopp_image = image.img_to_array(nopp_image)
	nopp_image = nopp_image / 255.0 #rescale the image by .1/255
	processed_image = np.expand_dims(nopp_image, axis=0)
	prediction = model.predict(processed_image).tolist()
	return prediction


print("Loading the Keras model...")
model = model_loader() #Calling model globally so that it can be used freely by other methods
print("Keras model loaded")
print("Keras Backend: "+ K.backend())
print("image_data_format: "+ K.image_data_format()) #default is channels_last
print("floatx: ", K.floatx())
print("epsilon: ", K.epsilon())
predictedList = predict()
print(predictedList)
#K.set_image_data_format('channels_first')
#print("New image_data_format: "+ K.image_data_format())