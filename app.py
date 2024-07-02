import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model


 
#1. Create a streamlit title widget, this will be shown first
st.title("SeeFood/HotDog NotHotDog")
 
upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
	# Open the image with PIL and reshape it to 224,224,3
	image= Image.open(upload)
	
	# Instantiate ImageDataGenerator to perform pre-processing on the loaded image
	image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	image_resize = 224 # the target size of VGG
	
	
	# reshape it to 224,224,3 and display
	image_resized = image.resize((image_resize, image_resize))
	c1.image(image_resized)
	
	image_resized = np.array(image_resized) # as numpy array
	
	# use image_generator to transform the image_transformed
	x = np.expand_dims(image_resized, axis=0)
	img_transformed = image_generator.flow(x)
	
	c1.header('Input Image')
	#c1.write(y.shape)
	
	#load the model
	model_vgg = load_model('classifier_vgg_model.h5')
	
	# Evaluate on test_generator
	eval_vgg = model_vgg.predict(img_transformed)
	# transforms smt like [0.33,0.67] into [0,1]
	pred_vgg = (eval_vgg > 0.5).astype("int32")
	
	class_names = {0: 'Hot Dog', 1: 'Not Hot Dog'}
	# meaning: [0,1]: not_hot_dog
	#          [1,0]: hot_dog
	
	ind = pred_vgg[:,1][0]
	prediction = class_names[ind]
	
	c2.header('Output')
	c2.subheader('What you have is :')
	c2.write(prediction )
