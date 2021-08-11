# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : File to load the saved multi-class model, predict class of single
#               image and print the result. Used for evaluation of the model
# How to use  : Replace variables in CAPS according to needs of the test set
# =============================================================================

import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

#to display the decimals normally
np.set_printoptions(suppress=True)

path_to_image='PATH TO IMAGE'
model_path='PATH TO SAVED MODEL'
IMG_SIZE = 150

#function to prepare the image in order to be given to the model
def prepare(filepath):
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    norm_array=new_array/255.0
    # return the image with shaping that TF wants.
    return norm_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


#loads the saved model from the path specified
model=tf.keras.models.load_model(model_path+"Filters_32,32,64,128,128,128-Dense_64")

#to display the raw image
image = Image.open(path_to_image)
image.show()

#prepares image via function call and supplies it to the model
prediction = model.predict([prepare(path_to_image)])  #passing the list of the image

print("Multi-Class Classification Model Prediction:")
print(prediction)
