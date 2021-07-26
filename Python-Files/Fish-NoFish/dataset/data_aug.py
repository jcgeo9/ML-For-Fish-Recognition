# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : File in order to extend the dataset by creating additional
#               images. To be used only if images gathered for the dataset are
#               not enough for performing model training.
# How to use  : Replace variables in CAPS according to needs of the dataset   
# =============================================================================

from tensorflow.keras.preprocessing import image
import numpy as np
import os

dir_files='PATH TO DATASET DIRECTORY'
number_specified_images='NUMBER OF SPECIFIED IMAGES'
folder_to_store_new='NAME OF FOLDER TO BE CREATED TO STORE THE IMAGES PRODUCED'

#creates a folder where the produced images will be stored
os.mkdir(folder_to_store_new+'/')

#loops through image files in the dataset directory, converting each file to
#array, producing number of specified images for each array based on criteria
#given as a variable named 'train_datagen'
for filename in os.listdir(dir_files):

    im=image.load_img(dir_files+'/'+filename)
    im=image.img_to_array(im)
    im=np.expand_dims(im,axis=0)

    train_datagen = image.ImageDataGenerator(
          rotation_range=30,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          vertical_flip=True,
          fill_mode='nearest')

    imageGen = train_datagen.flow(im, batch_size=1, save_to_dir=folder_to_store_new+"/",
    	save_prefix="image", save_format="jpg")

    total=0
    for g in imageGen:
    	total += 1
    	print(counter)
    	if total == number_specified_images:
    		break
