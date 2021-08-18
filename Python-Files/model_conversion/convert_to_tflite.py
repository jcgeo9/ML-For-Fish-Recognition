# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : File in order to convert saved models to .tflite instances.
#               To be used after the desired model are trained and saved
# How to use  : Replace variables in CAPS according to needs of the dataset   
# =============================================================================

import tensorflow as tf

model_path='PATH TO SAVED MODEL'
tflite_model_name='NAME OF THE NEWLY CREATED TFLITE MODEL'

#convert the model by loading the saved model to the converter
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()

#save the tflite model
with open(tflite_model_name+'.tflite', 'wb') as f:
  f.write(tflite_model)
