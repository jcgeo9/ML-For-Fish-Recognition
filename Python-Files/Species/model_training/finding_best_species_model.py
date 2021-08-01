# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : Use in order to train multiple models and store logs for each
#               one in a folder. The logs will be used for comparing the
#               accuracy and loss of models using Tensorboard.
#               When finished executing, it should produce 243 model logs
#               as that is the number of the combinations of up to 5
#               convolutional layers with 3 values to be filled with
#               To be used when dataset is completely ready and renamed
# How to use  : Replace variables in CAPS according to needs of the dataset
# =============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle
import numpy as np

filters = [32,64,128]
count = 0

#insert arrays of dataset
#features
X=pickle.load(open("drive/MyDrive/PROJECT_ML_BINARY_CLASSIFIER/X_species_150x150.pickle","rb"))
#class
y=pickle.load(open("drive/MyDrive/PROJECT_ML_BINARY_CLASSIFIER/y_species_150x150.pickle","rb"))

X=X.astype('float32')
X=X/255.0
X=np.round(X,4)

path_to_training_set='PATH TO X AND y .pickle FILES'
save_log_dir="PATH TO THE DIRECTORY WHERE MODELS LOGS WILL BE STORED"
number_of_epochs='SPECIFY NUMBER OF EPOCHS FOR THE MODELS'

#function that creates model name based on number of filters and dense layers
#given, then calls train model in order to train the model
def save_and_train_model_data(filters):
    global count
    count+=1
    model_name=str(count)+".Filters_"+",".join(str(x) for x in filters)
                    +"-Dense_64"
    train_model(model_name,filters)

#uses info(filters and dense layers) passed from 'save_and_train_model_data'
#function to train model
def train_model(model_name,filters):
    #creates tensorboard callback for the model
    tensorboard=TensorBoard(log_dir=save_log_dir+model_name)
               
    #initialise model
    model=Sequential()
    #adds the input layer which takes the features(X)
    model.add(Conv2D(filters[0],(3,3),input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
    model.add(BatchNormalization())
    #adds the next convolutional layer(s) with filters
    model.add(Conv2D(filters[1],(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
    model.add(BatchNormalization())
    #adds the next convolutional layer(s) with filters
    model.add(Conv2D(filters[2],(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
    model.add(BatchNormalization())
    #adds the next convolutional layer(s) with filters
    model.add(Conv2D(filters[3],(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
    model.add(BatchNormalization())
    #adds the next convolutional layer(s) with filters
    model.add(Conv2D(filters[4],(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
    model.add(BatchNormalization())

    #flattens the data for input to the dense layer(s)
    model.add(Flatten())

    #adds the dense layer(s)
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #adds the output layer
    model.add(Dense(20))
    model.add(Activation('softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=number_of_epochs, validation_split=0.1765,callbacks=[tensorboard])

#nested loops of filters array in order to make changes to the
#convolutional layer filters, exploring every combination of filters
for fil1 in filters:
  for fil2 in filters:
    for fil3 in filters:
      for fil4 in filters:
        for fil5 in filters:
          save_and_train_model_data(filters=[fil1,fil2,fil3,fil4,fil5])
