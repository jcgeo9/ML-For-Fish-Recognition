# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : Use in order to train a single model and save it to the specified
#               path. To be used after finding_best_model.py in order to save
#               model instances of the best models.
# How to use  : Replace variables in CAPS according to needs of the dataset
# =============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle


path_to_training_set='PATH TO X AND y .pickle FILES'
save_model_dir='PATH TO THE DIRECTORY WHERE MODEL WILL BE SAVED'
save_model_name='NAME OF THE MODEL'
number_of_epochs='SPECIFY NUMBER OF EPOCHS FOR THE MODELS'

#insert arrays of dataset
#features
X=pickle.load(open(path_to_training_set+"/X_species.pickle","rb"))
#class
y=pickle.load(open(path_to_training_set+"/y_species.pickle","rb"))

#normalizes the data
X=X/255.0

#initialise model
model=Sequential()

#change first parameter of Conv2D to suit filters of model's needs
#uncomment or add convolutional layers based on the model's structure
#the Input layer must always be included

#adds the input layer which takes the features(X)
model.add(Conv2D(32,(3,3),input_shape=X.shape[1:],padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

#adds the next convolutional layer(s) with filters
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
model.add(BatchNormalization())

#adds the next convolutional layer(s) with filters
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
model.add(BatchNormalization())

#adds the next convolutional layer(s) with filters
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
model.add(BatchNormalization())

#adds the next convolutional layer(s) with filters
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
model.add(BatchNormalization())

#adds the next convolutional layer(s) with filters
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2, 2)))
model.add(BatchNormalization())

# =============================================================================

#change parameter of Dense to suit number of nodes of the model's needs
#uncomment, comment or add dense layers based on the model's structure
#the Flatten and Output layer must always be included

#flattens the data for input to the dense layer(s)
model.add(Flatten())

#adds the dense layer(s)
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#adds the output layer
model.add(Dense(20))
model.add(Activation('softmax'))

# =============================================================================

#trains model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=number_of_epochs, validation_split=0.1765)

model.save(save_model_dir+'/'+save_model_name)
