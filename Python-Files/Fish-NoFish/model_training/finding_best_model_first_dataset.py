# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : Use in order to train multiple models and store logs for each
#               one in a folder. The logs will be used for comparing the
#               accuracy and loss of models using Tensorboard.
#               When finished executing, it should produce 819 model logs
#               as the combinations of up to 3 convolutional layers with up to
#               3 dense layers are 819
#               To be used when dataset is completely ready and renamed
# How to use  : Replace variables in CAPS according to needs of the dataset
# =============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle

filters = [32,64,128]
denses = [32,64,128,256]
count = 0

path_to_training_set='PATH TO X AND y .pickle FILES'
save_log_dir='PATH TO THE DIRECTORY WHERE MODELS LOGS WILL BE STORED'
number_of_epochs='SPECIFY NUMBER OF EPOCHS FOR THE MODELS'

#insert arrays of dataset
#features
X=pickle.load(open(path_to_training_set+"/X.pickle","rb"))
#class
y=pickle.load(open(path_to_training_set+"/y.pickle","rb"))

#normalizes the data
X=X/255.0

#function that creates model name based on number of filters and dense layers given
#then calls train model in order to train the model
def save_and_train_model_data(filters,dense):
    global count
    count+=1
    model_name=str(count)+".Filters_"+",".join(str(x) for x in filters)+"-Dense_"+",".join(str(x) for x in dense)
    print(model_name)
    train_model(model_name,filters,dense)


#uses info(filters and dense layers) passed from 'save_and_train_model_data' function to train model
def train_model(model_name,filters,dense):
    #creates tensorboard callback for the model
    tensorboard=TensorBoard(log_dir=save_log_dir+"/"+model_name)
                            
    #initialise model
    model=Sequential()
    #adds the input layer which takes the features(X)
    model.add(Conv2D(filters[0],(3,3),input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #adds the next convolutional layer(s) with filters
    if len(filters)>1:
        model.add(Conv2D(filters[1],(3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
    if len(filters)>2:
        model.add(Conv2D(filters[2],(3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

    #flattens the data for input to the dense layer(s)
    model.add(Flatten())

    #adds the dense layer(s)
    if len(dense)>0:
        model.add(Dense(dense[0]))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
    if len(dense)>1:
        model.add(Dense(dense[1]))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

    #adds the output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #trains and stores models log
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=number_of_epochs, validation_split=0.1765, callbacks=[tensorboard])


#loops through number 1 to 3 which corresponds to number of convolutional layers
#if layer=1 then it uses only one loop in filters and a nested loop for dense
#layers between 0 and 2 inclusively.
#increases the number of nested for loops for filters when number of convolutional
#layers is increased
for layer in range(1,4):
     if layer == 1:
         for filter in filters:
             for dense_layer in range(0,3):
                 if dense_layer == 0:
                     save_and_train_model_data(filters=[filter],dense=[])
                 if dense_layer == 1:
                     for dense in denses:
                         save_and_train_model_data(filters=[filter],dense=[dense])
                 if dense_layer == 2:
                     for dense in denses:
                         for dense2 in denses:
                             save_and_train_model_data(filters=[filter],dense=[dense,dense2])

     if layer == 2:
         for filter in filters:
             for filter2 in filters:
                 for dense_layer in range(0,3):
                     if dense_layer == 0:
                         save_and_train_model_data(filters=[filter,filter2],dense=[])
                     if dense_layer == 1:
                         for dense in denses:
                             save_and_train_model_data(filters=[filter,filter2],dense=[dense])
                     if dense_layer == 2:
                         for dense in denses:
                             for dense2 in denses:
                                 save_and_train_model_data(filters=[filter,filter2],dense=[dense,dense2])

    if layer == 3:
        for filter in filters:
            for filter2 in filters:
                for filter3 in filters:
                    for dense_layer in range(0,3):
                        if dense_layer == 0:
                            save_and_train_model_data(filters=[filter,filter2,filter3],dense=[])
                        if dense_layer == 1:
                            for dense in denses:
                                save_and_train_model_data(filters=[filter,filter2,filter3],dense=[dense])
                        if dense_layer == 2:
                            for dense in denses:
                                for dense2 in denses:
                                    save_and_train_model_data(filters=[filter,filter2,filter3],dense=[dense,dense2])
