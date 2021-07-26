# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : File to convert training set into image arrays and store it to
#               a .pickle file for easier access when training models.
#               To be used after rename_files.py, and after dataset has been
#               separated to training and test sets.
# How to use  : Replace variables in CAPS according to needs of the dataset   
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

data_dir='PATH TO DATASET DIRECTORY'
categories=["Fish","Not_Fish"]
img_size=150

training_data=[]
X=[]
y=[]


#reads every image from categories Fish and Not_Fish, converts them to
#grayscale, resizes them to 150x150 and stores them to a list as arrays
#containing image data and image class
for cat in categories:
    path=os.path.join(data_dir,cat)
    class_num=categories.index(cat)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_img_array=cv2.resize(img_array,(img_size,img_size))
            training_data.append([new_img_array, class_num])
        except Exception as e:
            print(str(e))

#shuffles the elements of the list
random.shuffle(training_data)

#appends image features in X list and image class in y list
for features,label in training_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1, img_size,img_size, 1) #1 because it is grayscale
y=np.array(y)
print(X.shape)

#stores X and y lists in corresponding .pickle files
pickle_out=open("X_binary.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y_binary.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
