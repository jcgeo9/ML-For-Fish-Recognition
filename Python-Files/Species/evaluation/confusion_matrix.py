# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : File to load the saved species model, predict class of images
#               of its test set and plot and save its confusion matrix
#               To be used after test set is converted to .pickle files and
#               model is trained and saved
# How to use  : Replace variables in CAPS according to needs of the test set
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import os
import cv2
import tensorflow as tf
import pickle
import sys

save_fig_loc='PATH TO WHERE THE FIGURE SHOULD BE SAVED'
name_fig='NAME OF THE FIGURE'
data_dir='PATH TO TEST SET DIRECTORY'
model_path='PATH TO SAVED MODEL'

#function to plot and save the confusion matrix based on class of images
#and predicted class
def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)

    sns.set(font_scale=1.6)
    svm=sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    figure = svm.get_figure()  
    figure.savefig(save_fig_loc+name_fig, dpi=400)


#insert arrays of test set
#features
X=pickle.load(open(data_dir+"X_species_test_data.pickle","rb"))
#class
y=pickle.load(open(data_dir+"y_species_test_data.pickle","rb"))

#normalize the test set
X=X.astype('float32')
X=X/255.0

#loads the saved model from the path specified
model=tf.keras.models.load_model(model_path+"Filters_32,32,64,128,128,128-Dense_64")

#predict the classes of the images in the test set
y_pred = np.argmax(model.predict(X), axis=-1)

#call the plot function to plot and save the confusion matrix
plot_cm(y,y_pred,(40,30))
