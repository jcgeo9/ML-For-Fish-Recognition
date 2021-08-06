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

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
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

save_roc_loc='PATH TO WHERE THE ROC SHOULD BE SAVED'
name_roc='NAME OF THE ROC'
name_roc_zoomed='NAME OF THE ZOOMED ROC'
data_dir='PATH TO TEST SET DIRECTORY'
model_path='PATH TO SAVED MODEL'

#insert arrays of test set
#features
X=pickle.load(open(data_dir+"X_combined_test_data.pickle","rb"))
#class
y=pickle.load(open(data_dir+"y_combined_test_data.pickle","rb"))

#normalize the test set
X=X/255.0

#loads the saved model from the path specified
model=tf.keras.models.load_model(model_path+"Binary_Filters_32,32,64,64,64-Dense_64_BEST")

#predict the classes of the images in the test set
y_pred = model.predict_classes(X)

#produce the roc curve based on y and predicted y
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred)

#calculate the auc of the graph
auc_keras = auc(fpr_keras, tpr_keras)

#plot and save the ROC curve with y limit 0-1 and x limit 0-1 (normal graph)
plt.figure(figsize=(12,12))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='(area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.savefig(save_roc_loc+name_roc, dpi=400)
plt.show()

#plot and save the ROC curve with y limit 0.9-1 and x limit 0-0.1 (zoomed in graph)
plt.figure(figsize=(12,12))
plt.xlim(0, 0.1)
plt.ylim(0.9, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
 plt.savefig(save_roc_loc+name_roc_zoomed, dpi=400)
plt.show()



