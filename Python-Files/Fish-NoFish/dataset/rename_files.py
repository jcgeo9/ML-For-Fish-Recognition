# =============================================================================
# Created By  : Giannis Kostas Georgiou
# Project     : Machine Learning for Fish Recognition (Individual Project)
# =============================================================================
# Description : File in order to rename the files of the dataset
#               to the needs of the project. To be used after data_aug.py.
# How to use  : Replace variables in CAPS according to needs of the dataset
# =============================================================================

import os

path ='PATH TO DATASET DIRECTORY'
file_name='NEW NAME OF FILES'

#gets the files from the specified path
files = os.listdir(path)

#renames each file from gathered files to 'file_name+index of the file' and stores the images as .jpg
for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join([file_name+'-'+str(index), '.jpg'])))
