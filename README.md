# ML-For-Fish-Recognition
Created for my Master's Thesis. Contains datasets, python code and Android Studio code.

Name: Giannis Kostas Georgiou

Student ID: 20090232

Supervisor: Peter McBurney

## Repository Contents

This repository is divided into two directories:
* Dataset Preparation, Model Training and Model Testing Python Code
* Android App Code and APK File

## Datasets

The datasets used in this repository and their corresponding kaggle repositories are:
* Fish-No Fish
  * https://www.kaggle.com/giannisgeorgiou/fish-or-no-fish-simple-images
* Species
  * https://www.kaggle.com/giannisgeorgiou/fish-species
* Fish-No Fish with Species Images
  * https://www.kaggle.com/giannisgeorgiou/fish-or-no-fish-species-images
* Fish-No Fish with Combined Images
  * https://www.kaggle.com/giannisgeorgiou/fish-or-no-fish-simple-and-species-images

## Instructions on Python Files

Instructions on how to use the python files are provided inside each file as a comment section on top but in general one should:
1. Download the dataset of his choice (Can be either one of the above or another binary or multi-class dataset)
2. Download the directory corresponding to the downloaded dataset
3. Use the "/dataset" directory to convert and store the dataset
4. Use the "/model_training" directory to train models and find the most suitable
5. Use the "/model_testing" directory to test trained models of his choice

If one wishes to avoid training models and wants to obtain the trained ones they are located in "/trained_models" directory and can be used after the following:
1. Download the model directory
2. Loading the model using the following command
>#loads the saved model from the path specified
>
>model=tf.keras.models.load_model(model_path)

## Instructions on Android App Files

If one wishes to download and use the app:
1. Download the .apk file in the directory
2. If downloaded from an android device, install it and it is ready to use.
3. If downloaded from a PC, send the .apk file to an android device, install it and it is ready to use

If one wishes to make changes to the app:
1. Download the "/fish_recognition" directory
2. Open it via Android Studio
3. Start editing

If one wishes to use a new model in the app:
1. Convert the model to .tflite instance 
2. Upload .tflite model to assets directory
3. Change files according to the .tflite model name


**If one chooses to use a model which is not referenced here, training models with these files may not produce a good result. Model accuracy and loss depends on the domain and the dataset, thus changing the dataset but not the model architecture will have a different result**
