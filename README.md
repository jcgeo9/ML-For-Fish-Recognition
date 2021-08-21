# ML-For-Fish-Recognition
Created for my Master's Thesis.

Name: Giannis Kostas Georgiou

Student ID: 20090232

Supervisor: Dr Peter McBurney

The Android App files and info are located at:
* https://github.com/jcgeo9/ML-For-Fish-Recognition-App

## Repository Contents

This repository is divided into the following directories:
* Dataset Preparation, Model Training and Model Testing Python Code
* Saved Binary and Multi-Class Classification Models

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
1. Download the dataset of choice (Can be either one of the above or another binary or multi-class dataset)
2. Download the directory corresponding to the downloaded dataset
3. Use the "/dataset" directory to convert and store the dataset
4. Use the "/model_training" directory to train models and find the most suitable
5. Use the "/model_testing" directory to test trained models of his choice

If one wishes to avoid training models and wants to obtain the trained ones they are located in "/Saved-Models" directory and can be used after the following:
1. Download the model directory
2. Loading the model using the following command
>#loads the saved model from the path specified
>
>model=tf.keras.models.load_model(model_path)

**If one chooses to use a model which is not stored here, training models with these files may not produce a good result. Model accuracy and loss depends on the domain and the dataset, thus changing the dataset but not the model architecture will have a different result**
