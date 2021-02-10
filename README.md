# Dog_Breed_Predictor
[//]: # (Image References)

[image1]: /images/Brittany_02625.jpg "Default Image"

### Table of Contents
1. [Description](#description)

2. [Data](#data)

    i. [Dependencies](#dependencies)
        
    ii. [File Descriptions](#files)
        
3. [Instructions](#instructions)

4. [Acknowledgements](#acknowledgements)


### Description <a name="description"></a>

This project is part of the Udacity Data Scientist Nanodegree. Its purpose is to show how to build a pipeline that can be used within an app to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.

First, a classifier is trained using the train_classifier.py file in the app folder. This model is saved as a .json file to the saved_models folder along with an .hdf5 file containing the best weights determined during training. In order to successfully run this file, additional files from the full [Udacity project repository](https://github.com/udacity/dog-project/) are required.

Next, a prediction can be made utilizing this trained classifier by running the predict.py file found in the app folder. By default, a prediction is made on the default image shown below. Otherwise, a file path can be provided as an argument.

![Default Image][image1]

For the full Udacity project, navigate to the following github repository and follow the instructions in the README file:
```	
https://github.com/udacity/dog-project/blob/master/README.md
```
### Data <a name="data"></a>

#### Dependencies <a name="dependencies"></a>
#### train_classifier.py
* Python: version 3.5+
* File Processing: numpy, sklearn.datasets.load_files
* Keras Libraries: utils.np_utils, models.Sequential, layers.MaxPooling2D, layers.GlobalAveragePooling2D, layers.Dropout, layers.Desne, callbacks.ModelCheckpoint
* Additional Files: the full set of dog_images used for training, and DogResnet50Data.npz found in the full [Udacity project repository](https://github.com/udacity/dog-project/)

#### predict.py
* Python: version 3.5+
* Image Processing: cv2, numpy
* File Processing: glob
* Keras Libraries: models.model_from_json, preprocessing.image, applications.resnet50.ResNet50, applications.resnet50.preprocess_input
* Additional Files: haarcascade_frontalface_alt.xml found in the full [Udacity project repository](https://github.com/udacity/dog-project/)

#### File Descriptions <a name="files"></a>
`dog_app Notebook:` A full walkthrough of the dog breed classification pipeline.

`dog_app HTML:` An HTML file of the dog_app notebook.

`train_classifier.py:` A Python file used for training a classification model on dog images, saving the best weights found during fitting, and saving the model trained model to a .json file.

`predict.py:` A Python file used for predicting a dog breed for a supplied image file using the model trained in train_classifier.py

`images/Brittany_02625.jpg:` The default file used for prediction when running predict.py without supplying an image filepath as an argument.

`saved_models/Resnet50_model.json:` The trained model file produced when running train_classifier.py

`saved_models/weights.best.Resnet50.hdf5:` The weights file produced during model training when running train_classifier.py

### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
	`python train_classifier.py`

2. Run the following command in the project's root directory to predict a dog breed for an image.
	- To make a prediction using the default image file
		`python predict.py`
	- To make a prediction using your own image file
		example: `python predict.py --image images\Brittany_02625.jpg`
		
### Acknowledgements<a name="acknowledgements"></a>
* This program is part of [Udacity](https://www.udacity.com/)'s Data Scientist Nanodegree
