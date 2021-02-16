# Dog_Breed_Predictor
[//]: # (Image References)

[image1]: /images/Brittany_02625.jpg "Default Image"

### Table of Contents
1. [Motivation](#motivation)

2. [Data](#data)

    i. [Dependencies](#dependencies)
        
    ii. [File Descriptions](#files)
        
3. [Instructions](#instructions)

4.[Summary of Results](#summary)

5. [Acknowledgements](#acknowledgements)


### Motivation <a name="motivation"></a>

This project comes from the Udacity Data Scientist Nanodegree program which utilizes convolutional neural networks (CNN) to classify images of dogs according to their breed. When an image file path is provided to the algorithm, the image goes through several steps. First, it is checked for the presence of a human face. If one is detected, the algorithm will predict a dog breed for it. However, if no face is found, the image is then checked for the presence of a dog and, if detected, will predict the breed it most resembles.

First, a classifier is trained via Transfer Learning using the *train_classifier.py* file. Bottleneck features are loaded before the model is compiled and trained on them. This model is saved as a *.json* file to the *saved_models* folder along with an *.hdf5* file containing the best weights determined during training. In order to successfully run this file, additional files from the full [Udacity project repository](https://github.com/udacity/dog-project/) are required.

Second, a web server containing the prediction pipeline is started by the user from the command line by running the *run_server.py* file. Once an image path is sent to the server via the *predict.py* file, the image is opened and read so that it can be checked for the presence of a human face by a face classifier. If a human face is detected, that indication is added to a results list for later printing. The image path is then used to predict a dog breed for the human and the predicted breed is added to a results list. If no human face has been detected, however, the image is checked for the presense of a dog. If a dog is detected in the image, that indication is also added to a results list. (If no dog is detected, an error is displayed.) The image is then converted to a 3D tensor and then to a 4D tensor to be used as input to a classifier. A pre-trained ResNet-50 model is then loaded and used to predict a label classification for the image. Subsequently, that label is used to predict a dog breed using Transfer Learning, bottleneck features, and the pretrained model from the step above. The predicted breed is added to the results list for printing.

Finally, the prediction is output to the screen. By default, a prediction is made on the default image shown below. Otherwise, a file path can be provided as an argument. Admittedly, this model isn't perfect but it does attain over 80% accuracy.

![Default Image][image1]

For the full Udacity project, navigate to the following github repository and follow the instructions in the README file:
```	
https://github.com/udacity/dog-project/blob/master/README.md
```
### Data <a name="data"></a>

#### Dependencies <a name="dependencies"></a>
  #### *train_classifier.py*
  * **Python:** version 3.5+
  * **File Processing:** numpy, sklearn.datasets.load_files
  * **Keras Libraries:** utils.np_utils, models.Sequential, layers.MaxPooling2D, layers.GlobalAveragePooling2D, layers.Dropout, layers.Dense, callbacks.ModelCheckpoint
  * **Additional Files:** the full set of dog images used for training, and *DogResnet50Data.npz* found in the full [Udacity project repository](https://github.com/udacity/dog-project/)

#### *pred_request.py*
  * **Python:** version 3.5+
  * **HTTP Request:** requests
  * **Argument Parser:** argparse
  * **Additional Files:** *Brittany_02625.jpg*, set as the default argument in the argument parser and is found in the full [Udacity project repository](https://github.com/udacity/dog-project/)

#### *run_server.py*
  * **Python:** version 3.5+
  * **Web App:** flask
  * **Image Processing:** cv2, numpy
  * **File Processing:** glob
  * **Keras Libraries:** models.model_from_json, preprocessing.image, applications.resnet50.ResNet50, applications.resnet50.preprocess_input
  * **Additional Files:** *haarcascade_frontalface_alt.xml* found in the full [Udacity project repository](https://github.com/udacity/dog-project/)


#### File Descriptions <a name="files"></a>
`dog_app.ipynb:` A notebook file containing the full walkthrough of the dog breed classification pipeline.

`report.html:` An HTML file of the *dog_app* notebook.

`train_classifier.py:` A Python file used for training a classification model on dog images, saving the best weights found during fitting, and saving the trained model to a .json file.

`run_server.py:` A Python file used for running a web server and predicting a dog breed for a supplied image file using the model trained in *train_classifier.py*

`pred_request:` A Python file used for sending an image to the web server in order to receive a prediction.

`images/Brittany_02625.jpg:` The default file used for prediction when running *pred_request.py* without supplying an image filepath as an argument.

`saved_models/Resnet50_model.json:` The trained model file produced when running *train_classifier.py*

`saved_models/weights.best.Resnet50.hdf5:` The weights file produced during model training when running *train_classifier.py*


### Instructions <a name="instructions"></a>
1. (Optional) Run the following commands in the project's root directory to train the model: 
	`python train_classifier.py`

2. Run the following command in the project's root directory to start the web server in order to make a prediciton.
	`python run_server.py`

3. Run the following command ***in a second command prompt*** in the project's root directory to predict a dog breed for an image.
	- To make a prediction using the default image file: 
		`python pred_request.py`
	- To make a prediction using your own image file: 
		**example** **-** `python pred_request.py --image images\Brittany_02625.jpg`
		

### Summary of Results<a name="summary"></a>
The process of creating a way to detect both humans and dogs and then make a prediction of dog breed is not exactly a short and simple one. We utilized a feature-based cascade classifier from OpenCV to detect human faces. This comes with the drawback that a user must pass in an image that provides a clear view of a human face for the face to be accurately detected. This could be circumvented by instead implementing a multi-task cascaded CNN (MTCNN) which can detect the presence of additional facial features. But for our purposes here, the feature-based classifier is sufficient.
We also created an improved CNN by utilizing transfer learning and bottleneck features from a pretrained model to reach over 80% accuracy. It is possible to achieve even better results by putting in the time to tune the model parameters even further, to add additional layers, and to perhaps train over a different number of epochs.
In the end, we produced a workable solution for using convolutional neural networks to detect whether an image contains a human or a dog and to predict a dog breed for it with reasonably acceptable results.

### Acknowledgements<a name="acknowledgements"></a>
* This program is part of [Udacity](https://www.udacity.com/)'s Data Scientist Nanodegree and all image files come from their [repository](https://github.com/udacity/dog-project/)
* A write-up of this project can be found on [Medium](https://steveellingson.medium.com/do-you-look-like-your-dog-6937b7f71c0f) 
