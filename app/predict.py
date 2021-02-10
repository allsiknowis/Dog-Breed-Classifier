import cv2
import argparse
import numpy as np
from glob import glob
from extract_bottleneck_features import *
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input

def arg_parser():
    '''
    Description: Sets arguments for running this python file

    Input:  none

    Ouput:  args - the argument for an image filepath, which also indicates how to point to an image file
    '''
    # create a parser object
    parser = argparse.ArgumentParser(description='predict.py')
    # add an argument to the parser for supplying the predict.py app with an image filepath (an image of a Brittany is supplied by default)
    parser.add_argument('--image', help='Point to image file, for example: python predict.py --image images\Brittany_02625.jpg', default='./images/Brittany_02625.jpg')
    # assign the parser arguments to an args object
    args = parser.parse_args()

    return args


def load_json_model():
    '''
    Description: Loads the saved model that was trained when running train_classifier.py. Also sets the weights for the model that were saved when training the model in train_classifier.py

    Input:  none

    Output: loaded_model - the loaded model
    '''
    # open json file containing the saved model
    json_file = open('./saved_models/Resnet50_model.json', 'r')
    # load json file
    loaded_model_json = json_file.read()
    # close the json file
    json_file.close()
    # create the model using the loaded json file
    loaded_model = model_from_json(loaded_model_json)
    # load saved best weights into the model
    loaded_model.load_weights('./saved_models/weights.best.Resnet50.hdf5')

    return loaded_model


def path_to_tensor(img_path):
    '''
    Description: Converts an RGB image to a 3D tensor of shape (224, 224, 3) and then to a 4D tensor with shape (1, 224, 224, 3)

    Input:  img_path - a string indicating the filepath of the image

    Output: the 4D tensor of the provided image
    '''
	# load the image
    img = image.load_img(img_path, target_size=(224, 224))
    # convert the image to a 3D tensor
    tensor_3D = image.img_to_array(img)
    # convert the 3D tensor to a 4D tensor
    tensor_4D = np.expand_dims(tensor_3D, axis=0)

    return tensor_4D


def face_detector(img_path):
    '''
    Decription: Converts a provided image to grayscale, detects whether the image contains a face, and then returns True if a face is detected.

    Input:  img_path - a string indicating the filepath of the image

    Ouput:  result - True or False for whether a face was found
    '''
    # load face classifier
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    # read in image located in img_path
    img = cv2.imread(img_path)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect presence of a face
    faces = face_cascade.detectMultiScale(gray)
    # get True or False result of whether a face was detected
    face_found = len(faces) > 0
    # if a face was found, add result to results list
    if face_found:
        results.append("That's a human!")

    return face_found


def dog_detector(img_path):
    '''
    Description: Returns True if a dog is detected in the image stored at img_path

    Input:  img_path - a string indicating the filepath of the image

    Output: True or False
    '''
    # get the predicted label to determine if the image contains a dog
    prediction = predict_labels(img_path)

    return ((prediction <= 268) & (prediction >= 151))


def predict_labels(img_path):
    '''
    Description: Preprocesses an image located in img_path so that it can be sent to the model which returns the prediction vector for the image

    Input:  img_path - a string indicating the filepath of the image

    Output: pred - an integer corresponding to the model's predicted object class
    '''
    # loads the ResNet-50 model using imagenet weights
    model_ResNet50 = ResNet50(weights='imagenet', include_top=False, pooling="avg")
    # preprocess the 4D image tensor
    img = preprocess_input(path_to_tensor(img_path))
    # get the index for the label prediction
    pred = np.argmax(model_ResNet50.predict(img))

    return pred


def predict_breed(img_path):
    '''
    Description: Extracts bottleneck features using the supplied image filepath and obtains a predicted vector for the bottleneck feature. Loads the dog breed names and matches the predicted vector to its breed.

    Input:  img_path - a string indicating the filepath of the image

    Output: breed - the dog breed that matches the prediction made for the supplied image in img_path
    '''
    # load the saved model trained in train_classifier.py
    my_model = load_json_model()

    # load ResNet-50 model with imagenet weights
    model_ResNet50 = ResNet50(weights='imagenet', include_top=False)

    # extract bottleneck feature
    bottleneck_feature = model_ResNet50.predict(preprocess_input(path_to_tensor(img_path)))

    # make a prediction on bottleneck_feature
    pred = my_model.predict(bottleneck_feature)
    # get the index for the prediction
    arg_max = np.argmax(pred)

    # load dog breed names
    dog_names = [item[20:-1] for item in sorted(glob("./data/dog_images/train/*/"))]

    # retrieve the dog breed name using the prediction
    breed = dog_names[arg_max]
    # format the dog breed name
    breed = breed.split(".", 1)[1].replace("_"," ").title()

    return breed


def load_image(img_path):
    '''
    Description: Loads an image using a provided image path and converts the imageto RGB

    Input:  img_path - a string indicating the filepath of the image

    Output: cv_rgb - the RGB image
    '''
    # load the image
    image = cv2.imread(img_path)
    # convert BGR image to RGB
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return cv_rgb


def run_app(img_path):
    '''
    Description: Checks a provided image for the presence of a human face using the face_detector function. If a human face is detected, it is added to a results list along with with a predicted dog breed. If a human face is not detected, the presence of a dog is checked using the dog_detector function. If a dog is detected in the image, it is added to a results list along with the predicted dog breed. If neither a human face nor dog is detected, an error is displayed.

    Input: img_path - a string indicating the filepath of the image

    Ouput: none
    '''
    # create a tensor of vowels for making result grammatically correct
    vowels = ("A", "E", "I", "O", "U")

    # if a human face is detected, predict a dog breed for it
    if face_detector(img_path):
        '''
        Description: Checks whether a picture file located in img_path contains a human face. If it does, the image is used to predict the dog breed. Results are added to a list for printing when the app has finished running.

        Input:  img_path - a string indicating the filepath of the image

        Output: none
        '''
        # predict dog breed for human
        predicted_breed = predict_breed(img_path)
        # add grammatically correct breed to results list
        if predicted_breed.startswith(vowels):
            results.append("But they look like an {}.\n".format(predicted_breed))
        else :
            results.append("But they look like a {}.\n".format(predicted_breed))


    # if a dog is detected, predict its breed
    elif dog_detector(img_path):
        '''
        Description: Checks whether a picture file located in img_path contains a dog. If it does, the image is used to predict the dog breed. Results are added to a list for printing when the app has finished running.

        Input:  img_path - a string indicating the filepath of the image

        Output: none
        '''
        # if the image contains a dog, add it to results list
        results.append("That's a dog!")
        # predict breed
        predicted_breed = predict_breed(img_path)
        # add grammatically correct breed to results list
        if predicted_breed.startswith(vowels):
            results.append("And it looks like an {}.\n".format(predicted_breed))
        else :
            results.append("And it looks like a {}.\n".format(predicted_breed))

    else:
        # if no human face or dog is detected in image, display error
        print ("\nSorry, neither a human nor a dog was found in that file :(\nTry another.")


def main():
    '''
    Description: Runs this predict.py app to determine whether an image file contains a human or a dog and to predict a dog breed for the image either way. Results are printed at the end.

    Input:  none

    Output: none
    '''
    # create a global list to hold the results
    global results
    results = []
    # get the parser arguments
    args = arg_parser()
    # run the app using the image filepath in the parser
    run_app(args.image)
    # print a blank line to make the results look nicer
    print()
    # print the results
    for res in results:
        print(res)

if __name__ == '__main__':
    main()
