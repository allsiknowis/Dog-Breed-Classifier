import io
import cv2
import sys
import numpy as np
import argparse
import random
from glob import glob
from PIL import Image
from extract_bottleneck_features import *
from keras.applications import imagenet_utils
from keras.models import model_from_json, model_from_yaml
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def arg_parser():
    '''
    Description: Sets arguments for running this python file

    Input: none

    Ouput:  the argument indicating that an image file is required as an argument
    '''

    parser = argparse.ArgumentParser(description='predict.py')
    parser.add_argument('--image', help='Point to image file.', default='./dog_images/004_Akita/Akita_00244.jpg', type=str)

    args = parser.parse_args()

    return args


def load_json_model():
    # load json and create model
    json_file = open('./saved_models/Resnet50_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('./saved_models/weights.best.Resnet50.hdf5')
    print('Loaded model from disk')

    return loaded_model


def path_to_tensor(img_path):
    '''
    Description: Converts an RGB image to a 3D tensor and then to a 4D tensor
        with shape (1, 224, 224, 3)

    Input: img_path - the file path of the image to be converted to a tensor

    Output: the 4D tensor of the provided image
    '''
    print('Converting image to tensor...')
    #img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    #tensor_3D = image.img_to_array(img)

    #print('\nThe shape of the 3D tensor feature is {}\n'.format(tensor_3D.shape))

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    #tensor_4D = np.expand_dims(tensor_3D, axis=0)

    #print('\nThe shape of the 4D tensor feature is {}\n'.format(tensor_4D.shape))

    #print('Conversion complete!\n')
    #image = Image.open(io.BytesIO(img_path))

    # if the image mode is not RGB, convert it
    #if img.mode != "RGB":
    #    img = img.convert("RGB")

	# resize the input image and preprocess it
    img = image.load_img(img_path, target_size=(224, 224))
    tensor_3D = image.img_to_array(img)
    print('\nThe shape of the 3D tensor feature is {}\n'.format(tensor_3D.shape))
    tensor_4D = np.expand_dims(tensor_3D, axis=0)
    print('\nThe shape of the 4D tensor feature is {}\n'.format(tensor_4D.shape))
	#img = imagenet_utils.preprocess_input(img)

    print('Conversion complete!\n')

    return tensor_4D


def face_detector(img_path):
    '''
    Decription: Converts a provided image to grayscale, detects
        whether the image contains a face, and then returns
        True if a face is detected.

    Input: img_path - the file path for an image

    Ouput: result - True or False for whether a face was found
    '''
    print('Checking for a human face...')

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    result = len(faces) > 0

    if result:
        print('Face found!')
    else: print('No face found.\n')

    return result


def dog_detector(img_path):
    '''
    Description: Returns True if a dog is detected in the image stored at img_path

    Input: img_path - the file path of the image in which you want to detect
        a dog

    Output: True or False

    '''

    print('Detecting dog...\n')
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def ResNet50_predict_labels(img_path):
    '''
    Description: Preprocesses an image located at img_path so that it can be
        sent to the model which returns the prediction vector for the image

    Input:  img_path - the file path for the image whose object class you want to predict

    Output: pred - an integer corresponding to the model's predicted object class
    '''
    model_ResNet50 = ResNet50(weights='imagenet', include_top=False, pooling="avg")
    #Resnet50_model = load_yaml_model()

    print('Preprocessing the image...\n')
    img = preprocess_input(path_to_tensor(img_path))
    print('Preprocess complete!\n')

    print('Predicting label...')
    pred = np.argmax(model_ResNet50.predict(img))
    print('Prediction complete!')

    return pred


def Resnet50_predict_breed(img_path):
    '''
    Description: Extracts bottleneck features using the supplied image filepath and obtains a predicted vector for the bottleneck feature. Loads the dog breed names and matches the predicted vector to its breed.

    Input:  img_path - the filepath for the image you want to make a prediction for

    Output: The dog breed that matches the prediction made for the supplied image
    '''

    Resnet50_model = load_json_model()

    model_ResNet50 = ResNet50(weights='imagenet', include_top=False)

    print('Predicting breed...\n')
    # extract bottleneck features
    #bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    bottleneck_feature = model_ResNet50.predict(preprocess_input(path_to_tensor(img_path)))

    print('\nThe shape of the bottleneck feature is {}\n'.format(bottleneck_feature.shape))

    pred = Resnet50_model.predict(bottleneck_feature)
    print('\nThe shape of the pred is {}\n'.format(pred.shape))

    arg_max = np.argmax(pred)
    print('The arg_max is: {}\n'.format(arg_max))

    #bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)

    #print('\nThe shape of the bottleneck feature is now {}\n'.format(bottleneck_feature.shape))

    # obtain predicted vector
    #predicted_vector = Resnet50_model.predict(bottleneck_feature)

    # load dog breed names
    dog_names = [item[20:-1] for item in sorted(glob("./data/dog_images/train/*/"))]

    # return formatted dog breed that is predicted by the model
    #breed = dog_names[np.argmax(predicted_vector)]

    breed = dog_names[arg_max]
    print('Breed: {}\n'.format(breed))
    breed = breed.split(".", 1)[1].replace("_"," ").title()

    return breed


def load_image(img_path):
    '''
    Description: Loads an image using a provided image path, converts the image
        to RGB, then displays the image.

    Input: img_path - the file path of the image to be displayed

    Output: none
    '''
    # load the image
    image = cv2.imread(img_path)
    #img = Image.open(io.BytesIO(image))

    # convert BGR image to RGB
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return cv_rgb


def run_app(img_path):
    '''
    Description: Checks a provided image for the presence of a human face using the face_detector function. If a human face is detected, it is indicated along with a predicted dog breed. If a human face is not detected, the presence of a dog is checked using the dog_detector function. If a dog is detected in the image, it is indicated along with the predicted dog breed. If neither a human face or dog is detected, an error is displayed.

    Input: img_path - the file path of the image for which you want to make a
        prediction

    Ouput: none
    '''
    # create a tensor of vowels
    vowels = ("A", "E", "I", "O", "U")

    # if a human face is detected, predict a dog breed for it
    if face_detector(img_path):
        #load_image(img_path)
        print ("That's a human!")
        predicted_breed = Resnet50_predict_breed(img)
        # make sure the output is grammatically accurate
        if predicted_breed.startswith(vowels):
            print("But they look like an {}.\n".format(predicted_breed))
        else :print("But they look like a {}.\n".format(predicted_breed))

    # if a dog is detected, predict its breed
    elif dog_detector(img_path):
        #load_image(img_path)
        print ("That's a dog!")
        predicted_breed = Resnet50_predict_breed(img_path)
        # make sure the output is grammatically accurate
        if predicted_breed.startswith(vowels):
            print("And it looks like an {}.\n".format(predicted_breed))
        else :print("And it looks like a {}.\n".format(predicted_breed))

    else:
        # if no human face or dog is detected in image, display error
        print ("Sorry, that was an invalid image.")


def main():
    run_app('./images/Brittany_02625.jpg')
    #img_filepath ='./images/Brittany_02625.jpg'
    #run_app(img_filepath)
    #if len(sys.argv) == 1:
    #    print("What's the deal?")

    #    img_filepath = sys.argv[1:] # get filename of image
    #    print(img_filepath)

    #    run_app(img_filepath)  # run app
    #else:
    #    print('Please provide a filepath for the image you would like to classify as the first argument.\n\nExample: python predict.py ../images/Brittany_02625.jpg')

if __name__ == '__main__':
    main()
