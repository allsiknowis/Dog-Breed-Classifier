import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.callbacks import ModelCheckpoint


def load_dataset(path):
    '''
    Description: Loads datasets for train, test, and validation

    Input: path - the file containing dog images

    Output: dog_targets - an array of labels specifying the breed
                of each dog in data
    '''
    data = load_files(path)
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)

    return dog_targets


def train_model():
    '''
    Descriptions: Loads train and valid data, loads the defined model, compiles the model, creates a checkpointer to save the model that attains the best validation loss, then trains the model.

    Input: none

    Output: none
    '''
    print('Getting bottleneck features...')
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    train_Resnet50 = bottleneck_features['train']
    valid_Resnet50 = bottleneck_features['valid']
    print('Completed!\n')

    print('Defining architecture...')
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    Resnet50_model.add(Dense(133, activation='softmax'))
    print('ResNet50 architecture defined.\n')

    Resnet50_model.summary()

    print('Compiling model...')
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Compile successful!\n')

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', verbose=1, save_best_only=True)

    print('Loading targets...')
    train_targets = load_dataset('./data/dog_images/train')
    valid_targets = load_dataset('./data/dog_images/valid')
    print('Completed!\n')

    print('Please wait while the model is trained...\n')

    Resnet50_model.fit(train_Resnet50, train_targets,
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    print('Model trained successfully!')

    # serialize model to JSON
    print('Saving model...')
    model_json = Resnet50_model.to_json()
    with open('./saved_models/Resnet50_model.json', 'w') as json_file:
        json_file.write(model_json)

    # serialize model to YAML
    model_yaml = Resnet50_model.to_yaml()
    with open('saved_models/Resnet50_model.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    print("Saved model to disk")


def main():
    train_model()

if __name__ == '__main__':
    main()
