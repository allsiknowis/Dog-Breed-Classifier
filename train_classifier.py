import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense

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
    # load bottleneck features
    bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
    # create training set of bottleneck features
    train_Resnet50 = bottleneck_features['train']
    # create validation set of bottleneck features
    valid_Resnet50 = bottleneck_features['valid']
    print('Completed!\n')

    print('Defining architecture...')
    # create an instance of a Sequential model
    Resnet50_model = Sequential()
    # add GlobalAveragePooling2D layer to the model
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    # add a Dense layer to the model
    Resnet50_model.add(Dense(133, activation='softmax'))
    print('ResNet50 architecture defined.\n')

    # show a summary of the model
    Resnet50_model.summary()

    print('Compiling model...')
    # compile the model
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Compile successful!\n')

    # create a checkpointer object for saving the best weights during fitting
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', verbose=1, save_best_only=True)

    print('Loading targets...')
    # load the training dog image targets
    train_targets = load_dataset('./data/dog_images/train')
    # load the validation dog image targets
    valid_targets = load_dataset('./data/dog_images/valid')
    print('Completed!\n')

    print('Please wait while the model is trained...\n')
    # train the model using training and validation data
    Resnet50_model.fit(train_Resnet50, train_targets,
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    print('Model trained successfully!')

    # serialize model to .json for use in predict.py file
    print('Saving model...')
    model_json = Resnet50_model.to_json()
    with open('./saved_models/Resnet50_model.json', 'w') as json_file:
        json_file.write(model_json)
    print('Model saved successfully!\n')


def main():
    train_model()

if __name__ == '__main__':
    main()
