from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

def create_model():
    model_resnet = Sequential()
    
    # add the required layers
    model_resnet.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    model_resnet.add(Dense(256, activation='relu'))
    model_resnet.add(Dropout(0.5))
    model_resnet.add(Dense(133, activation='softmax'))

    # lood the weights with the lowest val_loss that were saved
    model_resnet.load_weights('./app/saved_models/weights.best.resnet.hdf5')

    return model_resnet