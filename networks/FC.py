from keras.models import Sequential
from keras.layers import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization



def GetNetArchitecture(input_shape):
   
    model = Sequential()
    

    model.add(Flatten(input_shape=input_shape))
    #model.add(Reshape((-1), input_shape=input_shape))
    model.add(Dense(625, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(625, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))


    return model
