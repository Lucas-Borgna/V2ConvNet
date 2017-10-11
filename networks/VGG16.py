from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

from keras import initializers
import numpy as np

def GetNetArchitecture(input_shape):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    #model.add(Convolution2D(16, 3, 3, activation='linear',kernel_initializer = 'random_uniform', bias_initializer = 'ones'))
    #model.add(Convolution2D(16, 3, 3, init = 'normal', activation = 'linear'))
    model.add(Convolution2D(32, (3, 3), activation = 'linear'))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.3))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer = 'random_uniform', bias_initializer = 'ones'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    #print("Block One Output")
    #print(model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))

    #model.add(Convolution2D(16, 3, 3, activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'ones'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))

    #model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer = 'random_uniform', bias_initializer = 'ones'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    #print("Block Two Output")
    #print(model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    #print("Block Three Output")
    #print(model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    #print("Block Four Output")
    #print(model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    #print("Block Five Output")
    #print(model.output_shape)
    
    model.add(Flatten())

    #print("Flatten Output")
    #print(model.output_shape)


    # model.add(Dense(232, activation='relu'))
    # model.add(Dropout(0.5))

    # print("Dense One Output")
    # print(model.output_shape)

    model.add(Dense(232, activation='relu'))
    model.add(Dropout(0.5))

    #print("Dense Two Output")
    #print(model.output_shape)

    model.add(Dense(2, activation='softmax'))

    return model
