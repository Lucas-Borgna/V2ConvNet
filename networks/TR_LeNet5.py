from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ThresholdedReLU



nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

def GetNetArchitecture(input_shape, weights_path=None):
    model = Sequential()
    #model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid',input_shape=input_shape))
    model.add(Convolution2D(nb_filters, (3,3), border_mode='valid',input_shape=input_shape, activation = 'linear'))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(ThresholdedReLU(theta = 1.0))
    model.add(Convolution2D(nb_filters, (3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(232))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
