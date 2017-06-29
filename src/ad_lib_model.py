from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import layers
from keras import optimizers
from keras import initializers
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')

def model_architecture(input_shape, conv_layers, dense_layers, dropout=.5):
    """
    Params:
    input_shape - tuple (color channels, x, y)
    conv_layers - list of dicts {depth:int, filter:(x,y), activation:'activation'}
    dense_layers - list of dicts {units:int, activation:'activation'}
    dropout - float
    nb_epoch - int

    Returns:
    Compiled model
    """
    model = Sequential()

    #instantiate initializer with random seed
    glorot = initializers.glorot_uniform(seed=42)
    #Add conv layers
    for i in range(conv_layers[0]['num']):
        if i == 0:
            model.add(Conv2D(conv_layers[0]['depth'], conv_layers[0]['filter'],
                             padding='same', activation=conv_layers[0]['activation'],
                             kernel_initializer=glorot, input_shape=input_shape))
        else:
            model.add(Conv2D(conv_layers[0]['depth'], conv_layers[0]['filter'],
                             padding='same', activation=conv_layers[0]['activation'],
                             kernel_initializer=glorot))
    model.add(MaxPooling2D(2,2))

    #remove first dict from list
    conv_layers.pop(0)

    for layer in conv_layers:
        for num in range(layer['num']):
            model.add(Conv2D(layer['depth'], layer['filter'],
                      padding='same', activation=layer['activation'],
                      kernel_initializer=glorot))
        model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    #Add dense layers
    for layer in dense_layers:
        model.add(Dense(layer['units'], activation=layer['activation']))
        if dropout and layer['activation'] != 'softmax':
            model.add(Dropout(dropout))
    #adam = optimizers.Adam(lr=.0003)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    print(model.summary())
    return model
