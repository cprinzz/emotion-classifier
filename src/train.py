from keras.models import Sequential, model_from_json
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
import json
from keras import backend as K
K.set_image_dim_ordering('th')

with open('../experiments/models/model_7.json') as f:
    model = model_from_json(f.read())
model.load_weights('../experiments/weights/weights_7.h5')

X_train_fname = '../data/X_train_192.npy'
Y_train_fname = '../data/Y_train_192.npy'
X_train = np.load(X_train_fname)
Y_train = np.load(Y_train_fname)

adam = optimizers.Adam(lr=.0005)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())

hist = model.fit(X_train, Y_train, verbose=1, shuffle=True, epochs=15,
                 validation_split=.2)
