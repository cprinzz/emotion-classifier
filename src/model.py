from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')

X_train_fname = '../data/X_train.npy'
Y_train_fname = '../data/Y_train.npy'
X_train = np.load(X_train_fname)
Y_train = np.load(Y_train_fname)
X_cv_fname = '../data/X_cv.npy'
Y_cv_fname = '../data/Y_cv.npy'
X_cv = np.load(X_cv_fname)
Y_cv = np.load(Y_cv_fname)

# datagen = ImageDataGenerator(rescale=1./255,
#                              rotation_range=10,
#                              shear_range=0.2,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              horizontal_flip=True)
#
# datagen.fit(X_train)

print('Building model...')
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(1, 96, 96)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(.25))

# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# #model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, verbose=1, shuffle=True)
# model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
#                     steps_per_epoch=len(X_train)/32, epochs=320)

# serialize model to JSON
model_json = model.to_json()
with open("model_3.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("weights_3.h5")
print("Saved model to disk")
