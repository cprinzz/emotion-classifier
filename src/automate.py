from ad_lib_model import model_architecture
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
import json
import os
from keras import backend as K
K.set_image_dim_ordering('th')

def update_results(arch_dict, acc, val_acc, loss, val_loss):
    """
    Appends training results to results.csv
    """

    e_dict = {'model_id':[arch_dict['model_id']],
              'X_train_fname':[arch_dict['X_train_fname']],
              'Y_train_fname':[arch_dict['Y_train_fname']],
              'dropout':[arch_dict['dropout']],
              'epochs':[arch_dict['epochs']],
              'acc':[acc],
              'val_acc':[val_acc],
              'loss':[loss],
              'val_loss':[val_loss]
             }
    results = pd.read_csv('../experiments/results.csv')[list(e_dict.keys())]
    updated = pd.concat([results, DataFrame.from_dict(e_dict)], ignore_index=True)

    updated.to_csv('../experiments/results.csv')


def log_model(arch_dict, model):
    """
    Saves models and weights in model_{model_id}.json format
    """
    model_id = arch_dict['model_id']
    model_json = model.to_json()
    with open('../experiments/models/model_{}.json'.format(model_id), 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('../experiments/weights/weights_{}.h5'.format(model_id))
    print('Saved model to disk')


def train_models(arch_dict):
    #load training data
    X_train_fname = arch_dict['X_train_fname']
    Y_train_fname = arch_dict['Y_train_fname']
    X_train = np.load(X_train_fname)
    Y_train = np.load(Y_train_fname)

    conv_layers = arch_dict['conv_layers']

    dense_layers = arch_dict['dense_layers']

    dropout = arch_dict['dropout']

    epochs = arch_dict['epochs']

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model = model_architecture(input_shape=input_shape, conv_layers=conv_layers,
                               dense_layers=dense_layers, dropout=dropout)

    #train model
    hist = model.fit(X_train, Y_train, verbose=1, shuffle=True, epochs=epochs,
                     validation_split=.2)
    training_metrics = hist.history
    acc = training_metrics['acc']
    val_acc = training_metrics['val_acc']
    loss = training_metrics['loss']
    val_loss = training_metrics['val_loss']

    #log model, weights, and results
    log_model(arch_dict, model)
    update_results(arch_dict, acc=acc, val_acc=val_acc,
                   loss=loss, val_loss=val_loss)

def continue_training(arch_dict, model_id, epochs):
    #load training data
    X_train_fname = arch_dict['X_train_fname']
    Y_train_fname = arch_dict['Y_train_fname']
    X_train = np.load(X_train_fname)
    Y_train = np.load(Y_train_fname)

    #find model and weights files
    model_file = ''
    weights_file = ''
    for (dirpath, dirnames, filenames) in os.walk('../', topdown=False):
        if dirpath.split('\\')[-1] == 'models':
            for f in filenames:
                if f.find(str(model_id)) != -1:
                    model_file = os.path.join(dirpath, f)
        if dirpath.split('\\')[-1] == 'weights':
            for f in filenames:
                if f.find(str(model_id)) != -1:
                    weights_file = os.path.join(dirpath, f)

    #Load model from file and compile
    with open(model_file) as f:
        model = model_from_json(f.read())

    model.load_weights(weights_file)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, verbose=1, shuffle=True, epochs=epochs,
                     validation_split=.2)

    training_metrics = hist.history
    acc = training_metrics['acc']
    val_acc = training_metrics['val_acc']
    loss = training_metrics['loss']
    val_loss = training_metrics['val_loss']

    #log model, weights, and results
    log_model(arch_dict, model)
    update_results(arch_dict, acc=acc, val_acc=val_acc,
                   loss=loss, val_loss=val_loss)


if __name__ == '__main__':
    with open('../experiments/experiments.json', 'r') as f:
        experiments = json.loads(f.read())
    #continue_training(experiments[5], 7, 15)
    train_models(experiments[10])
