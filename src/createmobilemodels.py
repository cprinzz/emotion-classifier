import coremltools
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import os
import os.path as osp

def create_coreml_model(model_fname, weights_fname):
    coreml_model = coremltools.converters.keras.convert((model_fname, weights_fname))
    coreml_model.save('../mobile_models/emotion_detector.mlmodel')

def create_tensorflow_model(model_fname, weights_fname):
    with open('../experiments/models/model_5.json') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_fname)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    graph = K.get_session().graph
    tf.train.write_graph(graph, '../mobile_models', 'emotion_detector.pb', as_text = False)



model = '../experiments/models/model_5.json'
weights = '../experiments/weights/weights_5.h5'

create_tensorflow_model(model, weights)
