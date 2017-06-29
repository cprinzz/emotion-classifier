import numpy as np
from keras.models import model_from_json

'''
Attributes:
model - compiled keras model
'''
class FacialEmotionPredictor(object):

    def __init__(self, model_path, weights_path):
        '''
        Params:
        model_path - path to json model
        weights_path - path to h5 weights file
        '''
        with open(model_path) as f:
            self.model = model_from_json(f.read())

        self.model.load_weights(weights_path)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def predict(image):
        '''
        Params:
        image - np array dim (1, 192, 192)

        Returns:
        tuple (dim (6,) array, int) where the array is composed of the probabilities
        for each emotion and the int is the index of the max probability

        emotion_map = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
        '''

        probabilities = self.model.predict(image)
        prediction = np.argmax(probabilities)

        return (probabilities, prediction)
