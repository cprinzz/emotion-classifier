from keras.models import model_from_json
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')

model_path = '../experiments/models/model.json'
weights_path = '../experiments/weights/weights.h5'

X_test_fname = '../data/X_test_scaled_jaffe_ck_dlib.npy'
Y_test_fname = '../data/Y_test_scaled_jaffe_ck_dlib.npy'
X_test = np.load(X_train_fname)
Y_test = np.load(Y_train_fname)
Y_test = np.argmax()

fep = FacialEmotionPredictor(model_path, weights_path)

predictions = []
for i in range(5):
    if fep.predict(X_test[i])[1] == Y_test[i]:
        print('sick bro')
    else:
        print('nah bro')
