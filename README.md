# emotion-classifier
Facial emotion classifier using Keras

## Datasets:
This model was trained using the Cohn-Kanade (http://www.pitt.edu/~emotion/ck-spread.htm), JAFFE (http://www.kasrl.org/jaffe.html), and FER-2013 (http://www-etud.iro.umontreal.ca/~goodfeli/fer2013.html) datasets of labeled images.

## Data Processing:
Exploratory analysis, cleaning, normalization, and train/test set splitting were done in the CreateTrainTestCV notebook. Preprocessing steps included converting to grayscale, cropping and scaling to 192x192px, and normalizing pixel intensity values between 0 and 1. The images were converted into numpy arrays to be used in training.

## Model Training:
Experiments were conducted using experiments.py and automate.py. Experiments.py created a json file of neural net architectures to be stacked and compiled using ad_lib_model.py. That way, you can queue models to train and let all of them train overnight. Results are logged in results.csv.

## Model Evaluation:
The best performance came from model 5 which had 12 convolutional layers and 3 fully connected layers, achieving a test set accuracy of 57%.
