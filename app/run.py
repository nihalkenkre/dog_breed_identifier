import sys
from glob import glob
import cv2
import numpy as np
import pickle
import os
import matplotlib as plt

from scripts.extract_bottleneck_features import extract_Resnet50
from scripts.create_model import create_model

from flask import Flask, redirect, url_for, request, render_template, current_app
from flask_socketio import SocketIO, emit, send

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

from PIL import Image

from threading import Thread, Event

# instantiate the resnet model to detect presence of humans or dogs in the image
ResNet50_model = ResNet50(weights='imagenet')

# Model which is 'appended' to the resnet 50 model
model_resnet = create_model()

# load the dog names/labels.
breed_names = pickle.load(open('./app/data/breed_names.pkl', 'rb'))
#identified_dog_breed = None

face_cascade = cv2.CascadeClassifier(
    './app/haarcascades/haarcascade_frontalface_alt.xml')

UPLOAD_FOLDER = os.path.join('static')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['SERVER_NAME'] = '127.0.0.1:8000'
#app.config['APPLICATION_ROOT'] = './app'
#app.config['PREFERRED_URL_SCHEME'] = 'http'

socketio = SocketIO(app)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def look_for_human(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0


def get_dog_breed(img_path):
    # get a preprocessed tensor representation of the input image which can be passed to the resnet50 model
    img = preprocess_input(path_to_tensor(img_path))

    # get the bottleneck values of the resnet50 model for this image
    # these values will be passed to our model above as input
    # and it would be able to provide a prediction
    bottleneck_values = extract_Resnet50(img)

    # Use the dog names list to return the label of the prediction
    prediction = breed_names[np.argmax(model_resnet.predict(
        bottleneck_values))]

    return prediction


def identify(img_path):
    img = cv2.imread(img_path)

    # look for human
    print('looking for human...')
    found_human = look_for_human(img)
    found_dog = False

    # if human is found, look for the dog breed resembling the human
    if found_human:
        print('found.\n')
        print('looking for dog breed which resembles the human...')
        found_dog = dog_detector(img_path)

        # if a resembling dog breed is found return the breed
        if found_dog:
            breed = get_dog_breed(img_path)
            print('found ' + breed + '\n')

            return breed
        else:
            print('not found.\n')
    else:
        print('not found.\n')

    # if a dog breed resembling a human is not found, look for a dog 
    # and try to identify the breed
    if not found_dog:
        # look for dog

        print('looking for dog...')
        found_dog = dog_detector(img_path)

        # if dog is found then look for the breed
        if found_dog:
            print('found.\n')
            print('looking for breed...')
            breed = get_dog_breed(img_path)
            print(breed + '\n')

            return breed
    else:
        print('not found.\n')

    if not found_human and not found_dog:
        print("\nCould not detect dog or human.\n")

    print("================================\n")

    return None

# This class provides a thread object that can be setup to run whenever
# a new image is to be identified.
#
# This runs the identification in the 'background', and notifies the front end
#   with the name of the breed when identification is complete
class Deamon(Thread):
    def __init__(self):
        Thread.__init__(self)

        # path of the image
        self.img_path = None

    def run(self):
        if self.img_path is not None:
            identified_dog_breed = identify(self.img_path)

            # emits a signal with the name of breed for the receiver
            # the receiver is the socket on identified.html page
            emit('identify_done', identified_dog_breed if identified_dog_breed is not None else 'Could not identify dog breed')


deamon = Deamon()


@app.route('/', methods=['GET'])
def index_html():
    return render_template('index.html')


@app.route('/favicon.ico')
def get_favicon():
    # Returning data to prevent the error/warning on the browser console
    return b'0'


@app.route('/identified.html', methods=['GET', 'POST'])
def identified_html():
    if len(request.files) > 0:
        f = request.files['upload']
        upload_path = os.path.join('./app/static/', f.filename)
        f.save(upload_path)

        # Create a 'thumbnail' image to be displayed on the front end
        # with the appropriate size
        with Image.open(upload_path) as img:
            img.thumbnail((400, 400))
            img.save(os.path.join('./app/static/', 'thumb_' + f.filename))

        # Set the variable in the thread.
        deamon.img_path = upload_path

        return render_template('identified.html', img_src='thumb_' + f.filename)


# This is a signal from the front end that there is a
# new img waiting to be identified
@socketio.on('connect', namespace='/identify_breed')
def handle_connect(data):
    deamon.run()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

    # if the deamon thread is alive, shut it down gracefully
    if deamon.is_alive():
        deamon.join()
