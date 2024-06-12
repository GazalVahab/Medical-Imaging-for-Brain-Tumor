from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import imutils
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your pre-trained model
model = load_model('mobilenet_model(1).h5')

# Function to crop the image
def crop_img(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

    return new_img

# Preprocessing and prediction function
def preprocess_and_predict(image_path):
    img = cv2.imread(image_path)
    #cropped_img = crop_img(img)
    resized_img = cv2.resize(img, (224, 224))
    normalized_img = resized_img / 255.0
    normalized_img = np.expand_dims(normalized_img, axis=0)  # Add batch dimension

    prediction = model.predict(normalized_img)
    return prediction

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        prediction = preprocess_and_predict(filepath)

        # Add your prediction statement here
        if prediction[0] < 0.5:
            result = 'Tumor'
        else:
            result = 'Normal'

        return render_template('result.html', image_url=url_for('static', filename='uploads/' + filename), prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
