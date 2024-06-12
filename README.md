Medical Imaging: Brain Tumor Detection using CNN and Transfer Learning

This project involves using Convolutional Neural Networks (CNNs) and Transfer Learning to detect brain tumors in medical images. The project is divided into two main parts: model training and a web application for prediction.

Model Training

There are two notebooks for training the models:

    Brain_Tumor_Detection_Project.ipynb: This notebook trains the models using images that have undergone preprocessing type 1.
    BrainTumorDetection_Project.ipynb: This notebook trains the models using images that have undergone preprocessing type 2.

After training various models, the best performance was achieved by the MobileNet model trained using preprocessing type 1. Therefore, this model was selected for deployment in the web application.

Web Application

The web application is built using Flask and allows users to upload medical images for brain tumor detection.
Features

  Image Upload: Users can upload an image for prediction.
  Prediction Result: After clicking 'Predict', the result is displayed along with the uploaded image.
  Navigation: Users can return to the home page to upload another image using the 'Upload Another Image' button.

Files Uploaded

    Brain_Tumor_Detection_Project.ipynb: Notebook for training models using preprocessing type 1.
    BrainTumorDetection_Project.ipynb: Notebook for training models using preprocessing type 2.
    app.py: Flask application file.
    index.html: HTML file for the home page.
    result.html: HTML file for displaying the prediction result.
    style.css: CSS file for styling the web pages.
    br_img.jpg: Background image used in the website.
