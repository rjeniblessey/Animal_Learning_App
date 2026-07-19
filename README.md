Project Title: Animal Learning app

 Project Overview
The Intelligence Game Based Language Education System is an AI-powered learning application designed to help users learn language vocabulary using images and audio.
The system uses a deep learning model trained on an image dataset to recognize objects and predict their names in sequence. Once an image is predicted, the system speaks the name of the object using audio output, making learning more interactive and engaging like a game.
The trained model is developed using MobileNetV2 and later converted into TensorFlow Lite (TFLite) format to deploy the system as a mobile application.
The application contains Next and Previous buttons to navigate through images and predict them one by one in sequence order.

Project Description
In this project, a dataset containing labeled images of different objects is collected and used to train a deep learning model.The trained model learns to identify the image classes and is capable of predicting unseen images (new images not used during training).
When the application runs:
1.	An image is displayed on the screen.
2.	The trained MobileNetV2 model predicts the class (image name).
3.	The predicted name is converted into audio using a text-to-speech module.
4.	The system announces the image name through voice.
5.	The user can click Next or Previous buttons to move forward or backward through the image sequence.
This approach helps users learn vocabulary visually and audibly, making the system suitable for children and beginners in language learning.

Techniques Used
1. Dataset Preparation
•	Image dataset containing multiple object classes.
•	Images are labelled according to their class names.
•	Dataset is split into:
o	Training set
o	Testing set
2. Deep Learning Model – MobileNetV2
•	MobileNetV2 is used as the base CNN model.
•	Transfer learning technique is applied.
•	Final classification layer is modified based on number of classes.
•	Model is trained using:
o	Image preprocessing
o	Data augmentation
o	Optimization using Adam optimizer
3. Prediction on Unseen Data
•	New images not present in training set are passed to the model.
•	The model predicts one image at a time in sequence order.
•	Output is the class name of the image.
4. Audio Output (Text to Speech)
•	Predicted class label is converted into speech.
•	Audio feedback helps reinforce learning.
5. Model Conversion to Mobile App
•	Trained model is converted to TensorFlow Lite (.tflite).
•	TFLite model is integrated into a mobile application.
•	Ensures fast and lightweight execution on mobile devices.

Features 

✔ Image-based learning
✔ Sequence prediction (one-by-one image processing)
✔ Next and Previous navigation buttons
✔ Audio output for predicted image name
✔ Lightweight deep learning model (MobileNetV2)
✔ Offline prediction capability using TFLite
✔ User-friendly mobile interface
✔ Supports unseen image prediction
✔ Interactive and game-based learning style

 Tools & Technologies Used
 
•	Programming Language: Python
•	Deep Learning Framework: TensorFlow / Keras
•	Model: MobileNetV2
•	Mobile Deployment: TensorFlow Lite (TFLite)
•	Dataset Type: Image dataset
•	Text to Speech: Android TTS / gTTS
•	Platform: Android Mobile Application

