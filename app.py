from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from PIL import Image
#from gtts import gTTS
import time
import random
import pyttsx3

# -------------------------------------------------
# BASE DIRECTORY
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Template folder (outside Coding)
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "Templates")

STATIC_DIR = os.path.join(BASE_DIR, "..", "Static_css")

app = Flask(__name__,
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model_path = os.path.join(BASE_DIR, "..", "Save_Model", "animal_model1.h5")
model = tf.keras.models.load_model(model_path)

class_names = [
    'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer',
    'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey',
    'Rabbit', 'Skunk', 'Tiger', 'Wild Boar'
]

# Correct image folder
IMAGE_DIR = os.path.join(BASE_DIR, "..", "Images", "animal_data")

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# Shuffle only once
random.shuffle(image_files)


if len(image_files) == 0:
    raise Exception("No images found inside Images/animal_data")

current_index = 0


# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)
    idx = np.argmax(pred)
    return class_names[idx]


# -------------------------------------------------
# SERVE IMAGES
# -------------------------------------------------
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    global current_index

    img_name = image_files[current_index]
    label = predict_image(os.path.join(IMAGE_DIR, img_name))

    return render_template("index.html",
                           image=img_name,
                           label=label)

@app.route("/next")
def next_image():
    global current_index

    current_index = (current_index + 1) % len(image_files)

    img_name = image_files[current_index]
    label = predict_image(os.path.join(IMAGE_DIR, img_name))

    return jsonify({"image": img_name, "label": label})


@app.route("/previous")
def previous_image():
    global current_index

    current_index = (current_index - 1) % len(image_files)

    img_name = image_files[current_index]
    label = predict_image(os.path.join(IMAGE_DIR, img_name))

    return jsonify({"image": img_name, "label": label})


@app.route("/speak", methods=["POST"])
def speak():
    text = request.json["text"]

    filename = f"voice_{int(time.time()*1000)}.mp3"
    save_path = os.path.join(BASE_DIR, "..", "Output_Audios", filename)

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file(text, save_path)
    engine.runAndWait()

    return jsonify({"audio": filename})

# -------------------------------------------------
# SERVE AUDIO
# -------------------------------------------------
@app.route("/audio/<filename>")
def serve_audio(filename):
    audio_dir = os.path.join(BASE_DIR, "..", "Output_Audios")
    return send_from_directory(audio_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)
