from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import io

# -------------------------------------------------
# BASE DIRECTORY
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)  # Allow mobile app to connect


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

# -------------------------------------------------
# IMAGE DIRECTORY (for browse mode)
# -------------------------------------------------
IMAGE_DIR = os.path.join(BASE_DIR, "..", "Images", "animal_data")

if os.path.exists(IMAGE_DIR):
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(image_files)
else:
    image_files = []

current_index = 0


# -------------------------------------------------
# PREDICTION FUNCTION (from file path)
# -------------------------------------------------
def predict_from_path(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)
    idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100

    return class_names[idx], round(confidence, 2)


# -------------------------------------------------
# PREDICTION FUNCTION (from image bytes)
# -------------------------------------------------
def predict_from_bytes(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)
    idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100

    return class_names[idx], round(confidence, 2)


# -------------------------------------------------
# ROUTE 1: Health Check
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Animal Classifier API is live!",
        "total_classes": len(class_names),
        "classes": class_names
    })


# -------------------------------------------------
# ROUTE 2: Upload Image from Mobile & Predict
# Mobile sends image → API returns label + confidence
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send image with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = file.read()
        label, confidence = predict_from_bytes(file_bytes)

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# ROUTE 3: Get Current Image (Browse Mode)
# -------------------------------------------------
@app.route("/current", methods=["GET"])
def current_image():
    global current_index

    if not image_files:
        return jsonify({"error": "No images found in IMAGE_DIR"}), 404

    img_name = image_files[current_index]
    label, confidence = predict_from_path(os.path.join(IMAGE_DIR, img_name))

    return jsonify({
        "success": True,
        "image_url": f"/images/{img_name}",
        "label": label,
        "confidence": confidence,
        "index": current_index,
        "total": len(image_files)
    })


# -------------------------------------------------
# ROUTE 4: Next Image
# -------------------------------------------------
@app.route("/next", methods=["GET"])
def next_image():
    global current_index

    if not image_files:
        return jsonify({"error": "No images found"}), 404

    current_index = (current_index + 1) % len(image_files)
    img_name = image_files[current_index]
    label, confidence = predict_from_path(os.path.join(IMAGE_DIR, img_name))

    return jsonify({
        "success": True,
        "image_url": f"/images/{img_name}",
        "label": label,
        "confidence": confidence,
        "index": current_index,
        "total": len(image_files)
    })


# -------------------------------------------------
# ROUTE 5: Previous Image
# -------------------------------------------------
@app.route("/previous", methods=["GET"])
def previous_image():
    global current_index

    if not image_files:
        return jsonify({"error": "No images found"}), 404

    current_index = (current_index - 1) % len(image_files)
    img_name = image_files[current_index]
    label, confidence = predict_from_path(os.path.join(IMAGE_DIR, img_name))

    return jsonify({
        "success": True,
        "image_url": f"/images/{img_name}",
        "label": label,
        "confidence": confidence,
        "index": current_index,
        "total": len(image_files)
    })


# -------------------------------------------------
# ROUTE 6: Serve Images
# -------------------------------------------------
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


# -------------------------------------------------
# ROUTE 7: Get All Class Names
# -------------------------------------------------
@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "success": True,
        "classes": class_names,
        "total": len(class_names)
    })


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    # host="0.0.0.0" makes it accessible on your local network
    # Mobile device must be on the same WiFi to connect
    app.run(debug=True, host="0.0.0.0", port=5000)