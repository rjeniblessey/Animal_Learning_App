import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import os

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
model = tf.keras.models.load_model("../Save_Model/animal_model1.h5")

# --------------------------
# CLASS NAMES
# --------------------------
class_names = [
    'Armadilles', 'Bear', 'Birds', 'Cow', 'Crocodile', 'Deer',
    'Elephant', 'Goat', 'Horse', 'Jaguar', 'Monkey',
    'Rabbit', 'Skunk', 'Tiger', 'Wild Boar'
]

# --------------------------
# IMAGE PATH
# --------------------------
img_path = "../Images/animal_data/goat.jpg"

if not os.path.exists(img_path):
    print("❌ Image not found")
    exit()

# --------------------------
# READ IMAGE
# --------------------------
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --------------------------
# PREPARE IMAGE FOR MODEL
# --------------------------
IMG_SIZE = 224
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_norm = img_resized / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# --------------------------
# PREDICTION
# --------------------------
pred = model.predict(img_input, verbose=0)
idx = np.argmax(pred)
pred_class = class_names[idx]
confidence = pred[0][idx] * 100

print("✅ Predicted:", pred_class)
print(f"✅ Confidence: {confidence:.2f}%")

# --------------------------
# DISPLAY IMAGE (MEDIUM SIZE)
# --------------------------
display_img = cv2.resize(img_bgr, (600, 600))

cv2.putText(
    display_img,
    f"{pred_class} ({confidence:.2f}%)",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 0),
    2
)

cv2.putText(
    display_img,
    "Click image to hear name | ESC to exit",
    (20, 585),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)

# --------------------------
# MOUSE CLICK → SPEAK (FIXED)
# --------------------------
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        engine = pyttsx3.init()      # IMPORTANT FIX
        engine.say(f"This is a {pred_class}")
        engine.runAndWait()
        engine.stop()

# --------------------------
# SHOW WINDOW
# --------------------------
cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Prediction", 600, 600)
cv2.setMouseCallback("Prediction", on_mouse_click)

print("🖱️ Click image → speak")
print("❌ Press ESC to exit")

while True:
    cv2.imshow("Prediction", display_img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
