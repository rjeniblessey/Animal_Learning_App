import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
import os

# --------------------------
# LOAD MODEL
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
# IMAGE DIRECTORY
# --------------------------
IMAGE_DIR = "../Images/animal_data"
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

if not image_files:
    print("❌ No images found")
    exit()

index = 0
current_label = ""

# --------------------------
# TEXT TO SPEECH (FIXED)
# --------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.stop()                 # 🔴 REQUIRED for multiple clicks
    engine.say(f"This is a {text}")
    engine.runAndWait()

# --------------------------
# PREDICTION FUNCTION
# --------------------------
def predict_image(path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(rgb, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_input, verbose=0)
    idx = np.argmax(pred)

    return img, class_names[idx], pred[0][idx] * 100

# --------------------------
# DISPLAY IMAGE
# --------------------------
def show_image():
    global current_label

    img_path = os.path.join(IMAGE_DIR, image_files[index])
    img, label, conf = predict_image(img_path)
    current_label = label   # ✅ always updated

    display = cv2.resize(img, (800, 600))

    # Prediction text
    cv2.putText(
        display,
        f"{label} ({conf:.2f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    # Previous button
    cv2.rectangle(display, (50, 520), (200, 580), (255, 0, 0), -1)
    cv2.putText(display, "Previous", (70, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Next button
    cv2.rectangle(display, (600, 520), (750, 580), (255, 0, 0), -1)
    cv2.putText(display, "Next", (650, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Instruction
    cv2.putText(display, "Click Image = Speak",
                (280, 580),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    cv2.imshow("Animal Learning App", display)

# --------------------------
# MOUSE EVENT HANDLER
# --------------------------
def mouse_event(event, x, y, flags, param):
    global index

    if event == cv2.EVENT_LBUTTONDOWN:

        # PREVIOUS
        if 50 <= x <= 200 and 520 <= y <= 580:
            index = (index - 1) % len(image_files)
            show_image()

        # NEXT
        elif 600 <= x <= 750 and 520 <= y <= 580:
            index = (index + 1) % len(image_files)
            show_image()

        # IMAGE CLICK → SPEAK (works every time)
        else:
            speak(current_label)

# --------------------------
# START APPLICATION
# --------------------------
cv2.namedWindow("Animal Learning App", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Animal Learning App", 800, 600)
cv2.setMouseCallback("Animal Learning App", mouse_event)

show_image()

print("🧠 Kids Animal Learning App")
print("🖱 Click image → Speak name")
print("🔵 Click Next / Previous buttons")
print("❌ Press ESC to exit")

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
