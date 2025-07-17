import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
import cv2
import pandas as pd

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model("models/mnist_improved_model.h5")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

# --- Preprocess Drawn Image to MNIST Format ---
def preprocess_canvas_image(image_data_rgba):
    # Convert RGBA to grayscale
    img_pil = Image.fromarray(image_data_rgba.astype("uint8"), mode="RGBA").convert("L")

    # Convert to NumPy and invert (MNIST is white digit on black)
    img_array = np.array(img_pil)
    img_array = 255 - img_array

    # Threshold to remove soft edges
    _, img_thresh = cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY)

    # Get bounding box of the digit
    coords = cv2.findNonZero(img_thresh)
    if coords is None:
        return np.zeros((1, 28, 28, 1))  # Empty fallback if nothing drawn

    x, y, w, h = cv2.boundingRect(coords)
    digit = img_thresh[y:y+h, x:x+w]

    # applying a slight blur to improve digit shape
    digit = cv2.GaussianBlur(digit, (3, 3), 0)

    # Pad digit to square (preserve aspect ratio)
    size = max(w, h)
    padded = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # Resize to 20x20 and paste onto 28x28 canvas
    digit_resized = cv2.resize(padded, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = digit_resized

    # Normalize to 0–1 and reshape
    canvas = np.clip(canvas, 0, 255).astype("float32") / 255.0
    return canvas.reshape(1, 28, 28, 1)

# --- UI Header ---
st.title("MNIST Digit Recognition App")
st.write("Draw a digit (0–9) in the canvas below. The model will predict what digit it sees.")

# --- Canvas Component ---
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#000000",       # Black digit
    background_color="#FFFFFF",   # White background
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Prediction Logic ---
if canvas_result.image_data is not None:
    if canvas_result.image_data.sum() > 0:
        st.subheader("Your Drawing:")
        st.image(canvas_result.image_data, width=200)

        # Preprocess
        input_image = preprocess_canvas_image(canvas_result.image_data)

        # Show what model sees
        st.subheader("What the Model Sees:")
        st.image(input_image[0], width=150, clamp=True)

        # Predict
        prediction = model.predict(input_image)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        st.success(f" Prediction: *{predicted_digit}*")
        st.info(f"Confidence: *{confidence:.2f}%*")

        # All probabilities
        st.subheader("All Class Probabilities:")
        st.bar_chart(pd.DataFrame(prediction[0], index=range(10)))
    else:
        st.info("Please draw a digit on the canvas.")

# --- Sidebar Info ---
st.sidebar.header("ℹ About")
st.sidebar.info(
    "This app uses a CNN trained on the MNIST dataset.\n"
    "It accepts user-drawn digits using Streamlit's canvas.\n\n"
    "Preprocessing includes color inversion, centering, resizing, blurring, and normalization."
)