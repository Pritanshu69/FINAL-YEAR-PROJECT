import streamlit as st

st.set_page_config(page_title="Sign Language Converter")

import cv2
import numpy as np
import os
import tensorflow as tf

from utils.preprocessing import preprocess_frame
from utils.sentence_builder import add_word, clear_sentence, get_sentence

# ------------------------------
# ✅ Model loading
# ------------------------------
model_path = os.path.join("model", "asl_model.h5")
model = None

st.write(f"🔍 Trying to load model from: `{os.path.abspath(model_path)}`")

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ Model file not found! Please check the path.")

# ------------------------------
# Label names - update as needed
# ------------------------------
class_names = ['hello', 'thanks', 'yes', 'no', 'please', 'i', 'you', 'love']

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🤟 Sign Language to Sentence Converter")

# Webcam control using session state
if "camera" not in st.session_state:
    st.session_state.camera = None

run = st.checkbox("📷 Start Webcam")

# Sentence buttons
if st.button("🧹 Clear Sentence"):
    clear_sentence()

# Sentence output
st.subheader("📝 Current Sentence:")
st.write(get_sentence())

# Image placeholder
FRAME_WINDOW = st.image([])

# ------------------------------
# Webcam and prediction loop
# ------------------------------
if run and model is not None:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)

    ret, frame = st.session_state.camera.read()
    if not ret:
        st.error("❌ Unable to access camera.")
    else:
        frame = cv2.flip(frame, 1)
        processed = preprocess_frame(frame)

        prediction = model.predict(processed)
        predicted_word = class_names[np.argmax(prediction)]
        add_word(predicted_word)

        # Overlay prediction on video
        cv2.putText(frame, f"Prediction: {predicted_word}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ------------------------------
# Cleanup on stop
# ------------------------------
else:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
