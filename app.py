import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_frame
from utils.sentence_builder import add_word, clear_sentence, get_sentence

# Check if the model file exists
model_path = os.path.join("model", "asl_model.h5")
if os.path.exists(model_path):
    print("Model file found! Loading with TensorFlow...")
    try:
        # Load the model using TensorFlow
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found! Check the file path.")

class_names = ['hello', 'thanks', 'yes', 'no', 'please', 'i', 'you', 'love']  # update as needed

st.set_page_config(page_title="Sign Language Converter")
st.title("🤟 Sign Language to Sentence Converter")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
run = st.checkbox('Start Webcam')

if st.button("Clear Sentence"):
    clear_sentence()

if st.button("Speak Sentence"):
    sentence = get_sentence()
    st.success(f"🗣 Speaking: {sentence}")
    engine.say(sentence)
    engine.runAndWait()

st.subheader("📝 Current Sentence:")
st.write(get_sentence())

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    processed = preprocess_frame(frame)
    
    prediction = model.predict(processed)
    predicted_word = class_names[np.argmax(prediction)]

    # Add word to sentence buffer
    add_word(predicted_word)

    # Display prediction on frame
    cv2.putText(frame, f"Prediction: {predicted_word}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()

