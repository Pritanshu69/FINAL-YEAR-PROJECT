import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_frame
from utils.sentence_builder import add_word, clear_sentence, get_sentence

# Load model and class names
model = load_model(os.path.join("model", "asl_model.h5"))
class_names = ['hello', 'thanks', 'yes', 'no', 'please', 'i', 'you', 'love']  # update as needed

# Text-to-Speech
engine = pyttsx3.init()

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

