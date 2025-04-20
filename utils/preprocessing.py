import cv2
import numpy as np

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Match model input
    img = img / 255.0                    # Normalize
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img
