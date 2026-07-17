import os
import cv2
import numpy as np

IMG_SIZE = 64

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found!")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def save_model(model, model_path="models/model.h5"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved at: {model_path}")


def load_saved_model(model_path="models/model.h5"):
    from tensorflow.keras.models import load_model
    return load_model(model_path)
