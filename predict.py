# predict.py

import tensorflow as tf
import numpy as np
import argparse
from preprocess import audio_to_log_mel_spec
from config import CLASS_NAMES

def predict(file_path, model_path):
    # Load and process audio
    spec = audio_to_log_mel_spec(file_path)
    spec = np.expand_dims(spec, axis=-1)  # add channel dim
    spec = np.expand_dims(spec, axis=0)   # add batch dim

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(spec)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    print(f"\n🎧 Predicted Sound: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict chicken sound from a .wav file")
    parser.add_argument("file", help="Path to .wav file")
    parser.add_argument("--model", default="models/trained_custom_model.keras", help="Path to saved model")

    args = parser.parse_args()
    predict(args.file, args.model)
