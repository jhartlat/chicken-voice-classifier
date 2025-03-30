import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
import os
import tensorflow as tf
from preprocess import audio_to_log_mel_spec
from config import CLASS_NAMES
import argparse

def record_audio(duration=1.5, fs=22050, channels=1, filename="recording.wav"):
    """
    Record a snippet of audio and save it to 'filename'.
    This file is overwritten on each loop.
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, audio)
    return filename

def main():
    # Confidence threshold: if the prediction's confidence is below this value,
    # we mark the result as "Unknown".
    threshold = 0.70
    model_path = "models/trained_custom_model.keras"

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded. Starting real-time prediction loop.")

    # Continuous loop: record, predict, display, then sleep briefly.
    while True:
        # Record audio snippet (overwrites the same file each time)
        filename = record_audio(duration=1.5)

        try:
            # Convert audio to a log-mel spectrogram
            spec = audio_to_log_mel_spec(filename)
        except Exception as e:
            print(f"Error processing audio: {e}")
            continue

        # Add channel and batch dimensions for prediction input shape
        spec = np.expand_dims(spec, axis=-1)   # add channel dimension
        spec = np.expand_dims(spec, axis=0)      # add batch dimension

        # Run the model prediction
        predictions = model.predict(spec)
        pred_class_index = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_class_index]
        confidence = predictions[0][pred_class_index]

        # Check if confidence is above threshold
        if confidence < threshold:
            result = "Unknown"
            print(f"Prediction: {result} (Confidence: {confidence*100:.2f}%)")
        else:
            result = pred_class
            print(f"Prediction: {result} (Confidence: {confidence*100:.2f}%)")

        # Sleep for a short interval (e.g., 2 seconds) before recording again.
        time.sleep(2)

if __name__ == "__main__":
    main()
