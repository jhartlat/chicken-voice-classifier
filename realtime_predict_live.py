import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import tensorflow as tf
import os
import time

from preprocess import audio_to_log_mel_spec
from config import CLASS_NAMES

# Global parameters for audio
DURATION = 1.5  # seconds for prediction window
FS = 22050      # sampling rate
BUFFER_LENGTH = int(DURATION * FS)

# Global buffer to hold the latest audio samples (mono)
audio_buffer = np.zeros((BUFFER_LENGTH, 1), dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    # indata shape: (frames, channels)
    # Append the new samples and keep only the most recent BUFFER_LENGTH samples
    audio_buffer = np.vstack((audio_buffer, indata))
    if len(audio_buffer) > BUFFER_LENGTH:
        audio_buffer = audio_buffer[-BUFFER_LENGTH:]

class RealTimeLiveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Chicken Voice Classifier")

        # Labels for prediction and confidence
        self.prediction_label = tk.Label(root, text="Prediction: N/A", font=("Helvetica", 24))
        self.prediction_label.pack(pady=5)

        self.confidence_label = tk.Label(root, text="Confidence: N/A", font=("Helvetica", 22))
        self.confidence_label.pack(pady=5)

        self.alert_label = tk.Label(root, text="", font=("Helvetica", 20), fg="red")
        self.alert_label.pack(pady=5)

        # Matplotlib figure for live waveform
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # Load trained model
        self.model_path = "models/trained_custom_model.keras"
        print("Loading model...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded.")

        # Set prediction confidence threshold
        self.threshold = 0.70

        # Start the audio input stream with a callback
        self.stream = sd.InputStream(channels=1, samplerate=FS, callback=audio_callback)
        self.stream.start()

        # Start update loops: one for the live waveform, one for periodic predictions.
        self.update_waveform()
        self.update_prediction()

    def update_waveform(self):
        global audio_buffer
        # Create a time vector based on the current buffer length and DURATION seconds
        t = np.linspace(0, DURATION, num=len(audio_buffer))
        self.ax.clear()
        self.ax.plot(t, audio_buffer, color='blue')
        self.ax.set_title("Live Audio Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.fig.tight_layout()
        self.canvas.draw()
        # Update the waveform display every 50ms
        self.root.after(50, self.update_waveform)

    def update_prediction(self):
        global audio_buffer
        # Save the current buffer to a temporary file for prediction
        temp_filename = "temp_recording.wav"
        # Ensure the buffer shape is (n_samples, channels)
        write(temp_filename, FS, audio_buffer)

        try:
            # Process the temporary file into a log-mel spectrogram (for model input)
            spec = audio_to_log_mel_spec(temp_filename)
        except Exception as e:
            print("Error processing audio for prediction:", e)
            spec = np.zeros((128, 130))

        # Prepare the spectrogram shape for model prediction
        input_spec = np.expand_dims(spec, axis=-1)  # add channel dimension
        input_spec = np.expand_dims(input_spec, axis=0)  # add batch dimension

        # Run prediction using the loaded model
        predictions = self.model.predict(input_spec)
        pred_class_index = np.argmax(predictions)
        confidence = predictions[0][pred_class_index]

        # Check the confidence threshold to update the result
        if confidence < self.threshold:
            result = "Unknown"
            self.alert_label.config(text="Low confidence: Unknown sound detected!")
        else:
            result = CLASS_NAMES[pred_class_index]
            self.alert_label.config(text="")

        # Update the prediction and confidence labels
        self.prediction_label.config(text=f"Prediction: {result}")
        self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")

        # Remove the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        # Schedule next prediction update every 3500ms (3.5 seconds)
        self.root.after(3500, self.update_prediction)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeLiveGUI(root)
    root.mainloop()
