import tkinter as tk
from tkinter import filedialog
from predict import predict

def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Choose a .wav file",
        filetypes=[("WAV files", "*.wav")]
    )
    return file_path

def main():
    print("Select a .wav file to classify.")
    wav_path = choose_file()

    if not wav_path:
        print("No file selected. Exiting.")
        return

    model_path = "models/trained_custom_model.keras"
    print(f"Classifying: {wav_path}")
    predict(wav_path, model_path)

if __name__ == "__main__":
    main()
