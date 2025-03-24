# Chicken Voice Classifier

This project is a machine learning system that listens to chicken sounds and predicts what type of sound it is. The goal is to help detect distress or important behavior in chickens by classifying their vocalizations.

## Features

- Converts `.wav` audio files into spectrograms (visual sound patterns)
- Trains a Convolutional Neural Network (CNN) to classify sounds
- Classifies sounds into four types:
  - alarm
  - egg
  - feeding
  - heat
- Saves a trained model you can use later
- Predicts sound types from new `.wav` files

---

## Project Structure

```plaintext
chicken-voice-classifier/
│
├── data/                     # Contains subfolders with audio files (alarm, egg, feeding, heat)
├── models/                   # Trained model is saved here
├── config.py                 # Global settings like paths and audio options
├── preprocess.py             # Turns audio files into spectrograms
├── dataset.py                # Loads and prepares the dataset for training/testing
├── model_factory.py          # Builds the CNN model
├── train.py                  # Trains the model
├── evaluate.py               # Tests the model and shows accuracy
├── predict.py                # Predicts the class of a new audio file
└── main.py                   # Command line tool to train or evaluate the model
```


## Requirements
Make sure Python is installed. Then install the needed libraries:

```bash
pip install -r requirements.txt
```


## Data Setup
Place your .wav files into this folder structure:

```plaintext
data/
├── alarm/
├── egg/
├── feeding/
└── heat/
```
Each subfolder should only contain .wav files for that class. These will be used to train the model.


## How to Train the Model
To train the model on your dataset, run:

```bash
python main.py --train
```

After training, a model will be saved in:
```bash
models/trained_custom_model.keras
```


## How to Evaluate the Model
To test the Model and view its performance:
```bash
python main.py --evaluate
```
This will print a classification report and confusion matrix.


## How to Predict with New Sounds
If you have a .wav file and want to know what it means, run:
```bash
python predict.py path/to/your-file.wav
```


## Future Plans
- Run the model on a Raspberry Pi
- Record live chicken sounds using a microphone
- Predict sounds in real time
- Send alerts or log unusual behavior


## Notes
- The model currently expects audio files around 1.5 seconds long.
- Only .wav files are supported.
- This project is still in progress and will continue to improve over time.

