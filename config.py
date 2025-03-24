# config.py

# Dataset
DATASET_PATH = "data"
CLASS_NAMES = ["alarm", "egg", "feeding", "heat"]
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Audio
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
MEL_BINS = 128
WAV_SIZE = 33075
WINDOW = 'hann'

# Model
INPUT_SHAPE = (128, 130, 1)
NUM_CLASSES = len(CLASS_NAMES)
MODEL_SAVE_DIR = "models"
DEFAULT_MODEL_NAME = "trained_custom_model.keras"
