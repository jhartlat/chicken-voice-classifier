# dataset.py

import os
import random
import numpy as np
from tqdm import tqdm
from preprocess import audio_to_log_mel_spec
from sklearn.model_selection import train_test_split
from config import DATASET_PATH, CLASS_NAMES, CLASS_TO_LABEL  # Imported from config.py

def load_dataset():
    data = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        label = CLASS_TO_LABEL[class_name]

        print(f"Loading {class_name}...")
        for filename in tqdm(os.listdir(class_dir)):
            if filename.endswith(".wav"):
                file_path = os.path.join(class_dir, filename)
                try:
                    log_mel_spec = audio_to_log_mel_spec(file_path)
                    data.append((log_mel_spec, label))
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    random.shuffle(data)
    return data

def prepare_data(test_size=0.2):
    dataset = load_dataset()
    X = np.array([item[0] for item in dataset])
    y = np.array([item[1] for item in dataset], dtype=int)


    # Add channel dimension for CNN
    X = X[..., np.newaxis]

    # One-hot encode labels
    num_classes = len(CLASS_NAMES)
    y_encoded = np.zeros((y.size, num_classes))
    y_encoded[np.arange(y.size), y] = 1

    return train_test_split(X, y_encoded, test_size=test_size, random_state=42)
