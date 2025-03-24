# evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from dataset import prepare_data
from config import CLASS_NAMES  # Updated import

# Config
MODEL_PATH = "models/trained_custom_model.keras"  # adjust as needed

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def evaluate():
    print("Loading test data...")
    _, x_test, _, y_test = prepare_data()

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Predicting...")
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)

if __name__ == "__main__":
    evaluate()
