
# train.py

from dataset import prepare_data
from model_factory import get_model
import os
import tensorflow as tf

# Configuration
MODEL_NAME = "custom"  # options: "custom", "inception", "resnet"
EPOCHS = 30
BATCH_SIZE = 64
SAVE_DIR = "models"
SAVE_NAME = f"trained_{MODEL_NAME}_model.keras"  # new standard format

def main():
    print(f"Preparing data...")
    x_train, x_test, y_train, y_test = prepare_data()

    print(f"Building {MODEL_NAME} model...")
    model = get_model(MODEL_NAME)
    model.summary()

    print("Starting training...\n")
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("\nEvaluating on test data...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # Save the model if accuracy is decent (optional threshold logic)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

if __name__ == "__main__":
    main()
