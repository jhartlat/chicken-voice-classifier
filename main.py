# main.py

import argparse
from train import main as train_model
from evaluate import evaluate

def run():
    parser = argparse.ArgumentParser(description="Chicken Voice Classification CLI")
    parser.add_argument('--train', action='store_true', help='Train a model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a saved model')

    args = parser.parse_args()

    if args.train:
        print("🔧 Training model...")
        train_model()

    elif args.evaluate:
        print("📊 Evaluating model...")
        evaluate()

    else:
        parser.print_help()

if __name__ == "__main__":
    run()
