import argparse

from train import main as train_main
from evaluate import evaluate as eval_model
# Note: inference.py should define a `predict(text: str) -> (label, confidence)` function

def parse_args():
    parser = argparse.ArgumentParser(description="Health Rumor Detection Pipeline")
    parser.add_argument(
        'mode',
        choices=['train', 'eval', 'infer'],
        help="Mode to run: 'train' to train model, 'eval' to evaluate on test set, 'infer' for single text inference"
    )
    parser.add_argument(
        '--text',
        type=str,
        help="Input text for inference (required if mode='infer')"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'train':
        print("Starting training...")
        train_main()
    elif args.mode == 'eval':
        print("Evaluating on test set...")
        eval_model()
    elif args.mode == 'infer':
        if args.text is None:
            print("Error: --text argument is required for inference mode.")
            return
        print(f"Performing inference on: {args.text}")
        try:
            from inference import predict
        except ImportError:
            print("Error: inference.py not found. Please implement inference.py with a predict function.")
            return
        label, confidence = predict(args.text)
        print(f"Prediction: {label}, Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()