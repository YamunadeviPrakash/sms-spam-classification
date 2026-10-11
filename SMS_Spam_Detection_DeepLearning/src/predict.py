"""
predict.py

Load the trained SMS spam detector and predict on new messages.

Usage:
    python src/predict.py "Congratulations! You have won a free prize!"
"""

import os
import sys
import json
import pickle

import numpy as np

# Allow running as `python src/predict.py` from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "spam_detector.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

_model = None
_tokenizer = None
_config = None


def _load_artifacts():
    """Lazily load model, tokenizer, and config. Raises clear errors if missing."""
    global _model, _tokenizer, _config

    if _model is not None and _tokenizer is not None and _config is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. Please run 'python src/train.py' first."
        )
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at '{TOKENIZER_PATH}'. Please run 'python src/train.py' first."
        )
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Config file not found at '{CONFIG_PATH}'. Please run 'python src/train.py' first."
        )

    # Import tensorflow only when needed (keeps startup fast if not used)
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    _model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    with open(CONFIG_PATH, "r") as f:
        _config = json.load(f)


def predict_message(message):
    """
    Predict whether a given SMS message is SPAM or HAM.

    Returns
    -------
    dict with keys: message, label, probability (spam probability, 0-1)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if message is None or not str(message).strip():
        raise ValueError("Input message is empty. Please provide a valid SMS message.")

    _load_artifacts()

    cleaned = clean_text(message)
    sequence = _tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=_config["max_length"],
        padding="post",
        truncating="post",
    )

    probability = float(_model.predict(padded, verbose=0)[0][0])
    threshold = _config.get("threshold", 0.5)
    label = "SPAM" if probability >= threshold else "HAM"

    return {
        "message": message,
        "label": label,
        "probability": probability,
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "Your SMS message here"')
        sys.exit(1)

    message = " ".join(sys.argv[1:])

    try:
        result = predict_message(message)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    spam_prob = result["probability"] * 100 if result["label"] == "SPAM" else result["probability"] * 100

    print(f"\nMessage: {result['message']}\n")
    print(f"Prediction: {result['label']}")
    print(f"Spam Probability: {result['probability'] * 100:.1f}%")


if __name__ == "__main__":
    main()
