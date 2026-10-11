"""
train.py

Full training pipeline for the SMS Spam Detection LSTM model.

Run with:
    python src/train.py
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for terminal use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Allow running as `python src/train.py` from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_pipeline

# ----------------------------------------------------------------------
# Paths (all relative to project root so this works anywhere)
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "SMSSpamCollection")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "spam_detector.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 64
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 42


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # ------------------------------------------------------------------
    # 1. Load + clean + preprocess dataset
    # ------------------------------------------------------------------
    print_header("STEP 1: Loading and Cleaning Dataset")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    df, stats = preprocess_pipeline(DATASET_PATH)

    print(f"Total raw messages       : {stats['total_raw_messages']}")
    print(f"Missing values found     : {stats['missing_values']}")
    print(f"Duplicate messages found : {stats['duplicate_count']}")
    print(f"Total messages (cleaned) : {stats['total_messages']}")
    print(f"Ham messages             : {stats['ham_count']} ({stats['ham_percentage']}%)")
    print(f"Spam messages            : {stats['spam_count']} ({stats['spam_percentage']}%)")

    # ------------------------------------------------------------------
    # 2. Exploratory Data Analysis
    # ------------------------------------------------------------------
    print_header("STEP 2: Exploratory Data Analysis")

    sns.set_style("whitegrid")

    # Class distribution plot
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x="label", data=df, hue="label", palette={"ham": "#4C72B0", "spam": "#C44E52"}, legend=False)
    ax.set_title("Spam vs Ham Distribution")
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    class_dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(class_dist_path, dpi=150)
    plt.close()
    print(f"Saved: {class_dist_path}")

    # Message length distribution
    df["message_length"] = df["clean_message"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="message_length", hue="label", bins=40, kde=True,
                 palette={"ham": "#4C72B0", "spam": "#C44E52"}, element="step")
    plt.title("Message Length Distribution (word count)")
    plt.xlabel("Number of Words per Message")
    plt.ylabel("Frequency")
    plt.tight_layout()
    msg_len_path = os.path.join(OUTPUT_DIR, "message_length_distribution.png")
    plt.savefig(msg_len_path, dpi=150)
    plt.close()
    print(f"Saved: {msg_len_path}")

    # Choose max sequence length based on actual data (95th percentile, capped)
    length_95 = int(np.percentile(df["message_length"], 95))
    MAX_LENGTH = max(20, min(length_95 + 5, 150))
    print(f"Selected MAX_LENGTH (based on 95th percentile of word counts): {MAX_LENGTH}")

    # ------------------------------------------------------------------
    # 3. Train / Test split (BEFORE tokenizer fitting, to avoid leakage)
    # ------------------------------------------------------------------
    print_header("STEP 3: Train/Test Split")

    X = np.array(df["clean_message"].tolist(), dtype=object)
    y = np.array(df["label_encoded"].tolist(), dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # ------------------------------------------------------------------
    # 4. Tokenization + Padding
    # ------------------------------------------------------------------
    print_header("STEP 4: Tokenization and Sequence Padding")

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train)  # fit ONLY on training data

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding="post", truncating="post")

    actual_vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size used   : {actual_vocab_size}")
    print(f"Max sequence length    : {MAX_LENGTH}")

    # ------------------------------------------------------------------
    # 5. Build LSTM model
    # ------------------------------------------------------------------
    print_header("STEP 5: Building LSTM Model")

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
        LSTM(LSTM_UNITS),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # ------------------------------------------------------------------
    # 6. Train model
    # ------------------------------------------------------------------
    print_header("STEP 6: Training Model")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_pad, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 7. Training graphs
    # ------------------------------------------------------------------
    print_header("STEP 7: Generating Training Graphs")

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, "training_accuracy.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"Saved: {acc_path}")

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training Loss", marker="o")
    plt.plot(history.history["val_loss"], label="Validation Loss", marker="o")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, "training_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Saved: {loss_path}")

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    print_header("STEP 8: Model Evaluation")

    y_pred_prob = model.predict(X_test_pad, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    # ------------------------------------------------------------------
    # 9. Save model, tokenizer, config
    # ------------------------------------------------------------------
    print_header("STEP 9: Saving Model Artifacts")

    model.save(MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer: {TOKENIZER_PATH}")

    config = {
        "max_length": MAX_LENGTH,
        "threshold": 0.5,
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "test_accuracy": round(float(test_accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved config: {CONFIG_PATH}")

    print_header("TRAINING COMPLETE")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
