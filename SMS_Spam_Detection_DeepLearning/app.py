"""
app.py

Streamlit web application for the SMS Spam Detection System.
Loads the already-trained model (does NOT train anything here).

Run with:
    streamlit run app.py
"""

import os
import json
import pickle

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "spam_detector.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    """Load the trained model, tokenizer, and config once and cache them."""
    missing = []
    if not os.path.exists(MODEL_PATH):
        missing.append(MODEL_PATH)
    if not os.path.exists(TOKENIZER_PATH):
        missing.append(TOKENIZER_PATH)
    if not os.path.exists(CONFIG_PATH):
        missing.append(CONFIG_PATH)

    if missing:
        return None, None, None, missing

    from keras.models import load_model

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return model, tokenizer, config, []


def clean_text(text):
    """Same lightweight cleaning used during training (kept in sync)."""
    import re
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s£$%!?.,:/@#&*+-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(message, model, tokenizer, config):
    from keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(message)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=config["max_length"],
        padding="post",
        truncating="post",
    )
    probability = float(model.predict(padded, verbose=0)[0][0])
    threshold = config.get("threshold", 0.5)
    label = "SPAM" if probability >= threshold else "HAM"
    return label, probability


def main():
    st.title("📱 SMS Spam Detector")
    st.caption("Deep Learning Based Text Classification")
    st.write("")

    model, tokenizer, config, missing = load_artifacts()

    if missing:
        st.error(
            "Model artifacts not found. Please train the model first by running:\n\n"
            "`python src/train.py`\n\n"
            "Missing files:\n" + "\n".join(f"- {m}" for m in missing)
        )
        st.stop()

    st.subheader("Enter your SMS message:")
    message = st.text_area(
        label="SMS message input",
        label_visibility="collapsed",
        height=140,
        placeholder="Type or paste an SMS message here...",
    )

    check_clicked = st.button("Check Message", type="primary", use_container_width=True)

    if check_clicked:
        if not message or not message.strip():
            st.warning("Please enter a message before checking.")
        else:
            try:
                label, probability = predict(message, model, tokenizer, config)
            except Exception as e:
                st.error(f"Something went wrong while making a prediction: {e}")
                st.stop()

            st.write("")
            if label == "SPAM":
                st.error(f"🚨 SPAM MESSAGE")
                st.write(f"**Spam Probability:** {probability * 100:.1f}%")
                st.progress(min(max(probability, 0.0), 1.0))
            else:
                st.success(f"✅ HAM / NOT SPAM")
                st.write(f"**Spam Probability:** {probability * 100:.1f}%")
                st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("About this model"):
        st.write(
            "This classifier uses an Embedding + LSTM deep learning architecture "
            "trained on the UCI SMS Spam Collection dataset."
        )
        if config:
            st.write(f"- Max sequence length: {config.get('max_length')}")
            st.write(f"- Test Accuracy: {config.get('test_accuracy', 'N/A')}")
            st.write(f"- Precision: {config.get('precision', 'N/A')}")
            st.write(f"- Recall: {config.get('recall', 'N/A')}")
            st.write(f"- F1-score: {config.get('f1_score', 'N/A')}")


if __name__ == "__main__":
    main()
