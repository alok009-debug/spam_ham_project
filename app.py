import json
import pickle
import string

import streamlit as st
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Spam / Ham Classifier",
    page_icon="📩",
    layout="centered",
)

# -------------------------------------------------
# Load model + tokenizer (cached so it loads only once)
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("spam_lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("config.json") as f:
        max_len = json.load(f)["max_len"]
    return model, tokenizer, max_len


model, tokenizer, MAX_LEN = load_artifacts()

# -------------------------------------------------
# Text cleaning — same as training
# -------------------------------------------------
puncts = string.punctuation
STOP = set(stopwords.words("english"))


def clean(text: str) -> str:
    text = str(text).replace("Subject", "")
    text = text.translate(str.maketrans("", "", puncts))
    return " ".join(w.lower() for w in text.split() if w.lower() not in STOP)


# -------------------------------------------------
# Prediction helper
# -------------------------------------------------
def classify(message: str):
    seq = tokenizer.texts_to_sequences([clean(message)])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob_spam = float(model.predict(seq, verbose=0)[0][0])
    pred = "SPAM" if prob_spam >= 0.5 else "HAM"
    confidence = prob_spam if pred == "SPAM" else 1 - prob_spam
    return pred, confidence, prob_spam


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📩 Spam / Ham Classifier")
st.caption("LSTM-based classifier trained on 9,390 balanced samples — **94.3% test accuracy**")

st.divider()

message = st.text_area(
    "Enter your message:",
    placeholder="Paste or type the message here...",
    height=180,
)

col1, col2 = st.columns([1, 1])

with col1:
    classify_clicked = st.button("🔍 Classify", use_container_width=True, type="primary")

with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if classify_clicked:
    if not message.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            pred, conf, prob_spam = classify(message)

        st.divider()

        if pred == "SPAM":
            st.error(f"### 🚨 {pred}")
        else:
            st.success(f"### ✅ {pred}")

        st.metric("Confidence", f"{conf * 100:.2f}%")

        c1, c2 = st.columns(2)
        c1.metric("Spam probability", f"{prob_spam * 100:.2f}%")
        c2.metric("Ham probability", f"{(1 - prob_spam) * 100:.2f}%")

        st.progress(min(max(prob_spam, 0.0), 1.0))

st.divider()

with st.expander("💡 Try an example"):
    st.markdown("**Spam example:**")
    st.code("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!", language=None)

    st.markdown("**Ham example:**")
    st.code("Hey, are we still meeting for lunch at 1pm tomorrow?", language=None)

    st.markdown("**Ham example from your dataset:**")
    st.code("into the kingdom of god and those that are entering in he lord pardon us in this thing we pray thee have us excused", language=None)