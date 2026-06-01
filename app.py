import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import nltk

# try:
#     nltk.data.find("corpora/stopwords")
# except:
#     nltk.download("stopwords")
# -----------------------
# PAGE CONFIG
# -----------------------
# st.balloons()
st.markdown(
    """
    <div style='text-align:center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/4221/4221484.png'
             width='120'>
    </div>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -----------------------
# LOAD CSS
# -----------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():

    rnn = tf.keras.models.load_model(
        "simple_rnn_imdb.h5"
    )

    lstm = tf.keras.models.load_model(
        "lstm_imdb.h5"
    )

    gru = tf.keras.models.load_model(
        "gru_imdb.h5"
    )

    return rnn, lstm, gru

# -----------------------
# LOAD TOKENIZER
# -----------------------
with open("imdb_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# -----------------------
# PREPROCESSING
# -----------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

def prepare(text):

    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding='post'
    )

    return padded

# -----------------------
# HEADER
# -----------------------
st.markdown("""
<div class='main-title'>
🎬 Movie Review Sentiment Analysis System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Deep Learning Based Sentiment Classification
</div>
""", unsafe_allow_html=True)

# -----------------------
# MODEL SELECTION
# -----------------------
model_choice = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"],
    horizontal=True
)

# -----------------------
# REVIEW INPUT
# -----------------------
review = st.text_area(
    "Enter your movie review here...",
    height=180
)

# -----------------------
# PREDICT
# -----------------------
if st.button("🎥 Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        data = prepare(review)

        if model_choice == "SimpleRNN":
            pred = rnn_model.predict(data)[0][0]

        elif model_choice == "LSTM":
            pred = lstm_model.predict(data)[0][0]

        else:
            pred = gru_model.predict(data)[0][0]

        sentiment = "Positive" if pred > 0.5 else "Negative"

        confidence = pred if pred > 0.5 else (1 - pred)

        positive_prob = pred * 100
        negative_prob = (1 - pred) * 100

        st.markdown("---")

        st.markdown(
            f"""
            <div class='result-box'>
            <h2>Sentiment: {sentiment}</h2>
            <h3>Confidence: {confidence*100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        if sentiment == "Positive":

            st.success(
            f"🎉 Positive Review Detected ({confidence*100:.2f}%)"
            )
            st.balloons()

        else:

            st.error(
                f"😞 Negative Review Detected ({confidence*100:.2f}%)"
            )
            

        st.subheader("Prediction Probabilities")

        st.progress(int(positive_prob))
        st.write(f"Positive Probability: {positive_prob:.2f}%")

        st.progress(int(negative_prob))
        st.write(f"Negative Probability: {negative_prob:.2f}%")

# -----------------------
# COMPARE ALL MODELS
# -----------------------
st.markdown("---")
st.subheader("🎭 Compare All Models")

if st.button("Compare Predictions"):

    if review.strip() != "":

        data = prepare(review)

        rnn_pred = rnn_model.predict(data)[0][0]
        lstm_pred = lstm_model.predict(data)[0][0]
        gru_pred = gru_model.predict(data)[0][0]

        import pandas as pd

        df = pd.DataFrame({
            "Model":["SimpleRNN","LSTM","GRU"],
            "Confidence":[
                round(max(rnn_pred,1-rnn_pred)*100,2),
                round(max(lstm_pred,1-lstm_pred)*100,2),
                round(max(gru_pred,1-gru_pred)*100,2)
            ],
            "Sentiment":[
                "Positive" if rnn_pred>0.5 else "Negative",
                "Positive" if lstm_pred>0.5 else "Negative",
                "Positive" if gru_pred>0.5 else "Negative"
            ]
        })

        st.dataframe(df, use_container_width=True)

        st.bar_chart(
            df.set_index("Model")["Confidence"]
        )