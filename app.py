import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# -----------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -----------------------
# MOVIE LOGO
# -----------------------
st.markdown(
    """
    <div style='text-align:center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/4221/4221484.png'
             width='120'>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# LOAD CSS
# -----------------------
with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

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


# IMPORTANT
rnn_model, lstm_model, gru_model = load_models()

# -----------------------
# LOAD TOKENIZER
# -----------------------
with open("imdb_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# -----------------------
# TEXT PREPROCESSING
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
# ANALYZE REVIEW
# -----------------------
if st.button("🎥 Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        data = prepare(review)

        if model_choice == "SimpleRNN":
            pred = float(
                rnn_model.predict(data, verbose=0)[0][0]
            )

        elif model_choice == "LSTM":
            pred = float(
                lstm_model.predict(data, verbose=0)[0][0]
            )

        else:
            pred = float(
                gru_model.predict(data, verbose=0)[0][0]
            )

        sentiment = (
            "Positive"
            if pred > 0.5
            else "Negative"
        )

        confidence = (
            pred
            if pred > 0.5
            else (1 - pred)
        )

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

        # Effects
        if sentiment == "Positive":

            st.success(
                f"🎉 Positive Review Detected ({confidence*100:.2f}%)"
            )

            st.balloons()

        else:

            st.error(
                f"😞 Negative Review Detected ({confidence*100:.2f}%)"
            )

            st.snow()

        # Probability Section
        st.subheader("📊 Prediction Probabilities")

        st.write(
            f"Positive Probability: {positive_prob:.2f}%"
        )
        st.progress(
            min(int(positive_prob), 100)
        )

        st.write(
            f"Negative Probability: {negative_prob:.2f}%"
        )
        st.progress(
            min(int(negative_prob), 100)
        )

# -----------------------
# COMPARE ALL MODELS
# -----------------------
st.markdown("---")

st.subheader(
    "🎭 Compare Predictions from All Models"
)

if st.button("Compare Predictions"):

    if review.strip() == "":

        st.warning(
            "Please enter a review first."
        )

    else:

        data = prepare(review)

        rnn_pred = float(
            rnn_model.predict(data, verbose=0)[0][0]
        )

        lstm_pred = float(
            lstm_model.predict(data, verbose=0)[0][0]
        )

        gru_pred = float(
            gru_model.predict(data, verbose=0)[0][0]
        )

        comparison_df = pd.DataFrame({

            "Model": [
                "SimpleRNN",
                "LSTM",
                "GRU"
            ],

            "Sentiment": [
                "Positive" if rnn_pred > 0.5 else "Negative",
                "Positive" if lstm_pred > 0.5 else "Negative",
                "Positive" if gru_pred > 0.5 else "Negative"
            ],

            "Confidence (%)": [
                round(max(rnn_pred, 1-rnn_pred)*100, 2),
                round(max(lstm_pred, 1-lstm_pred)*100, 2),
                round(max(gru_pred, 1-gru_pred)*100, 2)
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.subheader(
            "📈 Confidence Comparison"
        )

        st.bar_chart(
            comparison_df.set_index("Model")[
                "Confidence (%)"
            ]
        )