import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="wide"
)

# LOAD MODEL
model = tf.keras.models.load_model("titanic_ann_model.h5")

# CUSTOM CSS
st.markdown("""
<style>

body {
    background-color: #050816;
}

.main {
    background: linear-gradient(135deg,#050816,#0f172a);
    color: white;
}

h1, h2, h3 {
    color: #00f7ff;
    text-shadow: 0px 0px 15px #00f7ff;
}

.stButton>button {
    background: linear-gradient(90deg,#00f7ff,#ff00ff);
    color: white;
    border-radius: 15px;
    height: 3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
    box-shadow: 0px 0px 20px #00f7ff;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 35px #ff00ff;
}

.css-1d391kg {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,247,255,0.4);
    margin-bottom: 20px;
}

.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(255,0,255,0.3);
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="card">
    <h1 style='text-align:center;'>
        🚢 TITANIC SURVIVAL AI SYSTEM
    </h1>
    <h3 style='text-align:center; color:#ff00ff;'>
        Deep Learning Based Passenger Survival Prediction
    </h3>
</div>
""", unsafe_allow_html=True)

# DESCRIPTION
st.markdown("""
<div class="card">
<h3>🤖 Project Description</h3>

This AI-powered web application predicts passenger survival chances
using an Artificial Neural Network (ANN) built with TensorFlow & Keras.

Features:
- Deep Learning Prediction
- Real-time Inference
- Neon AI Dashboard
- Survival Probability Visualization
- Streamlit Cloud Deployment

</div>
""", unsafe_allow_html=True)

# INPUT AREA
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

st.markdown("</div>", unsafe_allow_html=True)

# PREPROCESSING
# SAME AS TRAINING

pclass_norm = (pclass - 1) / (3 - 1)
age_norm = age / 100
fare_norm = fare / 150

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# PREDICTION BUTTON
if st.button("🔮 Predict Survival"):

    prediction = model.predict(input_data)[0][0]

    survived_prob = float(prediction)
    nonsurvived_prob = 1 - survived_prob

    if survived_prob > 0.5:
        result = "✅ SURVIVED"
    else:
        result = "❌ NOT SURVIVED"

    # OUTPUT AREA
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Prediction</h3>
            <h2>{result}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Survival Probability</h3>
            <h2>{survived_prob:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <h3>Confidence</h3>
            <h2>{survived_prob*100:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # VISUALIZATION
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h3>📊 Probability Visualization</h3>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6,4))

    labels = ["Survived", "Not Survived"]
    values = [survived_prob, nonsurvived_prob]

    ax.bar(labels, values)

    ax.set_ylim([0,1])

    st.pyplot(fig)

# FOOTER
st.markdown("""
<br><br>
<div style='text-align:center;color:gray;'>
Developed using Streamlit + TensorFlow + Deep Learning
</div>
""", unsafe_allow_html=True)