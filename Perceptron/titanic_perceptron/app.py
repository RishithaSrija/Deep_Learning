import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model("titanic_ann_model.h5")

# =========================================================
# HEADER SECTION
# =========================================================

st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF;'>
    🚢 Titanic Survival Prediction System
    </h1>
    <h3 style='text-align: center; color: gray;'>
    Deep Learning Based Passenger Survival Prediction
    </h3>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# PROJECT DESCRIPTION
# =========================================================

st.subheader("📘 Project Description")

st.write("""
This application predicts whether a Titanic passenger would survive or not
using an Artificial Neural Network (ANN) model built with TensorFlow/Keras.

The model uses:
- Passenger Class
- Age
- Fare

The application demonstrates:
- Deep Learning Prediction
- TensorFlow Model Deployment
- Real-time Inference using Streamlit
""")

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

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
        max_value=600.0,
        value=120.0
    )

st.divider()

# =========================================================
# DATA PREPROCESSING
# =========================================================

def normalize_inputs(pclass, age, fare):

    # Min-Max Normalization

    pclass_norm = (pclass - 1) / (3 - 1)
    age_norm = (age - 1) / (80 - 1)
    fare_norm = fare / 600

    return np.array([[pclass_norm, age_norm, fare_norm]])

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Predict Survival"):

    # Normalize Inputs
    input_data = normalize_inputs(pclass, age, fare)

    # Model Prediction
    prediction = model.predict(input_data)

    probability = float(prediction[0][0])

    # Prediction Logic
    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    confidence = probability * 100

    st.divider()

    # =====================================================
    # OUTPUT SECTION
    # =====================================================

    st.subheader("📊 Prediction Output")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="Prediction Result",
            value=result
        )

    with c2:
        st.metric(
            label="Survival Probability",
            value=f"{probability:.2f}"
        )

    with c3:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    st.divider()

    # =====================================================
    # VISUALIZATION SECTION
    # =====================================================

    st.subheader("📈 Survival Probability Visualization")

    survive_prob = probability
    not_survive_prob = 1 - probability

    fig, ax = plt.subplots()

    labels = ["Survived", "Not Survived"]
    values = [survive_prob, not_survive_prob]

    ax.bar(labels, values)

    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")

    st.pyplot(fig)

    # =====================================================
    # PIE CHART
    # =====================================================

    fig2, ax2 = plt.subplots()

    ax2.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    ax2.set_title("Survival Distribution")

    st.pyplot(fig2)
