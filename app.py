import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os
# --------------------------
# App Configuration
# --------------------------
st.set_page_config(page_title="Anomaly Detection with GenAI", layout="wide")
st.title("🔍 Sensor Anomaly Detection using Autoencoder + GenAI")

# --------------------------
# Load Environment & GenAI
# --------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --------------------------
# Session State (for graph toggle)
# --------------------------
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False

# --------------------------
# Upload CSV
# --------------------------
uploaded_file = st.file_uploader("📤 Upload Sensor CSV File", type=["csv"])

if uploaded_file:
    # --------------------------
    # Load Data
    # --------------------------
    df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------
    # Feature Selection
    # --------------------------
    feature_cols = [
        col for col in df.columns
        if col not in ["timestamp", "anomaly", "anomaly_str"]
    ]

    X = df[feature_cols]

    # --------------------------
    # Scaling
    # --------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    st.success(f"Data scaled → Shape: {X_scaled.shape}")

    # --------------------------
    # Build Autoencoder
    # --------------------------
    input_dim = X_scaled.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)
    decoded = Dense(16, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")

    # --------------------------
    # Train Autoencoder
    # --------------------------
    with st.spinner("🧠 Training Autoencoder..."):
        autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            shuffle=True,
            verbose=0
        )

    st.success("Autoencoder training completed")

    # --------------------------
    # Reconstruction Error
    # --------------------------
    recon = autoencoder.predict(X_scaled, verbose=0)
    loss = np.mean(np.abs(recon - X_scaled), axis=1)

    df["reconstruction_error"] = loss
    threshold = np.percentile(loss, 95)
    df["anomaly"] = df["reconstruction_error"] > threshold

    # --------------------------
    # Severity Classification
    # --------------------------
    df["severity"] = np.where(
        df["reconstruction_error"] > np.percentile(loss, 99),
        "Critical",
        "Warning"
    )

    st.write("### 🚨 Total Anomalies Detected:", df["anomaly"].sum())

    # --------------------------
    # Toggle Graph Button
    # --------------------------
    if st.button("📈 Open / Close Anomaly Graph"):
        st.session_state.show_graph = not st.session_state.show_graph

    if st.session_state.show_graph:
        fig, ax = plt.subplots()
        ax.plot(df["timestamp"], df["reconstruction_error"], label="Reconstruction Error")
        ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
        ax.scatter(
            df[df["anomaly"]]["timestamp"],
            df[df["anomaly"]]["reconstruction_error"],
            color="orange",
            label="Anomalies"
        )
        ax.legend()
        ax.set_title("Anomaly Detection (Toggle View)")
        st.pyplot(fig)

    # --------------------------
    # Feature Deviation Analysis
    # --------------------------
    anomaly_df = df[df["anomaly"]]

    deviation_summary = {}
    for col in feature_cols:
        deviation_summary[col] = {
            "normal_mean": round(df[~df["anomaly"]][col].mean(), 3),
            "anomaly_mean": round(anomaly_df[col].mean(), 3),
            "percent_change": round(
                (
                    (anomaly_df[col].mean() - df[~df["anomaly"]][col].mean())
                    / (df[~df["anomaly"]][col].mean() + 1e-6)
                ) * 100,
                2
            )
        }

    # --------------------------
    # GenAI Root Cause Button
    # --------------------------
    if st.button("🤖 Explain Anomalies (Root Cause Analysis)"):
        prompt = f"""
        You are a senior industrial reliability engineer.

        Anomalies were detected using an autoencoder that learned normal sensor behavior.

        Detection threshold (reconstruction error): {round(threshold, 4)}

        Sensor-wise deviation summary (normal vs anomaly):
        {deviation_summary}

        Severity distribution:
        {df['severity'].value_counts().to_dict()}

        TASKS:
        1. Rank the sensors causing anomalies by impact
        2. Explain the most likely physical or operational reason
        3. Mention whether the issue is gradual degradation or sudden failure
        4. Suggest specific corrective actions
        5. Clearly justify severity level (Warning or Critical)

        Respond in concise bullet points.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        st.subheader("🧠 GenAI Root Cause Explanation")
        st.write(response.choices[0].message.content)
