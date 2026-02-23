import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("weather_knn.pkl")

st.set_page_config(page_title="Weather Prediction App", layout="centered")

st.title("🌦️ Weather Prediction using KNN")
st.write("College Mini Project – Machine Learning Model")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("🌡 Temperature (°C)", 20, 40, 26)

with col2:
    humidity = st.slider("💧 Humidity (%)", 50, 90, 75)

# Show input summary
st.subheader("📌 Input Summary")
st.info(f"Temperature: {temperature} °C | Humidity: {humidity} %")

# Prediction
if st.button("🔍 Predict Weather"):

    input_data = np.array([[temperature, humidity]])
    prediction = model.predict(input_data)[0]

    # Probability (if available)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        confidence = np.max(prob) * 100
    else:
        confidence = None

    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.success("☀️ Weather: Sunny")
    else:
        st.success("🌧️ Weather: Rainy")

    if confidence:
        st.metric("Confidence Level", f"{confidence:.2f}%")

    # Simple Visualization
    st.subheader("📍 Weather Condition Plot")

    fig, ax = plt.subplots()
    ax.scatter(temperature, humidity, color="red", s=100)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Current Input Position")
    ax.set_xlim(20, 40)
    ax.set_ylim(50, 90)
    st.pyplot(fig)
