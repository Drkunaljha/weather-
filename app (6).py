import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("weather_knn.pkl")

st.title("Weather Prediction using KNN")
st.write("College Mini Project – Machine Learning")

# Input sliders
temperature = st.slider("Temperature (°C)", 20, 40, 26)
humidity = st.slider("Humidity (%)", 50, 90, 75)

# Prediction button
if st.button("Predict Weather"):
    prediction = model.predict([[temperature, humidity]])[0]

    if prediction == 0:
        st.success("☀️ Predicted Weather: Sunny")
    else:
        st.success("🌧️ Predicted Weather: Rainy")
