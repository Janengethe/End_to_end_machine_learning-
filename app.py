import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


red_model = joblib.load("red_best_model.pkl")


st.title("Wine Prediction")

# Define input fields for features
fixed_acidity = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=4.0, step=1.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.1, max_value=1.6, value=0.1, step=0.1)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
residual_sugar = st.number_input("Residual Sugar", min_value=0.9, max_value=15.5, value=0.9, step=1.0)

chlorides = st.number_input("Chlorides", min_value=0.0, max_value=0.7, value=0.4, step=0.1)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=1, max_value=72, value=10, step=1)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=6, max_value=290, value=20, step=1)
density = st.number_input("Density", min_value=0.9, max_value=1.5, value=0.9, step=0.01)
pH = st.number_input("pH", min_value=2.7, max_value=4.5, value=2.9, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.3, max_value=2.0, value=0.9, step=0.01)
alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=9.0, step=0.01)

# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "fixed_acidity": [fixed_acidity],
            "volatile_acidity": [volatile_acidity],
            "citric_acid": [citric_acid],
            "residual_sugar": [residual_sugar],
            "chlorides": [chlorides],
            "free_sulfur_dioxide": [free_sulfur_dioxide],
            "total_sulfur_dioxide": [total_sulfur_dioxide],
            "density": [density],
            "pH": [pH],
            "sulphates": [sulphates],
            "alcohol": [alcohol]
            
    }
    )

    # Scale input data using the same scaler used during training
    # scaler = StandardScaler()
    # input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the trained model
    prediction = red_model.predict(input_data)

    # Round the prediction to a whole number
    rounded_prediction = int(round(prediction[0]))
    
    # Provide context and display the prediction result
    st.write("Predicted Wine Quality:", prediction[0])
    st.write("Approximate Wine Quality Rating:", rounded_prediction, "out of 10")

    # Visualization of prediction
    plt.bar(["Predicted Quality", "Typical Range"], [prediction[0], 5.5], color=['blue', 'gray'])
    plt.xlabel("Category")
    plt.ylabel("Quality Rating")
    plt.title("Predicted Wine Quality")
    st.pyplot(plt)



