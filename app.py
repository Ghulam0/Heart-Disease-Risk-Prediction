import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the saved model
with open("heart_disease_model.pkl", "rb") as f:
    model, scaler, label_encoders = pickle.load(f)

st.title("Heart Disease Prediction App")
st.sidebar.header("Enter Patient Details")

# Collect user input fields
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
cholesterol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=400, value=200)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
heart_rate = st.sidebar.number_input("Heart Rate", min_value=40, max_value=200, value=70)
smoking = st.sidebar.selectbox("Smoking", ['Never', 'Former', 'Current'])
alcohol_intake = st.sidebar.selectbox("Alcohol Intake", ['None', 'Light', 'Moderate', 'Heavy'])
exercise_hours = st.sidebar.number_input("Exercise Hours per Week", min_value=0, max_value=20, value=3)
family_history = st.sidebar.selectbox("Family History", ['No', 'Yes'])
diabetes = st.sidebar.selectbox("Diabetes", ['No', 'Yes'])
obesity = st.sidebar.selectbox("Obesity", ['No', 'Yes'])
stress_level = st.sidebar.slider("Stress Level", min_value=1, max_value=10, value=5)
blood_sugar = st.sidebar.number_input("Blood Sugar", min_value=50, max_value=300, value=100)
angina = st.sidebar.selectbox("Exercise Induced Angina", ['No', 'Yes'])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])

# When button is clicked, process input and make prediction
if st.sidebar.button("Predict Heart Disease"):
    st.write("Button clicked! Preparing input data...")  # Debugging

    # Prepare input data
    input_data = pd.DataFrame([[
        age, gender, cholesterol, blood_pressure, heart_rate, smoking, alcohol_intake,
        exercise_hours, family_history, diabetes, obesity, stress_level, blood_sugar,
        angina, chest_pain
    ]], columns=[
        "Age", "Gender", "Cholesterol", "Blood Pressure", "Heart Rate", "Smoking",
        "Alcohol Intake", "Exercise Hours", "Family History", "Diabetes", "Obesity",
        "Stress Level", "Blood Sugar", "Exercise Induced Angina", "Chest Pain Type"
    ])

    # Fixing dtype issue by ensuring all values are proper data types
    input_data = input_data.astype(str)  # Convert all values to string (avoids dtype issues)
    
    # Convert numerical columns back to their proper dtype
    numeric_cols = ["Age", "Cholesterol", "Blood Pressure", "Heart Rate", "Exercise Hours", "Stress Level", "Blood Sugar"]
    input_data[numeric_cols] = input_data[numeric_cols].apply(pd.to_numeric)

    # Encode categorical data safely
    for col in input_data.columns:
        if input_data[col].dtype == 'object':  # Only encode categorical columns
            input_data[col] = input_data[col].fillna("Unknown")  # Handle missing values
            
            # Ensure the encoder exists
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                label_encoders[col].fit(list(input_data[col].unique()) + ["Unknown"])  

            # Add "Unknown" if missing
            if "Unknown" not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")

            # Replace unseen values with "Unknown"
            input_data[col] = input_data[col].apply(lambda x: x if x in label_encoders[col].classes_ else "Unknown")

            # Transform categorical column
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale numeric data
    input_data_scaled = scaler.transform(input_data)

    st.write("Data prepared! Making prediction...")  # Debugging

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

    st.write("Prediction complete!")  # Debugging
    st.subheader("Prediction Result")
    st.write(result)
