import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import pickle

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("ObesityDataSet.csv")
    return df

def preprocess_data(df):
    df = df.dropna().copy()
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != 'NObeyesdad':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, label_encoders

# --- Main App ---
st.title("🏥 Obesity Prediction ML App")

st.markdown("This app uses an SVM model to predict obesity levels based on lifestyle and physical metrics.")

# Custom labels for user-friendly sliders
display_names = {
    "Age": "Age",
    "Gender": "Gender",
    "Height": "Height (in meters)",
    "Weight": "Weight (in kg)",
    "CALC": "CALC – Alcohol Consumption Frequency",
    "FAVC": "FAVC – Frequent High Calorie Food",
    "FCVC": "FCVC – Frequency of Vegetable Consumption",
    "NCP": "NCP – Number of Main Meals",
    "SCC": "SCC – Snacking Between Meals",
    "SMOKE": "SMOKE – Smoking Habit",
    "CH2O": "CH2O – Daily Water Intake (liters)",
    "family_history_with_overweight": "Family History of Overweight",
    "FAF": "FAF – Physical Activity Frequency",
    "TUE": "TUE – Time Spent on Technology Devices"
}

# Load dataset
data = load_data()

if st.checkbox("Show Raw Dataset"):
    st.dataframe(data)

# Basic EDA
if st.checkbox("Show Summary & Class Distribution"):
    st.write("Summary Statistics")
    st.write(data.describe())
    st.write("Class Distribution")
    st.bar_chart(data["NObeyesdad"].value_counts())

# Preprocess
X_scaled, y, scaler, label_encoders = preprocess_data(data)

# Train model
model = SVC()
model.fit(X_scaled, y)

st.success("Model trained successfully!")

# User Input for Prediction
st.header("📊 Predict Obesity Level")

input_data = {}
for col in data.drop(columns=["NObeyesdad"]).columns:
    if data[col].dtype == "object":
        options = data[col].unique().tolist()
        label = display_names.get(col, col)
        input_data[col] = st.selectbox(f"{label}", options)
    else:
        label = display_names.get(col, col)
        input_data[col] = st.slider(f"{label}", float(data[col].min()), float(data[col].max()), float(data[col].mean()))

# Encode and scale input
input_df = pd.DataFrame([input_data])

for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col])

input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.subheader(f"🏁 Predicted Obesity Level: {prediction}")
