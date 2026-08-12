import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path


st.title("❤️ Heart Disease Prediction")

st.write("Enter the patient's information below to get a model prediction.")

# Load dataset
data_path = Path(__file__).parent / "heart.csv"
data = pd.read_csv(data_path, sep="\t")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=220, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):

    patient = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=X.columns)

    patient = scaler.transform(patient)

    prediction = model.predict(patient)[0]

    if prediction == 1:
        st.warning("The model predicts a possible presence of heart disease.")
    else:
        st.success("The model predicts a lower likelihood of heart disease.")

st.caption("This project is for educational purposes only and is not a medical diagnosis.")