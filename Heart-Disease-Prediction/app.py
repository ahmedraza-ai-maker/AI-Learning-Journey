import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Load paths
base_path = Path(__file__).parent

# Load dataset for column names
data_path = base_path / "heart.csv"
data = pd.read_csv(data_path, sep="\t")

X = data.drop("target", axis=1)

# Load trained model and scaler
model = joblib.load(base_path / "heart_model.pkl")
scaler = joblib.load(base_path / "scaler.pkl")

# Title
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter patient information to generate a Machine Learning prediction."
)

st.divider()

# Patient information
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )

    sex = st.selectbox("Sex", [0, 1])

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=220,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )

with col2:
    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia",
        [0, 1, 2, 3]
    )

st.divider()

# Prediction
if st.button("🔍 Predict", use_container_width=True):

    patient = pd.DataFrame(
        [[
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
        ]],
        columns=X.columns
    )

    # Scale patient data
    patient_scaled = scaler.transform(patient)

    # Prediction
    prediction = model.predict(patient_scaled)[0]

    # Probability
    probability = model.predict_proba(patient_scaled)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric(
        "Model Probability",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.warning(
            "⚠️ The model predicts a possible presence of heart disease."
        )
    else:
        st.success(
            "✅ The model predicts a lower likelihood of heart disease."
        )


# Feature Importance
st.divider()
st.subheader("📊 Model Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

st.bar_chart(
    importance_df.set_index("Feature")
)


# Model Evaluation
st.divider()
st.subheader("📈 Model Evaluation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "83.3%")

with col2:
    st.metric("Precision", "75%")

with col3:
    st.metric("Recall", "100%")

with col4:
    st.metric("F1 Score", "85.7%")


# Confusion Matrix
st.subheader("🔲 Confusion Matrix")

confusion_matrix = pd.DataFrame(
    [[2, 1],
     [0, 3]],
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

st.dataframe(confusion_matrix)

st.caption(
    "Educational project only. This Machine Learning prediction "
    "is not a medical diagnosis."
)