# ❤️ Heart Disease Prediction

A Machine Learning web application that predicts the possible presence of heart disease based on patient health information.

## 🚀 Live Demo

https://heart-disease-prediction-ahmed.streamlit.app/

## 📌 Project Overview

This project uses Machine Learning to analyze patient health-related features and generate a prediction.

The application is built with Streamlit and uses a tuned Random Forest Classifier for prediction.

## ✨ Features

- Patient health information input
- Heart disease prediction
- Prediction probability
- Random Forest Machine Learning model
- Hyperparameter tuning
- Feature importance visualization
- Model evaluation metrics
- Confusion matrix
- Interactive Streamlit interface

## 🧠 Machine Learning

### Model

Random Forest Classifier

### Data Processing

- Train-test split
- Feature scaling using StandardScaler
- Hyperparameter tuning
- Cross-validation
- Model evaluation

### Hyperparameters

The Random Forest model was tuned using:

- `n_estimators`
- `max_depth`
- `min_samples_split`

Best parameters found during tuning:

```text
n_estimators = 50
max_depth = None
min_samples_split = 5