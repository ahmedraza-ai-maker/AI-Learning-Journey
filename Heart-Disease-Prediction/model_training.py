import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib


# Load dataset
data = pd.read_csv("heart.csv", sep="\t")

# Features and target
X = data.drop("target", axis=1)
y = data["target"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


# Train and evaluate models
results = []

for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    else:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    cv_scores = cross_val_score(
        model,
        X_train_scaled if name != "Random Forest" else X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_accuracy = cv_scores.mean()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "CV Accuracy": cv_accuracy
    })


# Show results
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.round(3))


rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(round(grid_search.best_score_, 3))


best_model = grid_search.best_estimator_

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

print("\nTuned Random Forest Test Results:")
print("Accuracy:", round(test_accuracy, 3))
print("Precision:", round(test_precision, 3))
print("Recall:", round(test_recall, 3))
print("F1 Score:", round(test_f1, 3))


cm = confusion_matrix(y_test, test_predictions)

print("\nConfusion Matrix:")
print(cm)

import matplotlib.pyplot as plt

feature_importance = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

joblib.dump(best_model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")