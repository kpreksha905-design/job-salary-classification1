import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("data/job_salary_dataset.csv")

X = df.drop("salary_class", axis=1)
y = df["salary_class"]

categorical_features = ["education", "job_level", "industry", "location_type"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "job_salary_classifier.pkl")

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Job Salary Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nModel saved as job_salary_classifier.pkl")
