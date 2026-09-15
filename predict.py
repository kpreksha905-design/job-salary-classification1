import joblib
import pandas as pd

model = joblib.load("job_salary_classifier.pkl")

sample = pd.DataFrame([{
    "education": "Master",
    "experience_years": 8,
    "job_level": "Senior",
    "industry": "IT",
    "location_type": "Tier 1 City",
    "skill_count": 8
}])

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("Predicted Salary Class:", prediction)
print("\nClass probabilities:")
for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")
