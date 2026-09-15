# job-salary-classification1
#  Job Salary Classification

A machine learning project that predicts **salary categories — Low, Medium, or High —** based on a candidate's education, experience, seniority, industry, location, and skill count.

The project uses **Python, Pandas, Scikit-learn, Random Forest, Joblib, and Streamlit** and includes both a command-line prediction script and an interactive web application.

> **Note:** The included dataset is synthetic and intended for educational, academic, and machine-learning demonstration purposes. It should not be used for real-world salary decisions.

---

##  Project Overview

Salary classification is a useful machine learning problem for exploring how professional and career-related attributes can be used to predict compensation bands.

This project builds an end-to-end classification pipeline that:

* Loads and preprocesses job/candidate data
* Encodes categorical variables using `OneHotEncoder`
* Uses numerical features directly
* Trains a `RandomForestClassifier`
* Evaluates the model using accuracy and a classification report
* Generates a confusion matrix
* Saves the trained model using Joblib
* Provides individual salary predictions
* Supports batch candidate classification through CSV files
* Provides an interactive Streamlit dashboard

---

##  Prediction Classes

The model predicts one of three salary categories:

| Class         | Description                                   |
| ------------- | --------------------------------------------- |
|  **High**   | Higher compensation bracket                   |
|  **Medium** | Competitive / market-mid compensation bracket |
|  **Low**    | Entry / foundational compensation bracket     |

---

##  Dataset

The project uses a synthetic dataset containing **1,500 records**.

### Features

| Feature            | Type        | Description                      |
| ------------------ | ----------- | -------------------------------- |
| `education`        | Categorical | Highest education level          |
| `experience_years` | Numerical   | Years of professional experience |
| `job_level`        | Categorical | Employee seniority level         |
| `industry`         | Categorical | Industry sector                  |
| `location_type`    | Categorical | City/location tier               |
| `skill_count`      | Numerical   | Number of specialized skills     |
| `salary_class`     | Target      | Low, Medium, or High             |

### Dataset Distribution

* **Medium:** 773 records
* **Low:** 449 records
* **High:** 278 records

The dataset contains examples across education levels, seniority levels, industries, location tiers, experience levels, and skill counts.

---

##  Machine Learning Approach

### Algorithm

The project uses:

```text
RandomForestClassifier
n_estimators = 250
class_weight = "balanced"
random_state = 42
```

Random Forest was selected because it works well with nonlinear relationships and mixed feature types after preprocessing.

### Preprocessing

Categorical variables are transformed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

The following features are one-hot encoded:

* `education`
* `job_level`
* `industry`
* `location_type`

Numerical features are passed through directly:

* `experience_years`
* `skill_count`

The preprocessing and classifier are combined into a single Scikit-learn `Pipeline`, making inference consistent with training.

---

##  Model Pipeline

```text
Raw Candidate Data
        │
        ▼
┌──────────────────────┐
│ Feature Separation   │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
Categorical   Numerical
 Features      Features
     │           │
     ▼           ▼
OneHotEncoder  Passthrough
     │           │
     └─────┬─────┘
           ▼
   Random Forest
   Classifier
           │
           ▼
 Low / Medium / High
```

---

##  Model Evaluation

The training script uses an **80/20 stratified train-test split**.

The project reports an overall test accuracy of approximately:

**86%**

The evaluation also includes:

* Accuracy
* Classification report
* Precision
* Recall
* F1-score
* Confusion matrix

A confusion matrix is generated automatically and saved as:

```text
confusion_matrix.png
```

---

##  Streamlit Application

The project includes an interactive Streamlit dashboard for salary-band prediction.

### Individual Candidate Prediction

Users can enter:

*  Education
*  Years of experience
*  Job seniority
*  Industry
*  Location tier
*  Specialized skill count

The application returns:

* Predicted salary class
* Prediction confidence
* Probability for each salary class
* Compensation-related feature insights

### Batch Talent Screening

The application also supports uploading a CSV containing multiple candidates.

It can:

* Classify the entire candidate dataset
* Display salary-band counts
* Calculate prediction confidence
* Filter results by salary class
* Filter results by industry
* Export classified candidates as CSV

### Model Performance

The application includes a dedicated section displaying:

* Model architecture
* Preprocessing approach
* Accuracy benchmark
* Confusion matrix

---

##  Project Structure

```text
Job_Salary_Classification_Sklearn/
│
├── data/
│   └── job_salary_dataset.csv
│
├── app.py
├── train_model.py
├── predict.py
├── job_salary_classifier.pkl
├── confusion_matrix.png
├── requirements.txt
└── README.md
```

### File Description

**`train_model.py`**
Trains the Random Forest model, evaluates performance, saves the trained model, and generates the confusion matrix.

**`predict.py`**
Demonstrates how to load the trained model and make a prediction for a single candidate.

**`app.py`**
Runs the interactive Streamlit web application.

**`job_salary_classifier.pkl`**
Serialized Scikit-learn model pipeline saved using Joblib.

**`job_salary_dataset.csv`**
Synthetic dataset used for model training and demonstration.

**`confusion_matrix.png`**
Visualization of model predictions against the actual test labels.

**`requirements.txt`**
Python dependencies required to run the project.

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Job_Salary_Classification_Sklearn.git
cd Job_Salary_Classification_Sklearn
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Train the Model

To train the model from scratch:

```bash
python train_model.py
```

This will:

1. Load the dataset
2. Separate features and target
3. Encode categorical features
4. Split the data into training and testing sets
5. Train the Random Forest classifier
6. Evaluate the model
7. Save the trained model
8. Generate the confusion matrix

Generated files:

```text
job_salary_classifier.pkl
confusion_matrix.png
```

---

##  Make a Prediction

Run:

```bash
python predict.py
```

The script loads the trained model and predicts the salary class for an example candidate.

Example input:

```text
Education: Master
Experience: 8 years
Job Level: Senior
Industry: IT
Location: Tier 1 City
Skills: 8
```

The model returns the predicted salary class along with probabilities for each class.

---

##  Run the Streamlit App

Start the application with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

The application provides three main sections:

```text
 Candidate Salary Band Evaluation
Batch Talent Screening
 Model Performance & Confusion Matrix
```

---

##  Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical operations
* **Scikit-learn** — machine learning
* **Random Forest** — classification algorithm
* **OneHotEncoder** — categorical feature encoding
* **Joblib** — model serialization
* **Matplotlib** — visualization
* **Streamlit** — interactive web application

---

##  Key Machine Learning Concepts Demonstrated

This project demonstrates several practical machine-learning concepts:

* Supervised learning
* Multiclass classification
* Feature preprocessing
* Categorical encoding
* Numerical feature handling
* Scikit-learn pipelines
* Train/test splitting
* Stratified sampling
* Random Forest classification
* Class balancing
* Probability prediction
* Model evaluation
* Confusion matrices
* Model serialization
* Batch inference
* Interactive ML deployment

---

##  End-to-End Workflow

```text
Dataset
   │
   ▼
Data Preparation
   │
   ▼
Feature Engineering / Encoding
   │
   ▼
Train-Test Split
   │
   ▼
Random Forest Training
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Classification Report
   └── Confusion Matrix
   │
   ▼
Save Model with Joblib
   │
   ▼
Prediction
   │
   ├── Single Candidate
   └── Batch CSV
   │
   ▼
Streamlit Dashboard
```

---

##  Limitations

This project is primarily a **machine-learning learning/demo project**.

The dataset is synthetic, so the model should not be interpreted as an accurate representation of real-world salary markets.

Actual compensation can depend on many additional factors, including:

* Company
* Country and region
* Economic conditions
* Job demand
* Industry specialization
* Negotiation
* Performance
* Benefits and equity
* Supply and demand

Therefore, model predictions should not be used as the sole basis for employment or compensation decisions.

---

##  Future Improvements

Potential improvements include:

* Add real-world salary datasets
* Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Compare Random Forest with XGBoost, LightGBM, and Logistic Regression
* Add feature importance visualization
* Add SHAP-based explainability
* Perform cross-validation
* Add model versioning
* Add automated testing
* Add Docker support
* Deploy the Streamlit application
* Add salary regression to predict an actual salary range
* Add geographic salary information
* Add industry-specific models

---

##  Author

**Your Name**

If you found this project useful, consider starring the repository.

---

##  License

This project is intended for educational and demonstration purposes. Add an appropriate open-source license such as **MIT** if you intend to distribute the project under that license.
# job-salary-classification1
Machine learning project using Scikit-learn and Random Forest to classify job salaries into Low, Medium, and High bands, with an interactive Streamlit dashboard and batch prediction support.
