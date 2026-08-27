# Credit Risk Prediction System

An end-to-end Machine Learning project that predicts the probability
of credit risk for loan applicants.

The project combines Machine Learning, feature engineering,
model evaluation and a Django-based prediction interface.

---

## Project Overview

Credit risk assessment is an important problem in financial services.

The objective of this project is to develop a machine learning model
that predicts whether a loan applicant is likely to present credit risk.

The system produces:

- Risk probability
- Risk classification
- Low Risk
- Medium Risk
- High Risk

---

## Machine Learning Pipeline

The complete pipeline consists of:

1. Data loading
2. Data cleaning
3. Exploratory Data Analysis
4. Feature engineering
5. Missing-value handling
6. Categorical encoding
7. Class imbalance handling
8. Model comparison
9. XGBoost training
10. Hyperparameter tuning
11. Threshold optimization
12. Model serialization
13. Model evaluation
14. Django deployment

---

## Feature Engineering

Important engineered features include:

- AGE_YEARS
- EMPLOYMENT_YEARS

AGE_YEARS is derived from DAYS_BIRTH.

EMPLOYMENT_YEARS is derived from DAYS_EMPLOYED.

---

## Machine Learning Model

The final model uses XGBoost for binary classification.

The model generates a probability representing the predicted
credit risk of an applicant.

A decision threshold is then applied to convert the probability
into a final classification.

---

## Model Evaluation

The model is evaluated using:

- ROC-AUC
- PR-AUC
- Confusion Matrix
- Precision
- Recall
- F1-score
- Feature Importance

Evaluation plots are available inside:

evaluation/

---

## Application

The ML model is deployed through Django.

The application provides:

### Prediction

Users can enter applicant information and receive:

- Risk probability
- Risk level
- Recommendation

### Prediction History

The application displays recent predictions.

### Model Evaluation

The project includes visual evaluation results such as:

- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Feature Importance

---

## Technology Stack

### Programming

- Python

### Data Science

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

### Machine Learning

- Classification
- Feature Engineering
- Hyperparameter Tuning
- Threshold Optimization
- Model Evaluation

### Backend

- Django

### Model Deployment

- Joblib

---

## Project Structure

```text
credit-risk-prediction/
│
├── data/
│   └── raw/
│
├── models/
│
├── evaluation/
│
├── predictor/
│
├── templates/
│   └── predictor/
│
├── src/
│   ├── train.py
│   └── evaluate_model.py
│
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md