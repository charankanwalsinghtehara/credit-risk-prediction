import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "application_train.csv"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "deployment_credit_risk_model.joblib"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "deployment_model_metadata.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
FINAL_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# CREATE DEPLOYMENT FEATURES
# ============================================================

# Convert negative days to positive years

df["AGE_YEARS"] = (
    -df["DAYS_BIRTH"] / 365.25
)

# Handle employment sentinel

df["DAYS_EMPLOYED"] = df[
    "DAYS_EMPLOYED"
].replace(
    365243,
    np.nan
)

df["EMPLOYMENT_YEARS"] = (
    -df["DAYS_EMPLOYED"] / 365.25
)


# ============================================================
# SELECT DEPLOYMENT FEATURES
# ============================================================

deployment_features = [

    # Personal
    "AGE_YEARS",
    "CODE_GENDER",

    # Financial
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",

    # Family
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",

    # Employment
    "EMPLOYMENT_YEARS",
    "NAME_INCOME_TYPE",

    # Education
    "NAME_EDUCATION_TYPE",

    # Family status
    "NAME_FAMILY_STATUS",

    # Housing
    "NAME_HOUSING_TYPE",

    # Occupation
    "OCCUPATION_TYPE",

    # Contact
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL"
]


X = df[deployment_features].copy()

y = df["TARGET"]


print("\nNumber of deployment features:")
print(len(deployment_features))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=[
        "int64",
        "float64"
    ]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category"
    ]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# PREPROCESSING
# ============================================================

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count /
    positive_count
)


print(
    "\nScale positive weight:",
    scale_pos_weight
)


# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(

    objective="binary:logistic",

    eval_metric="auc",

    n_estimators=300,

    learning_rate=0.05,

    max_depth=5,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    gamma=0.1,

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining deployment model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# EVALUATION
# ============================================================

y_prob = pipeline.predict_proba(
    X_test
)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)


print("\nModel Performance")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC: {pr_auc:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


joblib.dump(
    pipeline,
    MODEL_PATH
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "threshold": FINAL_THRESHOLD,

    "feature_columns": deployment_features,

    "numerical_features": numerical_features,

    "categorical_features": categorical_features,

    "roc_auc": float(roc_auc),

    "pr_auc": float(pr_auc),

    "model_name": (
        "Deployment Credit Risk XGBoost"
    )
}


joblib.dump(
    metadata,
    METADATA_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 50)

print(
    "DEPLOYMENT MODEL TRAINING COMPLETE"
)

print("=" * 50)

print(
    "\nModel saved at:"
)

print(MODEL_PATH)

print(
    "\nMetadata saved at:"
)

print(METADATA_PATH)