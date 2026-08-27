import os
import sys
import warnings

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
    average_precision_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# PATH SETUP
# ============================================================

# Project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Allow importing features.py from the same src folder
sys.path.append(
    os.path.join(
        PROJECT_ROOT,
        "src"
    )
)

from features import create_features


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

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
    "credit_risk_model.joblib"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "model_metadata.joblib"
)

# Change this to your threshold from Phase 8
FINAL_THRESHOLD = 0.50


warnings.filterwarnings("ignore")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df = create_features(df)

print(f"Dataset shape after feature engineering: {df.shape}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("PREPARING FEATURES AND TARGET")
print("=" * 60)

X = df.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ],
    errors="ignore"
)

y = df["TARGET"]

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\nTarget distribution:")

print(
    y.value_counts(
        normalize=True
    )
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

print("\n" + "=" * 60)
print("IDENTIFYING FEATURE TYPES")
print("=" * 60)

numerical_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# NUMERICAL PREPROCESSING
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


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

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


# ============================================================
# COMBINE PREPROCESSING
# ============================================================

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
    ],
    remainder="drop"
)


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING CLASS IMBALANCE")
print("=" * 60)

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

print(f"Negative samples: {negative_count}")
print(f"Positive samples: {positive_count}")
print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING XGBOOST MODEL")
print("=" * 60)

# IMPORTANT:
# Replace these hyperparameters with your best parameters
# from Phase 7 RandomizedSearchCV if they are different.

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

final_pipeline = Pipeline(
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
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

final_pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predicted probabilities
y_prob = final_pipeline.predict_proba(
    X_test
)[:, 1]

# Predictions using final threshold
y_pred = (
    y_prob >= FINAL_THRESHOLD
).astype(int)


# ROC-AUC
roc_auc = roc_auc_score(
    y_test,
    y_prob
)

# PR-AUC
pr_auc = average_precision_score(
    y_test,
    y_prob
)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print(
    f"\nFinal Classification Threshold: "
    f"{FINAL_THRESHOLD}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(
    final_pipeline,
    MODEL_PATH
)

print(f"\nModel saved to:\n{MODEL_PATH}")


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "threshold": FINAL_THRESHOLD,

    "feature_columns": X.columns.tolist(),

    "numerical_features": numerical_features,

    "categorical_features": categorical_features,

    "model_name": "XGBoost Credit Risk Model",

    "roc_auc": float(roc_auc),

    "pr_auc": float(pr_auc),

    "random_state": RANDOM_STATE,

    "test_size": TEST_SIZE,

    "scale_pos_weight": float(
        scale_pos_weight
    )
}


joblib.dump(
    metadata,
    METADATA_PATH
)

print(
    f"\nMetadata saved to:\n"
    f"{METADATA_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Model: XGBoost Credit Risk Model")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Threshold: {FINAL_THRESHOLD}")

print("\nFiles created:")

print(
    "1. models/credit_risk_model.joblib"
)

print(
    "2. models/model_metadata.joblib"
)

print("\nProject model is ready for prediction.")