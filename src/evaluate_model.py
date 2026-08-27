import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deployment_credit_risk_model.joblib"
)


METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deployment_model_metadata.joblib"
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "application_train.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking project files...")


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not os.path.exists(METADATA_PATH):

    raise FileNotFoundError(
        f"\nMetadata not found:\n{METADATA_PATH}"
    )


if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD METADATA
# ============================================================

print("Loading model metadata...")

metadata = joblib.load(
    METADATA_PATH
)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("Loading original dataset...")

df = pd.read_csv(
    DATA_PATH
)


print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "TARGET"


if TARGET not in df.columns:

    raise ValueError(
        "TARGET column was not found "
        "in the original dataset."
    )


# ============================================================
# GET MODEL FEATURES
# ============================================================

feature_columns = metadata.get(
    "feature_columns"
)


if not feature_columns:

    raise ValueError(
        "feature_columns were not found "
        "inside deployment_model_metadata.joblib"
    )


print(
    f"\nModel expects {len(feature_columns)} features."
)


# ============================================================
# RECREATE ENGINEERED FEATURES
# ============================================================

print(
    "\nRecreating engineered features..."
)


# ------------------------------------------------------------
# AGE_YEARS
# ------------------------------------------------------------

if "AGE_YEARS" in feature_columns:

    if "DAYS_BIRTH" not in df.columns:

        raise ValueError(
            "DAYS_BIRTH is required to create "
            "AGE_YEARS."
        )


    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365.25
    )


    print(
        "Created: AGE_YEARS"
    )


# ------------------------------------------------------------
# EMPLOYMENT_YEARS
# ------------------------------------------------------------

if "EMPLOYMENT_YEARS" in feature_columns:

    if "DAYS_EMPLOYED" not in df.columns:

        raise ValueError(
            "DAYS_EMPLOYED is required to create "
            "EMPLOYMENT_YEARS."
        )


    # Home Credit uses 365243 to represent
    # applicants who are not currently employed.

    df["EMPLOYMENT_YEARS"] = (
        df["DAYS_EMPLOYED"]
        .replace(
            365243,
            0
        )
        / 365.25
    )


    print(
        "Created: EMPLOYMENT_YEARS"
    )


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_features = [

    column

    for column in feature_columns

    if column not in df.columns

]


if missing_features:

    raise ValueError(

        "These model features are still missing:\n"

        + "\n".join(
            str(feature)
            for feature in missing_features
        )

    )


print(
    "\nAll required model features are available."
)


# ============================================================
# CREATE X AND y
# ============================================================

X = df[
    feature_columns
].copy()


y = df[
    TARGET
].copy()


print(
    f"Feature matrix shape: {X.shape}"
)


print(
    f"Target shape: {y.shape}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print(
    "\nCreating evaluation test split..."
)


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    f"Training samples: {len(X_train)}"
)


print(
    f"Test samples: {len(X_test)}"
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)


probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# THRESHOLD
# ============================================================

threshold = metadata.get(
    "threshold",
    0.5
)


print(
    f"Using threshold: {threshold}"
)


predictions = (
    probabilities >= threshold
).astype(int)


# ============================================================
# ROC-AUC
# ============================================================

print(
    "\nCalculating ROC-AUC..."
)


fpr, tpr, _ = roc_curve(

    y_test,

    probabilities

)


roc_auc = auc(

    fpr,

    tpr

)


# ============================================================
# PRECISION-RECALL AUC
# ============================================================

print(
    "Calculating PR-AUC..."
)


precision, recall, _ = precision_recall_curve(

    y_test,

    probabilities

)


pr_auc = average_precision_score(

    y_test,

    probabilities

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "Creating confusion matrix..."
)


cm = confusion_matrix(

    y_test,

    predictions

)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm

)


disp.plot()


plt.title(
    "Credit Risk - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "confusion_matrix.png"

    ),

    dpi=150

)


plt.close()


# ============================================================
# ROC CURVE
# ============================================================

print(
    "Creating ROC curve..."
)


plt.figure()


plt.plot(

    fpr,

    tpr,

    label=f"ROC-AUC = {roc_auc:.4f}"

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--"

)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "roc_curve.png"

    ),

    dpi=150

)


plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

print(
    "Creating Precision-Recall curve..."
)


plt.figure()


plt.plot(

    recall,

    precision,

    label=f"PR-AUC = {pr_auc:.4f}"

)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "precision_recall_curve.png"

    ),

    dpi=150

)


plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nGenerating classification report..."
)


report = classification_report(

    y_test,

    predictions

)


print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(report)


with open(

    os.path.join(

        OUTPUT_DIR,

        "classification_report.txt"

    ),

    "w"

) as file:

    file.write(report)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "Creating feature importance..."
)


# ------------------------------------------------------------
# Get final estimator
# ------------------------------------------------------------

if hasattr(

    model,

    "named_steps"

):

    final_model = list(

        model.named_steps.values()

    )[-1]

else:

    final_model = model


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

if hasattr(

    final_model,

    "feature_importances_"

):


    importance = (

        final_model
        .feature_importances_

    )


    if len(importance) != len(feature_columns):

        print(
            "\nWARNING:"
        )

        print(
            "Number of feature importances "
            "does not match number of features."
        )

    else:

        importance_df = pd.DataFrame({

            "feature":
                feature_columns,

            "importance":
                importance

        })


        importance_df = (

            importance_df

            .sort_values(

                "importance",

                ascending=False

            )

        )


        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        importance_df.to_csv(

            os.path.join(

                OUTPUT_DIR,

                "feature_importance.csv"

            ),

            index=False

        )


        # ----------------------------------------------------
        # Top 20
        # ----------------------------------------------------

        top_features = (

            importance_df

            .head(20)

            .sort_values(

                "importance"

            )

        )


        plt.figure(

            figsize=(10, 7)

        )


        plt.barh(

            top_features["feature"],

            top_features["importance"]

        )


        plt.xlabel(
            "Importance"
        )


        plt.ylabel(
            "Feature"
        )


        plt.title(
            "Top 20 Feature Importance"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                OUTPUT_DIR,

                "feature_importance.png"

            ),

            dpi=150

        )


        plt.close()


        print(
            "Feature importance saved."
        )


else:

    print(
        "\nFeature importance is not available "
        "for this model."
    )


# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

metrics = {

    "roc_auc":
        float(roc_auc),

    "pr_auc":
        float(pr_auc),

    "threshold":
        float(threshold),

    "test_samples":
        int(len(y_test)),

    "actual_high_risk":
        int(y_test.sum()),

    "predicted_high_risk":
        int(predictions.sum())

}


joblib.dump(

    metrics,

    os.path.join(

        OUTPUT_DIR,

        "evaluation_metrics.joblib"

    )

)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")


print("=" * 60)

print(
    "MODEL EVALUATION COMPLETE"
)

print("=" * 60)


print(
    f"ROC-AUC             : {roc_auc:.4f}"
)


print(
    f"PR-AUC              : {pr_auc:.4f}"
)


print(
    f"Threshold           : {threshold:.4f}"
)


print(
    f"Test Samples        : {len(y_test)}"
)


print(
    f"Actual High Risk    : {y_test.sum()}"
)


print(
    f"Predicted High Risk : {predictions.sum()}"
)


print("=" * 60)


print(
    "\nFiles saved to:"
)


print(
    OUTPUT_DIR
)