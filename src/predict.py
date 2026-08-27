import joblib
import pandas as pd

from features import create_features


# Load model
model = joblib.load(
    "models/credit_risk_model.joblib"
)


# Load metadata
metadata = joblib.load(
    "models/model_metadata.joblib"
)


def predict_credit_risk(customer_data):

    # Convert dictionary to DataFrame
    df = pd.DataFrame(
        [customer_data]
    )

    # Apply feature engineering
    df = create_features(df)

    # Ensure correct feature order
    feature_columns = (
        metadata["feature_columns"]
    )

    df = df.reindex(
        columns=feature_columns
    )

    # Predict probability
    probability = (
        model.predict_proba(df)[0][1]
    )

    # Apply threshold
    threshold = metadata["threshold"]

    prediction = int(
        probability >= threshold
    )

    return {
        "risk_probability": float(
            probability
        ),
        "prediction": prediction,
        "risk_level": (
            "HIGH RISK"
            if prediction == 1
            else "LOW RISK"
        )
    }