import os
import joblib
import pandas as pd

from django.conf import settings
from django.shortcuts import render, redirect


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "deployment_credit_risk_model.joblib"
)

METADATA_PATH = os.path.join(
    settings.BASE_DIR,
    "models",
    "deployment_model_metadata.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

metadata = joblib.load(
    METADATA_PATH
)

FINAL_THRESHOLD = metadata["threshold"]


# ============================================================
# HOME
# ============================================================

def home(request):

    history_data = request.session.get(
        "prediction_history",
        []
    )

    context = {

        "roc_auc": round(
            metadata.get(
                "roc_auc",
                0
            ),
            4
        ),

        "pr_auc": round(
            metadata.get(
                "pr_auc",
                0
            ),
            4
        ),

        "threshold": FINAL_THRESHOLD,

        "feature_count": len(
            metadata.get(
                "feature_columns",
                []
            )
        ),

        "prediction_count": len(
            history_data
        ),

        "high_risk_count": sum(
            1
            for item in history_data
            if item["risk_level"] == "HIGH RISK"
        ),

        "average_risk": round(
            sum(
                item["probability"]
                for item in history_data
            ) / len(history_data),
            2
        ) if history_data else 0
    }

    return render(
        request,
        "predictor/index.html",
        context
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_risk(request):

    if request.method != "POST":

        return redirect("home")


    try:

        # ====================================================
        # INPUT DATA
        # ====================================================

        customer_data = {

            "AGE_YEARS": float(
                request.POST["AGE_YEARS"]
            ),

            "CODE_GENDER": request.POST[
                "CODE_GENDER"
            ],

            "AMT_INCOME_TOTAL": float(
                request.POST[
                    "AMT_INCOME_TOTAL"
                ]
            ),

            "AMT_CREDIT": float(
                request.POST[
                    "AMT_CREDIT"
                ]
            ),

            "AMT_ANNUITY": float(
                request.POST[
                    "AMT_ANNUITY"
                ]
            ),

            "AMT_GOODS_PRICE": float(
                request.POST[
                    "AMT_GOODS_PRICE"
                ]
            ),

            "CNT_CHILDREN": int(
                request.POST[
                    "CNT_CHILDREN"
                ]
            ),

            "CNT_FAM_MEMBERS": float(
                request.POST[
                    "CNT_FAM_MEMBERS"
                ]
            ),

            "EMPLOYMENT_YEARS": float(
                request.POST[
                    "EMPLOYMENT_YEARS"
                ]
            ),

            "NAME_INCOME_TYPE": request.POST[
                "NAME_INCOME_TYPE"
            ],

            "NAME_EDUCATION_TYPE": request.POST[
                "NAME_EDUCATION_TYPE"
            ],

            "NAME_FAMILY_STATUS": request.POST[
                "NAME_FAMILY_STATUS"
            ],

            "NAME_HOUSING_TYPE": request.POST[
                "NAME_HOUSING_TYPE"
            ],

            "OCCUPATION_TYPE": request.POST.get(
                "OCCUPATION_TYPE",
                None
            ),

            "FLAG_EMP_PHONE": int(
                request.POST[
                    "FLAG_EMP_PHONE"
                ]
            ),

            "FLAG_WORK_PHONE": int(
                request.POST[
                    "FLAG_WORK_PHONE"
                ]
            ),

            "FLAG_PHONE": int(
                request.POST[
                    "FLAG_PHONE"
                ]
            ),

            "FLAG_EMAIL": int(
                request.POST[
                    "FLAG_EMAIL"
                ]
            )
        }


        # ====================================================
        # DATAFRAME
        # ====================================================

        customer_df = pd.DataFrame(
            [customer_data]
        )


        customer_df = customer_df[
            metadata["feature_columns"]
        ]


        # ====================================================
        # PREDICTION
        # ====================================================

        probability = float(
            model.predict_proba(
                customer_df
            )[0][1]
        )


        prediction = int(
            probability >= FINAL_THRESHOLD
        )


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if probability < 0.30:

            risk_level = "LOW RISK"

        elif probability < FINAL_THRESHOLD:

            risk_level = "MEDIUM RISK"

        else:

            risk_level = "HIGH RISK"


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        if risk_level == "LOW RISK":

            recommendation = (
                "Customer has relatively low "
                "predicted credit risk."
            )

        elif risk_level == "MEDIUM RISK":

            recommendation = (
                "Customer requires additional "
                "financial assessment."
            )

        else:

            recommendation = (
                "Customer shows elevated predicted "
                "credit risk. Further assessment "
                "is recommended."
            )


        # ====================================================
        # HISTORY
        # ====================================================

        history_data = request.session.get(
            "prediction_history",
            []
        )


        history_data.append({

            "probability": round(
                probability * 100,
                2
            ),

            "risk_level": risk_level,

        })


        # Keep only latest 10
        history_data = history_data[-10:]


        request.session[
            "prediction_history"
        ] = history_data


        # ====================================================
        # RESULT
        # ====================================================

        context = {

            "risk_probability": round(
                probability,
                4
            ),

            "risk_percentage": round(
                probability * 100,
                2
            ),

            "prediction": prediction,

            "risk_level": risk_level,

            "recommendation": recommendation,

            "threshold": FINAL_THRESHOLD,

            "roc_auc": round(
                metadata.get(
                    "roc_auc",
                    0
                ),
                4
            ),

            "pr_auc": round(
                metadata.get(
                    "pr_auc",
                    0
                ),
                4
            )
        }


        return render(
            request,
            "predictor/result.html",
            context
        )


    except Exception as error:

        return render(
            request,
            "predictor/index.html",
            {
                "error": str(error)
            }
        )


# ============================================================
# HISTORY
# ============================================================

def history(request):

    history_data = request.session.get(
        "prediction_history",
        []
    )

    total_predictions = len(
        history_data
    )

    high_risk = sum(
        1
        for item in history_data
        if item["risk_level"] == "HIGH RISK"
    )

    medium_risk = sum(
        1
        for item in history_data
        if item["risk_level"] == "MEDIUM RISK"
    )

    low_risk = sum(
        1
        for item in history_data
        if item["risk_level"] == "LOW RISK"
    )


    average_risk = (

        round(
            sum(
                item["probability"]
                for item in history_data
            )
            / total_predictions,
            2
        )

        if total_predictions > 0

        else 0
    )


    context = {

        "history": history_data,

        "total_predictions":
            total_predictions,

        "high_risk":
            high_risk,

        "medium_risk":
            medium_risk,

        "low_risk":
            low_risk,

        "average_risk":
            average_risk
    }


    return render(
        request,
        "predictor/history.html",
        context
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_history(request):

    if request.method == "POST":

        request.session[
            "prediction_history"
        ] = []

    return redirect("history")