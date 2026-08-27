import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # Handle sentinel value
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        365243,
        np.nan
    )

    # Age and employment
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365.25
    )

    # Financial ratios
    df["INCOME_TO_CREDIT_RATIO"] = (
        df["AMT_INCOME_TOTAL"] /
        df["AMT_CREDIT"]
    )

    df["CREDIT_TO_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_TO_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_TO_ANNUITY_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_ANNUITY"]
    )

    df["CREDIT_TO_GOODS_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_GOODS_PRICE"]
    )

    # Family features
    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] /
        df["CNT_FAM_MEMBERS"]
    )

    df["CHILDREN_RATIO"] = (
        df["CNT_CHILDREN"] /
        df["CNT_FAM_MEMBERS"]
    )

    df["EMPLOYMENT_TO_AGE_RATIO"] = (
        df["EMPLOYMENT_YEARS"] /
        df["AGE_YEARS"]
    )

    # Age group
    df["AGE_GROUP"] = pd.cut(
        df["AGE_YEARS"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "65+"
        ]
    )

    # Employment group
    df["EMPLOYMENT_GROUP"] = pd.cut(
        df["EMPLOYMENT_YEARS"],
        bins=[-1, 1, 3, 5, 10, 20, 100],
        labels=[
            "<1 Year",
            "1-3 Years",
            "3-5 Years",
            "5-10 Years",
            "10-20 Years",
            "20+ Years"
        ]
    )

    # Document aggregate
    document_columns = [
        col for col in df.columns
        if col.startswith("FLAG_DOCUMENT")
    ]

    df["TOTAL_DOCUMENTS_SUBMITTED"] = (
        df[document_columns].sum(axis=1)
    )

    # External source features
    external_sources = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    df["EXT_SOURCE_MEAN"] = (
        df[external_sources].mean(axis=1)
    )

    df["EXT_SOURCE_MAX"] = (
        df[external_sources].max(axis=1)
    )

    df["EXT_SOURCE_MIN"] = (
        df[external_sources].min(axis=1)
    )

    df["EXT_SOURCE_STD"] = (
        df[external_sources].std(axis=1)
    )

    df["EXT_SOURCE_COUNT"] = (
        df[external_sources]
        .notna()
        .sum(axis=1)
    )

    # Contact information
    contact_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL"
    ]

    df["TOTAL_CONTACT_FLAGS"] = (
        df[contact_columns].sum(axis=1)
    )

    # Handle infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df