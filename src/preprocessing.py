# ============================================
# PREPROCESSING MODULE
# ============================================

"""
Purpose
-------
Handles:
- Data cleaning
- Missing value imputation
- Outlier treatment
- Feature engineering
- Encoding
- Scaling
- Exporting processed dataset
- Saving preprocessing pipeline

Best Practice
-------------
Uses reusable sklearn pipelines for:
- Training consistency
- Deployment compatibility
- Production readiness
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np

from pathlib import Path
import logging
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================
# LOGGER CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================
# PROJECT ROOT PATH
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================
# PREPROCESSING FUNCTION
# ============================================

def preprocess_data(df: pd.DataFrame):

    """
    Full preprocessing pipeline.
    """

    logging.info("========== STARTING PREPROCESSING ==========")

    # ============================================
    # COPY DATAFRAME
    # ============================================

    df = df.copy()

    # ============================================
    # REMOVE DUPLICATES
    # ============================================

    initial_shape = df.shape

    df.drop_duplicates(inplace=True)

    logging.info(
        f"Duplicates Removed: {initial_shape[0] - df.shape[0]}"
    )

    # ============================================
    # DROP CUSTOMER ID
    # ============================================

    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    # ============================================
    # DATA TYPE CORRECTIONS
    # ============================================

    numerical_columns = [
        "Age",
        "Income",
        "AdSpend",
        "ClickThroughRate",
        "ConversionRate",
        "WebsiteVisits",
        "PagesPerVisit",
        "TimeOnSite",
        "SocialShares",
        "EmailOpens",
        "EmailClicks",
        "PreviousPurchases",
        "LoyaltyPoints"
    ]

    for col in numerical_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ============================================
    # CONTEXT-AWARE MISSING VALUE IMPUTATION
    # ============================================

    median_cols = [
        "Income",
        "AdSpend",
        "LoyaltyPoints"
    ]

    mean_cols = [
        "WebsiteVisits",
        "PagesPerVisit",
        "TimeOnSite",
        "EmailClicks",
        "EmailOpens",
        "SocialShares",
        "ClickThroughRate",
        "ConversionRate"
    ]

    # Median Imputation
    for col in median_cols:

        if col in df.columns:

            df[col].fillna(
                df[col].median(),
                inplace=True
            )

    # Mean Imputation
    for col in mean_cols:

        if col in df.columns:

            df[col].fillna(
                df[col].mean(),
                inplace=True
            )

    # Mode Imputation for Categoricals
    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_columns:

        df[col].fillna(
            df[col].mode()[0],
            inplace=True
        )

    logging.info("✅ Missing values handled")


    # ============================================
    # OUTLIER CAPPING (1st - 99th Percentile)
    # ============================================

    for col in numerical_columns:

        if col in df.columns:

            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)

            df[col] = np.where(
                df[col] < lower,
                lower,
                df[col]
            )

            df[col] = np.where(
                df[col] > upper,
                upper,
                df[col]
            )

    logging.info("✅ Outliers capped")


    # ============================================
    # FEATURE ENGINEERING
    # ============================================

    # Engagement Score
    df["EngagementScore"] = (
        df["WebsiteVisits"] * 0.2 +
        df["PagesPerVisit"] * 0.2 +
        df["TimeOnSite"] * 0.2 +
        df["EmailClicks"] * 0.2 +
        df["SocialShares"] * 0.2
    )

    # Value Per Visit
    df["ValuePerVisit"] = (
        df["AdSpend"] /
        (df["WebsiteVisits"] + 1)
    )

    # Email CTR
    df["EmailCTR"] = (
        df["EmailClicks"] /
        (df["EmailOpens"] + 1)
    )

    # Customer Value
    df["CustomerValue"] = (
        df["PreviousPurchases"] *
        df["LoyaltyPoints"]
    )

    # Deep Engagement
    df["DeepEngagement"] = (
        df["PagesPerVisit"] *
        df["TimeOnSite"]
    )

    # Spend per Click
    df["Spend_per_Click"] = (
        df["AdSpend"] /
        (df["ClickThroughRate"] + 1e-5)
    )

    # Email Engagement Rate
    df["EmailEngagementRate"] = (
        df["EmailClicks"] /
        (df["EmailOpens"] + 1)
    )

    logging.info("✅ Feature engineering completed")


    # ============================================
    # SPLIT FEATURES & TARGET
    # ============================================

    X = df.drop("Conversion", axis=1)
    y = df["Conversion"]


    # ============================================
    # IDENTIFY COLUMN TYPES
    # ============================================

    num_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include="object"
    ).columns.tolist()


    # ============================================
    # NUMERICAL PIPELINE
    # ============================================

    num_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])


    # ============================================
    # CATEGORICAL PIPELINE
    # ============================================

    cat_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )

    ])


    # ============================================
    # COLUMN TRANSFORMER
    # ============================================

    preprocessor = ColumnTransformer([

        ("num", num_pipeline, num_cols),

        ("cat", cat_pipeline, cat_cols)

    ])


    # ============================================
    # TRANSFORM DATA
    # ============================================

    X_processed = preprocessor.fit_transform(X)


    # ============================================
    # GET FEATURE NAMES
    # ============================================

    encoded_cols = preprocessor.named_transformers_[
        "cat"
    ].named_steps[
        "encoder"
    ].get_feature_names_out(cat_cols)

    final_columns = (
        num_cols +
        list(encoded_cols)
    )


    # ============================================
    # CREATE FINAL DATAFRAME
    # ============================================

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=final_columns
    )

    final_df = pd.concat(
        [
            X_processed_df,
            y.reset_index(drop=True)
        ],
        axis=1
    )

    logging.info(
        f"✅ Final Processed Shape: {final_df.shape}"
    )


    # ============================================
    # SAVE PROCESSED DATA
    # ============================================

    PROCESSED_DATA_PATH = (
        BASE_DIR /
        "data/processed/processed_data_s.csv"
    )

    # 🔥 CHANGE THIS ONLY IF
    # your processed filename changes


    final_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    logging.info(
        f"✅ Processed data saved to:\n{PROCESSED_DATA_PATH}"
    )


    # ============================================
    # SAVE PREPROCESSOR PIPELINE
    # ============================================

    PREPROCESSOR_PATH = (
        BASE_DIR /
        "models/data_preprocessor_s.pkl"
    )

    # 🔥 CHANGE THIS ONLY IF
    # your preprocessor filename changes


    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    logging.info(
        f"✅ Preprocessor saved to:\n{PREPROCESSOR_PATH}"
    )


    # ============================================
    # RETURN OBJECTS
    # ============================================

    return (
        final_df,
        preprocessor
    )


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # ============================================
    # IMPORT DATA LOADER
    # ============================================

    from data_loader import load_data


    # ============================================
    # INPUT RAW DATA FILE PATH
    # ============================================

    RAW_DATA_PATH = (
        BASE_DIR /
        "data/raw/raw_data.csv"
    )

    # 🔥 CHANGE THIS ONLY IF
    # your raw dataset filename changes

    # Example:
    # "marketing_campaign.csv"


    # ============================================
    # LOAD DATA
    # ============================================

    df = load_data(RAW_DATA_PATH)


    # ============================================
    # RUN PREPROCESSING
    # ============================================

    processed_df, preprocessor = preprocess_data(df)


    # ============================================
    # SUCCESS MESSAGE
    # ============================================

    print("\n✅ Preprocessing Pipeline Executed Successfully!")