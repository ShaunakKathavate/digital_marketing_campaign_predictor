# ============================================
# DATA LOADER MODULE
# ============================================

"""
Purpose
-------
Handles:
- Loading raw dataset
- Validating required columns
- Dataset summary generation

Best Practice
-------------
Use dynamic project-root paths so the code works:
- Locally
- In Jupyter notebooks
- In Streamlit deployment
- On GitHub
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import logging
from pathlib import Path


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

# Dynamically finds project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================
# LOAD DATA FUNCTION
# ============================================

def load_data(file_path) -> pd.DataFrame:
    """
    Load CSV dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to CSV file

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """

    try:

        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(
                f"❌ File not found: {file_path}"
            )

        # Load dataset
        df = pd.read_csv(path)

        logging.info(
            f"✅ Dataset loaded successfully from:\n{file_path}"
        )

        logging.info(f"Dataset Shape: {df.shape}")

        return df

    except Exception as e:

        logging.error(f"❌ Error loading dataset: {e}")

        raise


# ============================================
# VALIDATE REQUIRED COLUMNS
# ============================================

def validate_columns(df: pd.DataFrame, required_columns: list):
    """
    Validate required dataset columns.
    """

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"❌ Missing Required Columns: {missing_cols}"
        )

    logging.info("✅ All required columns are present.")


# ============================================
# DATASET SUMMARY
# ============================================

def dataset_summary(df: pd.DataFrame):
    """
    Print dataset summary.
    """

    logging.info("========== DATASET SUMMARY ==========")

    logging.info(f"Rows: {df.shape[0]}")
    logging.info(f"Columns: {df.shape[1]}")

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATES ==========")
    print(df.duplicated().sum())


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # =====================================================
    # 🔥 INPUT RAW DATA FILE NAME HERE
    # =====================================================

    FILE_PATH = BASE_DIR / "data/raw/raw_data.csv"

    # Replace "raw_data_file_name.csv"
    # with actual raw dataset filename if different

    # Example:
    # FILE_PATH = BASE_DIR / "data/raw/marketing_campaign.csv"

    # =====================================================


    # ============================================
    # REQUIRED COLUMNS
    # ============================================

    REQUIRED_COLUMNS = [

        "Age",
        "Gender",
        "Income",

        "CampaignChannel",
        "CampaignType",

        "AdSpend",
        "ClickThroughRate",

        "WebsiteVisits",
        "PagesPerVisit",
        "TimeOnSite",

        "SocialShares",
        "EmailOpens",
        "EmailClicks",

        "PreviousPurchases",
        "LoyaltyPoints",

        "Conversion"
    ]


    # ============================================
    # LOAD DATA
    # ============================================

    df = load_data(FILE_PATH)


    # ============================================
    # VALIDATE DATA
    # ============================================

    validate_columns(df, REQUIRED_COLUMNS)


    # ============================================
    # DATASET SUMMARY
    # ============================================

    dataset_summary(df)


    # ============================================
    # SUCCESS MESSAGE
    # ============================================

    print("\n✅ Data Loader Executed Successfully!")