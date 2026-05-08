# ============================================
# ENSEMBLE MODEL TRAINING MODULE
# ============================================

"""
Purpose
-------
Handles:
- Train/Test Split
- Stratified Cross Validation
- Hyperparameter Tuning
- Class Imbalance Handling
- Ensemble Learning
- Probability Calibration
- Threshold Optimization
- Final Model Saving

Models Used
-----------
1. XGBoost
2. LightGBM
3. Random Forest

Final Model
-----------
Soft Voting Ensemble

Optimization Metric
-------------------
ROC-AUC
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np

from pathlib import Path
import logging
import joblib

from sklearn.model_selection import (

    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.metrics import (

    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.ensemble import (

    RandomForestClassifier,
    VotingClassifier
)

from sklearn.calibration import (
    CalibratedClassifierCV
)

import xgboost as xgb
import lightgbm as lgb


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
# LOAD PROCESSED DATA
# ============================================

def load_processed_data(file_path):

    df = pd.read_csv(file_path)

    logging.info(
        f"✅ Processed data loaded from:\n{file_path}"
    )

    logging.info(f"Dataset Shape: {df.shape}")

    return df


# ============================================
# SPLIT DATA
# ============================================

def split_data(df):

    X = df.drop("Conversion", axis=1)

    y = df["Conversion"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        stratify=y,

        random_state=42
    )

    logging.info("✅ Train-Test split completed")

    return X_train, X_test, y_train, y_test


# ============================================
# HYPERPARAMETER TUNING - XGBOOST
# ============================================

def tune_xgboost(X_train, y_train):

    logging.info("========== TUNING XGBOOST ==========")

    scale_pos_weight = (
        len(y_train[y_train == 0]) /
        len(y_train[y_train == 1])
    )

    xgb_model = xgb.XGBClassifier(

        eval_metric="auc",

        scale_pos_weight=scale_pos_weight,

        random_state=42
    )

    param_grid = {

        "n_estimators": [300, 500, 800],

        "max_depth": [3, 4, 5, 6],

        "learning_rate": [0.01, 0.03, 0.05],

        "subsample": [0.7, 0.8, 0.9],

        "colsample_bytree": [0.6, 0.7, 0.8]
    }

    search = RandomizedSearchCV(

        estimator=xgb_model,

        param_distributions=param_grid,

        n_iter=20,

        scoring="roc_auc",

        cv=5,

        verbose=1,

        n_jobs=-1,

        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    logging.info(
        f"✅ Best XGBoost ROC-AUC: {search.best_score_:.4f}"
    )

    logging.info(
        f"Best Parameters:\n{search.best_params_}"
    )

    return best_model


# ============================================
# HYPERPARAMETER TUNING - LIGHTGBM
# ============================================

def tune_lightgbm(X_train, y_train):

    logging.info("========== TUNING LIGHTGBM ==========")

    lgb_model = lgb.LGBMClassifier(
        random_state=42
    )

    param_grid = {

        "n_estimators": [300, 500],

        "learning_rate": [0.01, 0.03],

        "num_leaves": [31, 50, 70],

        "max_depth": [4, 5, 6]
    }

    search = RandomizedSearchCV(

        estimator=lgb_model,

        param_distributions=param_grid,

        n_iter=15,

        scoring="roc_auc",

        cv=5,

        verbose=1,

        n_jobs=-1,

        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    logging.info(
        f"✅ Best LightGBM ROC-AUC: {search.best_score_:.4f}"
    )

    logging.info(
        f"Best Parameters:\n{search.best_params_}"
    )

    return best_model


# ============================================
# RANDOM FOREST MODEL
# ============================================

def train_random_forest():

    rf_model = RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
    )

    logging.info("✅ Random Forest initialized")

    return rf_model


# ============================================
# BUILD ENSEMBLE MODEL
# ============================================

def build_ensemble_model(

    xgb_model,
    lgb_model,
    rf_model
):

    logging.info("========== BUILDING ENSEMBLE ==========")

    ensemble_model = VotingClassifier(

        estimators=[

            ("xgb", xgb_model),

            ("lgb", lgb_model),

            ("rf", rf_model)
        ],

        voting="soft",

        n_jobs=-1
    )

    return ensemble_model


# ============================================
# CROSS VALIDATION
# ============================================

def run_cross_validation(

    model,
    X_train,
    y_train
):

    logging.info("========== CROSS VALIDATION ==========")

    skf = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42
    )

    scores = cross_val_score(

        model,

        X_train,
        y_train,

        cv=skf,

        scoring="roc_auc",

        n_jobs=-1
    )

    logging.info(
        f"✅ Mean CV ROC-AUC: {np.mean(scores):.4f}"
    )

    return scores


# ============================================
# CALIBRATE MODEL
# ============================================

def calibrate_model(

    model,
    X_train,
    y_train
):

    logging.info("========== CALIBRATING MODEL ==========")

    calibrated_model = CalibratedClassifierCV(

        estimator=model,

        method="isotonic",

        cv=3
    )

    calibrated_model.fit(

        X_train,
        y_train
    )

    logging.info("✅ Probability calibration completed")

    return calibrated_model


# ============================================
# THRESHOLD OPTIMIZATION
# ============================================

def optimize_threshold(

    model,
    X_test,
    y_test
):

    logging.info("========== THRESHOLD OPTIMIZATION ==========")

    y_prob = model.predict_proba(X_test)[:,1]

    thresholds = np.arange(0.1, 0.91, 0.05)

    results = []

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        results.append({

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1_score": f1
        })

    results_df = pd.DataFrame(results)

    best_threshold = results_df.loc[
        results_df["f1_score"].idxmax(),
        "threshold"
    ]

    logging.info(
        f"✅ Best Threshold: {best_threshold}"
    )

    return best_threshold, results_df


# ============================================
# EVALUATE MODEL
# ============================================

def evaluate_model(

    model,
    X_test,
    y_test,
    threshold
):

    logging.info("========== MODEL EVALUATION ==========")

    y_prob = model.predict_proba(X_test)[:,1]

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print("\n========== CLASSIFICATION REPORT ==========\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    logging.info("✅ Evaluation completed")


# ============================================
# SAVE MODEL
# ============================================

def save_model(model):

    MODEL_PATH = (

        BASE_DIR /

        "models/final_ensemble_model_s.pkl"
    )

    # 🔥 CHANGE THIS ONLY IF
    # your model filename changes


    joblib.dump(

        model,
        MODEL_PATH
    )

    logging.info(
        f"✅ Final model saved to:\n{MODEL_PATH}"
    )


# ============================================
# COMPLETE TRAINING PIPELINE
# ============================================

def ensemble_training_pipeline(df):

    # ============================================
    # SPLIT DATA
    # ============================================

    X_train, X_test, y_train, y_test = split_data(df)


    # ============================================
    # TUNE XGBOOST
    # ============================================

    best_xgb = tune_xgboost(

        X_train,
        y_train
    )


    # ============================================
    # TUNE LIGHTGBM
    # ============================================

    best_lgb = tune_lightgbm(

        X_train,
        y_train
    )


    # ============================================
    # RANDOM FOREST
    # ============================================

    rf_model = train_random_forest()


    # ============================================
    # BUILD ENSEMBLE
    # ============================================

    ensemble_model = build_ensemble_model(

        best_xgb,
        best_lgb,
        rf_model
    )


    # ============================================
    # CROSS VALIDATION
    # ============================================

    run_cross_validation(

        ensemble_model,

        X_train,
        y_train
    )


    # ============================================
    # TRAIN ENSEMBLE
    # ============================================

    ensemble_model.fit(

        X_train,
        y_train
    )

    logging.info("✅ Ensemble model trained")


    # ============================================
    # CALIBRATION
    # ============================================

    calibrated_model = calibrate_model(

        ensemble_model,

        X_train,
        y_train
    )


    # ============================================
    # THRESHOLD OPTIMIZATION
    # ============================================

    best_threshold, threshold_results = (

        optimize_threshold(

            calibrated_model,

            X_test,
            y_test
        )
    )


    # ============================================
    # FINAL EVALUATION
    # ============================================

    evaluate_model(

        calibrated_model,

        X_test,
        y_test,

        best_threshold
    )


    # ============================================
    # SAVE MODEL
    # ============================================

    save_model(calibrated_model)

    logging.info("✅ Full ensemble pipeline completed")

    return (

        calibrated_model,

        best_threshold,

        threshold_results
    )


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # ============================================
    # INPUT PROCESSED DATA FILE PATH
    # ============================================

    PROCESSED_DATA_PATH = (

        BASE_DIR /

        "data/processed/processed_data_s.csv"
    )

    # 🔥 CHANGE THIS ONLY IF
    # your processed dataset filename changes


    # ============================================
    # LOAD DATA
    # ============================================

    df = load_processed_data(

        PROCESSED_DATA_PATH
    )


    # ============================================
    # RUN PIPELINE
    # ============================================

    model, threshold, results = (

        ensemble_training_pipeline(df)
    )


    # ============================================
    # SUCCESS MESSAGE
    # ============================================

    print(
        "\n✅ Ensemble Model Training Completed Successfully!"
    )