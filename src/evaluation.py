# ============================================
# MODEL EVALUATION MODULE
# ============================================

"""
Purpose
-------
Handles:
- Classification Metrics
- ROC-AUC Evaluation
- Threshold-based Predictions
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Threshold Performance Visualization

Compatible With
---------------
- XGBoost
- LightGBM
- Random Forest
- Ensemble Models

Outputs
-------
- Evaluation plots
- Saved reports
- Business insights
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np

from pathlib import Path
import logging
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,

    confusion_matrix,
    classification_report,

    roc_curve,
    precision_recall_curve
)

# ============================================
# LOGGER CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================
# SEABORN THEME
# ============================================

sns.set_theme(style="whitegrid")

# ============================================
# PROJECT ROOT PATH
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================
# OUTPUT DIRECTORY
# ============================================

OUTPUT_DIR = (
    BASE_DIR / "reports/evaluation_s"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================
# LOAD MODEL
# ============================================

def load_model(model_path):

    """
    Load trained model.
    """

    model = joblib.load(model_path)

    logging.info(
        f"✅ Model loaded from:\n{model_path}"
    )

    return model

# ============================================
# LOAD PROCESSED DATA
# ============================================

def load_processed_data(data_path):

    """
    Load processed dataset.
    """

    df = pd.read_csv(data_path)

    logging.info(
        f"✅ Processed data loaded from:\n{data_path}"
    )

    return df

# ============================================
# MODEL EVALUATION
# ============================================

def evaluate_model(

    model,
    X_test,
    y_test,
    threshold=0.5
):

    """
    Evaluate classification model.
    """

    logging.info(
        "========== MODEL EVALUATION =========="
    )

    # ============================================
    # PREDICTION PROBABILITIES
    # ============================================

    if hasattr(model, "predict_proba"):

        y_prob = (
            model.predict_proba(X_test)[:,1]
        )

    elif hasattr(model, "decision_function"):

        scores = (
            model.decision_function(X_test)
        )

        y_prob = (
            scores - scores.min()
        ) / (
            scores.max() - scores.min()
        )

    else:

        raise ValueError(
            "Model has neither predict_proba nor decision_function"
        )

    # ============================================
    # THRESHOLD PREDICTIONS
    # ============================================

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # ============================================
    # METRICS
    # ============================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    # ============================================
    # CLASSIFICATION REPORT
    # ============================================

    print("\n========== CLASSIFICATION REPORT ==========\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    # ============================================
    # SAVE METRICS
    # ============================================

    metrics_df = pd.DataFrame([{

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold
    }])

    metrics_df.to_csv(

        OUTPUT_DIR / "metrics_s.csv",

        index=False
    )

    logging.info(
        "✅ Metrics saved"
    )

    logging.info(
        "✅ Evaluation completed"
    )

    return y_prob, y_pred

# ============================================
# CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(

    y_test,
    y_pred
):

    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(

        y_test,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "confusion_matrix.png",

        bbox_inches="tight"
    )

    plt.close()

    logging.info(
        "✅ Confusion Matrix saved"
    )

# ============================================
# ROC CURVE
# ============================================

def plot_roc_curve(

    y_test,
    y_prob
):

    """
    Plot ROC Curve.
    """

    fpr, tpr, _ = roc_curve(

        y_test,
        y_prob
    )

    plt.figure(figsize=(8,6))

    plt.plot(

        fpr,
        tpr,

        label="ROC Curve"
    )

    plt.plot(

        [0,1],
        [0,1],

        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "roc_curve.png",

        bbox_inches="tight"
    )

    plt.close()

    logging.info(
        "✅ ROC Curve saved"
    )

# ============================================
# PRECISION-RECALL CURVE
# ============================================

def plot_precision_recall_curve(

    y_test,
    y_prob
):

    """
    Plot Precision-Recall Curve.
    """

    precision, recall, _ = (

        precision_recall_curve(

            y_test,
            y_prob
        )
    )

    plt.figure(figsize=(8,6))

    plt.plot(

        recall,
        precision
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "precision_recall_curve.png",

        bbox_inches="tight"
    )

    plt.close()

    logging.info(
        "✅ Precision-Recall Curve saved"
    )

# ============================================
# THRESHOLD ANALYSIS
# ============================================

def threshold_analysis(

    y_test,
    y_prob
):

    """
    Analyze threshold behavior.
    """

    thresholds = np.arange(

        0.1,
        0.91,
        0.05
    )

    precision_scores = []

    recall_scores = []

    f1_scores = []

    for threshold in thresholds:

        y_pred = (

            y_prob >= threshold

        ).astype(int)

        precision_scores.append(

            precision_score(

                y_test,
                y_pred,

                zero_division=0
            )
        )

        recall_scores.append(

            recall_score(

                y_test,
                y_pred,

                zero_division=0
            )
        )

        f1_scores.append(

            f1_score(

                y_test,
                y_pred,

                zero_division=0
            )
        )

    # ============================================
    # BEST THRESHOLD
    # ============================================

    best_threshold = thresholds[
        np.argmax(f1_scores)
    ]

    logging.info(
        f"✅ Best Threshold: {best_threshold:.2f}"
    )

    # ============================================
    # PLOT
    # ============================================

    plt.figure(figsize=(10,6))

    plt.plot(

        thresholds,
        precision_scores,

        label="Precision"
    )

    plt.plot(

        thresholds,
        recall_scores,

        label="Recall"
    )

    plt.plot(

        thresholds,
        f1_scores,

        label="F1 Score"
    )

    plt.axvline(

        best_threshold,

        linestyle="--",

        color="red",

        label=f"Best Threshold = {best_threshold:.2f}"
    )

    plt.xlabel("Threshold")

    plt.ylabel("Score")

    plt.title("Threshold Optimization")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "threshold_analysis.png",

        bbox_inches="tight"
    )

    plt.close()

    logging.info(
        "✅ Threshold Analysis saved"
    )

# ============================================
# FEATURE IMPORTANCE
# ============================================

def plot_feature_importance(

    model,
    X_test
):

    """
    Plot feature importance.
    """

    try:

        if hasattr(

            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

        elif hasattr(

            model,
            "coef_"
        ):

            importance = np.abs(
                model.coef_[0]
            )

        else:

            estimator = (
                model.estimator.estimators_[0]
            )

            importance = (
                estimator.feature_importances_
            )

        feature_names = (
            X_test.columns
        )

        importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance
        })

        importance_df = (

            importance_df
            .sort_values(

                by="Importance",

                ascending=False
            )
            .head(20)
        )

        plt.figure(figsize=(10,8))

        sns.barplot(

            data=importance_df,

            x="Importance",

            y="Feature"
        )

        plt.title(
            "Top 20 Feature Importances"
        )

        plt.tight_layout()

        plt.savefig(

            OUTPUT_DIR / "feature_importance.png",

            bbox_inches="tight"
        )

        plt.close()

        logging.info(
            "✅ Feature Importance saved"
        )

    except Exception as e:

        logging.warning(
            f"⚠️ Feature importance unavailable: {e}"
        )

# ============================================
# COMPLETE EVALUATION PIPELINE
# ============================================

def evaluation_pipeline(

    model,
    X_test,
    y_test,
    threshold=0.5
):

    """
    End-to-end evaluation pipeline.
    """

    # ============================================
    # EVALUATE MODEL
    # ============================================

    y_prob, y_pred = evaluate_model(

        model,

        X_test,

        y_test,

        threshold
    )

    # ============================================
    # GENERATE PLOTS
    # ============================================

    plot_confusion_matrix(

        y_test,

        y_pred
    )

    plot_roc_curve(

        y_test,

        y_prob
    )

    plot_precision_recall_curve(

        y_test,

        y_prob
    )

    threshold_analysis(

        y_test,

        y_prob
    )

    plot_feature_importance(

        model,

        X_test
    )

    logging.info(
        "✅ Full evaluation pipeline completed"
    )

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    # ============================================
    # MODEL PATH
    # ============================================

    MODEL_PATH = (

        BASE_DIR /

        "models/final_ensemble_model_s.pkl"
    )
    
    # 🔥 CHANGE THIS ONLY IF
    # model filename changes

    # ============================================
    # DATA PATH
    # ============================================

    DATA_PATH = (

        BASE_DIR /

        "data/processed/processed_data_s.csv"
    )

    # 🔥 CHANGE THIS ONLY IF
    # processed dataset filename changes

    # ============================================
    # LOAD DATA
    # ============================================

    df = load_processed_data(
        DATA_PATH
    )

    # ============================================
    # FEATURES/TARGET
    # ============================================

    X = df.drop(
        "Conversion",
        axis=1
    )

    y = df["Conversion"]

    # ============================================
    # TRAIN TEST SPLIT
    # ============================================

    from sklearn.model_selection import (
        train_test_split
    )

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=0.2,

            stratify=y,

            random_state=42
        )
    )

    # ============================================
    # LOAD MODEL
    # ============================================

    model = load_model(
        MODEL_PATH
    )

    # ============================================
    # RUN EVALUATION PIPELINE
    # ============================================

    evaluation_pipeline(

        model,

        X_test,

        y_test,

        threshold=0.5
    )

    # ============================================
    # SUCCESS MESSAGE
    # ============================================

    print(
        "\n✅ Evaluation Pipeline Completed Successfully!"
    )