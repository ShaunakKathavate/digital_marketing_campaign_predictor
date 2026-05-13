# ============================================
# STREAMLIT DASHBOARD APP
# ============================================

"""
Digital Marketing Conversion Prediction Dashboard

Features
--------
- Dataset Overview
- Business Insights
- Interactive Visualizations
- Model Performance Metrics
- Conversion Prediction
- Feature Importance
- Threshold Analysis

Run Command
-----------
streamlit run app.py
"""

# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(

    page_title="Marketing Conversion Dashboard",

    page_icon="📈",

    layout="wide"
)


# ============================================
# PROJECT ROOT PATH
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():

    DATA_PATH = (

        BASE_DIR /

        "data/processed/processed_data_s.csv"
    )

    # 🔥 CHANGE THIS ONLY IF
    # processed dataset filename changes

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    MODEL_PATH = (

        BASE_DIR /

        "models/final_ensemble_model_s.pkl"
    )

    # 🔥 CHANGE THIS ONLY IF
    # model filename changes

    model = joblib.load(MODEL_PATH)

    return model


# ============================================
# LOAD OBJECTS
# ============================================

df = load_data()

model = load_model()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(

    "Go To",

    [

        "Overview",

        "Business Insights",

        "Visualizations",

        "Model Evaluation",

        "Prediction System"
    ]
)


# ============================================
# OVERVIEW PAGE
# ============================================

if page == "Overview":

    st.title("📈 Marketing Conversion Prediction Dashboard")

    st.markdown("---")

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        df.shape[0]
    )

    col2.metric(
        "Total Features",
        df.shape[1]
    )

    conversion_rate = (
        df["Conversion"].mean() * 100
    )

    col3.metric(
        "Conversion Rate",
        f"{conversion_rate:.2f}%"
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


# ============================================
# BUSINESS INSIGHTS PAGE
# ============================================

elif page == "Business Insights":

    st.title("📊 Business Insights")

    st.markdown("---")

    # ============================================
    # CONVERSION RATE
    # ============================================

    conversion_rate = (
        df["Conversion"].mean() * 100
    )

    st.metric(
        "Overall Conversion Rate",
        f"{conversion_rate:.2f}%"
    )

    st.markdown("---")

    # ============================================
    # TOP FEATURES
    # ============================================

    st.subheader("Top Correlated Features")

    correlation = df.corr(
        numeric_only=True
    )["Conversion"].sort_values(
        ascending=False
    )[1:11]

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    correlation.plot(
        kind="bar",
        ax=ax
    )

    plt.title(
        "Top Features Correlated with Conversion"
    )

    st.pyplot(fig)

    st.markdown("---")

    # ============================================
    # HIGH VALUE SEGMENTS
    # ============================================

    st.subheader("High Value Customer Segments")

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    sns.boxplot(

        data=df,

        x="Conversion",

        y="Income",

        ax=ax
    )

    plt.title(
        "Income Distribution by Conversion"
    )

    st.pyplot(fig)


# ============================================
# VISUALIZATION PAGE
# ============================================

elif page == "Visualizations":

    st.title("📉 Visual Analytics")

    st.markdown("---")

    # ============================================
    # CONVERSION DISTRIBUTION
    # ============================================

    st.subheader("Conversion Distribution")

    fig, ax = plt.subplots(
        figsize=(6,4)
    )

    sns.countplot(

        data=df,

        x="Conversion",

        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # ============================================
    # CORRELATION HEATMAP
    # ============================================

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(14,10)
    )

    sns.heatmap(

        df.corr(numeric_only=True),

        cmap="coolwarm",

        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # ============================================
    # FEATURE DISTRIBUTION
    # ============================================

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(

        "Select Feature",

        df.columns[:-1]
    )

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    sns.histplot(

        df[selected_feature],

        kde=True,

        ax=ax
    )

    st.pyplot(fig)


# ============================================
# MODEL EVALUATION PAGE
# ============================================

elif page == "Model Evaluation":

    st.title("🤖 Model Evaluation")

    st.markdown("---")

    X = df.drop(
        "Conversion",
        axis=1
    )

    y = df["Conversion"]

    y_prob = model.predict_proba(X)[:,1]

    threshold = st.slider(

        "Select Prediction Threshold",

        0.1,
        0.9,
        0.5,
        0.05
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    # ============================================
    # METRICS
    # ============================================

    from sklearn.metrics import (

        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred
    )

    recall = recall_score(
        y,
        y_pred
    )

    f1 = f1_score(
        y,
        y_pred
    )

    roc_auc = roc_auc_score(
        y,
        y_prob
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col3.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col5.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )

    st.markdown("---")

    # ============================================
    # CONFUSION MATRIX
    # ============================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6,5)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # ============================================
    # CLASSIFICATION REPORT
    # ============================================

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)


# ============================================
# PREDICTION SYSTEM PAGE
# ============================================

elif page == "Prediction System":

    st.title("🎯 Customer Conversion Prediction")

    st.markdown("---")

    st.subheader("Enter Customer Details")

    # ============================================
    # USER INPUTS
    # ============================================

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    income = st.number_input(
        "Income",
        value=50000
    )

    ad_spend = st.number_input(
        "Ad Spend",
        value=1000
    )

    website_visits = st.slider(
        "Website Visits",
        1,
        50,
        5
    )

    pages_per_visit = st.slider(
        "Pages Per Visit",
        1,
        20,
        5
    )

    time_on_site = st.slider(
        "Time On Site",
        1,
        60,
        10
    )

    email_clicks = st.slider(
        "Email Clicks",
        0,
        50,
        5
    )

    social_shares = st.slider(
        "Social Shares",
        0,
        100,
        10
    )

    loyalty_points = st.number_input(
        "Loyalty Points",
        value=100
    )

    # ============================================
    # PREDICTION BUTTON
    # ============================================

    if st.button("Predict Conversion"):

        # ============================================
        # CREATE INPUT DATAFRAME
        # ============================================

        sample = X.iloc[0:1].copy()

        sample["Age"] = age
        sample["Income"] = income
        sample["AdSpend"] = ad_spend
        sample["WebsiteVisits"] = website_visits
        sample["PagesPerVisit"] = pages_per_visit
        sample["TimeOnSite"] = time_on_site
        sample["EmailClicks"] = email_clicks
        sample["SocialShares"] = social_shares
        sample["LoyaltyPoints"] = loyalty_points

        # ============================================
        # PREDICT
        # ============================================

        probability = (
            model.predict_proba(sample)[0][1]
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        st.markdown("---")

        st.subheader("Prediction Results")

        st.metric(
            "Conversion Probability",
            f"{probability:.2%}"
        )

        if prediction == 1:

            st.success(
                "✅ Likely to Convert"
            )

        else:

            st.error(
                "❌ Unlikely to Convert"
            )