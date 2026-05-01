import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/sample_data.csv")

df = load_data()

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return joblib.load("../models/trained_model.pkl")

model = load_model()

# ================================
# TITLE
# ================================
st.title("📊 Customer Conversion & Marketing Analytics Dashboard")

# ================================
# KPI SECTION
# ================================
col1, col2, col3 = st.columns(3)

conversion_rate = df['Conversion'].mean() * 100
avg_ad_spend = df['AdSpend'].mean()
total_conversions = df['Conversion'].sum()

with col1:
    st.metric("Conversion Rate", f"{conversion_rate:.2f}%")

with col2:
    st.metric("Avg Ad Spend", f"${avg_ad_spend:.2f}")

with col3:
    st.metric("Total Conversions", int(total_conversions))

# ================================
# SIDEBAR FILTERS
# ================================
st.sidebar.header("Filters")

channel = st.sidebar.multiselect(
    "Campaign Channel",
    options=df['CampaignChannel'].unique(),
    default=df['CampaignChannel'].unique()
)

campaign_type = st.sidebar.multiselect(
    "Campaign Type",
    options=df['CampaignType'].unique(),
    default=df['CampaignType'].unique()
)

filtered_df = df[
    (df['CampaignChannel'].isin(channel)) &
    (df['CampaignType'].isin(campaign_type))
]

# ================================
# CONVERSION BY CHANNEL
# ================================
st.subheader("Conversion Rate by Campaign Channel")

channel_conversion = pd.crosstab(
    filtered_df['CampaignChannel'],
    filtered_df['Conversion'],
    normalize='index'
) * 100

channel_conversion.columns = ['No Conversion', 'Conversion']

st.bar_chart(channel_conversion['Conversion'])

# ================================
# HEATMAP (CHANNEL × TYPE)
# ================================
st.subheader("Campaign Performance Heatmap")

pivot = pd.pivot_table(
    filtered_df,
    values='Conversion',
    index='CampaignChannel',
    columns='CampaignType',
    aggfunc='mean'
) * 100

fig, ax = plt.subplots()
sns.heatmap(pivot, annot=True, fmt=".1f", ax=ax)
st.pyplot(fig)

# ================================
# FUNNEL ANALYSIS
# ================================
st.subheader("Marketing Funnel")

funnel = {
    "Stage": ["Visits", "Engagement", "Email Clicks", "Conversion"],
    "Value": [
        filtered_df['WebsiteVisits'].sum(),
        filtered_df['PagesPerVisit'].sum(),
        filtered_df['EmailClicks'].sum(),
        filtered_df['Conversion'].sum()
    ]
}

funnel_df = pd.DataFrame(funnel)
st.bar_chart(funnel_df.set_index("Stage"))

# ================================
# CUSTOMER SEGMENTATION
# ================================
st.subheader("Customer Segmentation")

fig2, ax2 = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x="Age",
    y="Income",
    hue="Conversion",
    ax=ax2
)
st.pyplot(fig2)

# ================================
# MODEL PREDICTION
# ================================
st.subheader("🔮 Predict Customer Conversion")

age = st.slider("Age", 18, 70)
income = st.number_input("Income", value=50000)
ad_spend = st.number_input("Ad Spend", value=1000)
time_on_site = st.slider("Time on Site", 1, 20)

# Dummy encoding (must match training)
input_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "AdSpend": ad_spend,
    "TimeOnSite": time_on_site
}])

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.success(f"Conversion Prediction: {'YES' if prediction==1 else 'NO'}")
        st.info(f"Probability: {prob:.2f}")

    except Exception as e:
        st.error("Model input mismatch. Ensure preprocessing matches training.")

# ================================
# RAW DATA VIEW
# ================================
st.subheader("Raw Data")
st.dataframe(filtered_df.head())