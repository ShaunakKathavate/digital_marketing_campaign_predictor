# AI-Powered Customer Conversion Prediction & Marketing ROI Optimization

## 📌 Problem Statement
Businesses spend heavily on digital marketing but struggle to identify which customers are likely to convert. This leads to inefficient ad spend and low ROI.

## 🎯 Objective
Build a machine learning system to:
- Predict customer conversion probability
- Identify high-performing campaigns
- Optimize marketing spend (ROAS)

---

## 📊 Dataset Overview

Features include:
- Demographics: Age, Gender, Income
- Campaign Data: Channel, Type, AdSpend
- Engagement: WebsiteVisits, TimeOnSite, EmailClicks
- Historical: PreviousPurchases, LoyaltyPoints

Target:
- `Conversion (0/1)`

---

## 🔍 Key Insights

- PPC campaigns show highest conversion efficiency
- EmailClicks is the strongest predictor of conversion
- Returning customers convert 2–3x more
- High AdSpend ≠ high conversion (inefficiency detected)
- Social media drives traffic but low conversions

---

## 🧠 ML Approach

- Data Cleaning & Preprocessing
- Feature Engineering
- Model Training:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluation:
  - Accuracy
  - Precision / Recall
  - ROC-AUC

---

## 📈 Results

- Achieved high predictive performance on conversion classification
- Identified top drivers influencing customer decisions
- Enabled data-driven campaign optimization

---

## 🖥️ Dashboard

Interactive Streamlit app showing:
- Conversion trends
- Campaign performance
- ROI analysis
- Real-time predictions

---

## 📂 Project Structure

(Your folder tree here)

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/project-name.git
cd project-name

pip install -r requirements.txt
streamlit run app/app.py