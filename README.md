# 🚀 AI-Powered Customer Conversion Prediction System

An end-to-end Machine Learning project designed to optimize digital marketing campaigns by predicting customer conversions using demographic, behavioral, and campaign engagement data.

This project demonstrates a complete industry-standard ML workflow including:

* Data preprocessing pipeline
* Business-focused EDA
* Feature engineering
* Ensemble model training
* Hyperparameter tuning
* Threshold optimization
* Interactive Streamlit dashboard
* Deployment-ready modular architecture

---

# 📌 Problem Statement

Digital marketing campaigns generate large volumes of customer interaction data, but identifying high-conversion customers remains a challenge.

The goal of this project is to build a robust machine learning system capable of predicting whether a customer will convert based on:

* demographic attributes
* campaign engagement
* behavioral activity
* historical purchase behavior

This enables businesses to:

✅ improve campaign targeting
✅ increase conversion rates
✅ reduce customer acquisition cost
✅ maximize Return on Advertising Spend (ROAS)

---

# 🎯 Business Objectives

* Predict customer conversion probability
* Identify high-performing marketing channels
* Discover customer segments with high purchase intent
* Optimize campaign spending
* Improve marketing ROI through data-driven targeting

---

# 🧠 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Context-Aware Imputation
   ↓
Feature Engineering
   ↓
Outlier Treatment
   ↓
Encoding + Scaling
   ↓
Ensemble Model Training
   ↓
Hyperparameter Tuning
   ↓
Threshold Optimization
   ↓
Model Evaluation
   ↓
Streamlit Dashboard
```

---

# 📂 Project Structure

```text
ai-sme-solutions/
│
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_data.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_business_insights.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── utils.py
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── reports/
│   ├── business_insights.pdf
│   └── screenshots/
│
├── models/
│   ├── final_model.pkl
│   └── data_preprocessor.pkl
│
└── requirements.txt
```

---

# 📊 Dataset Description

## Demographic Features

| Feature    | Description                |
| ---------- | -------------------------- |
| CustomerID | Unique customer identifier |
| Age        | Customer age               |
| Gender     | Male/Female                |
| Income     | Annual income              |

---

## Marketing Features

| Feature          | Description              |
| ---------------- | ------------------------ |
| CampaignChannel  | Marketing channel        |
| CampaignType     | Campaign objective       |
| AdSpend          | Advertising spend        |
| ClickThroughRate | CTR                      |
| ConversionRate   | Campaign conversion rate |

---

## Engagement Features

| Feature       | Description           |
| ------------- | --------------------- |
| WebsiteVisits | Website visits        |
| PagesPerVisit | Avg pages per session |
| TimeOnSite    | Avg time on site      |
| SocialShares  | Social shares         |
| EmailOpens    | Email opens           |
| EmailClicks   | Email clicks          |

---

## Historical Features

| Feature           | Description     |
| ----------------- | --------------- |
| PreviousPurchases | Prior purchases |
| LoyaltyPoints     | Loyalty score   |

---

## Target Variable

| Feature    | Description                      |
| ---------- | -------------------------------- |
| Conversion | 1 = Converted, 0 = Not Converted |

---

# ⚙️ Data Preprocessing Pipeline

The preprocessing pipeline includes:

✅ duplicate removal
✅ datatype correction
✅ context-aware missing value imputation
✅ outlier capping (99th percentile)
✅ feature engineering
✅ categorical encoding
✅ numerical scaling
✅ reusable sklearn pipeline serialization

---

# 🛠️ Feature Engineering

Created advanced business-driven features such as:

* EngagementScore
* CustomerValue
* EmailCTR
* DeepEngagement
* Spend_per_Click
* ValuePerVisit

These engineered features significantly improved model performance.

---

# 🤖 Models Used

## Base Models

* XGBoost
* LightGBM
* Random Forest

---

## Final Model

### Soft Voting Ensemble

Combines:

* XGBoost
* LightGBM
* Random Forest

for robust prediction performance.

---

# 🔍 Hyperparameter Tuning

Implemented using:

* RandomizedSearchCV
* Stratified K-Fold Cross Validation
* ROC-AUC optimization

---

# 📈 Model Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve
* Threshold Optimization

---

# 🎯 Threshold Optimization

Business-aware threshold optimization was used instead of relying on the default 0.5 threshold.

This improved:

* recall
* precision balance
* marketing decision quality

---

# 📊 Streamlit Dashboard

Interactive dashboard features:

✅ Dataset overview
✅ Business insights
✅ Visual analytics
✅ Model evaluation
✅ Real-time conversion prediction

---

# 🚀 Running the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ai-sme-solutions.git

cd ai-sme-solutions
```

---

# 2️⃣ Create Virtual Environment

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

# 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Run Preprocessing

```bash
python src/preprocessing.py
```

---

# 5️⃣ Train Model

```bash
python src/model.py
```

---

# 6️⃣ Run Evaluation

```bash
python src/evaluation.py
```

---

# 7️⃣ Launch Streamlit Dashboard

```bash
streamlit run app/app.py
```

---

# 📦 Tech Stack

## Languages & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

# 📌 Key Business Insights

✔ Customers with higher engagement scores are more likely to convert
✔ Email clicks strongly influence conversion probability
✔ Certain campaign channels outperform others significantly
✔ Customer loyalty strongly correlates with repeat conversions
✔ Optimized thresholding improves campaign targeting efficiency

---

# 📈 Future Improvements

* Real-time prediction API
* MLflow experiment tracking
* Docker deployment
* CI/CD integration
* Cloud deployment
* Drift monitoring
* Automated retraining pipeline

---

# Project Summary

### AI-Powered Customer Conversion Prediction System

Developed an end-to-end machine learning system to predict customer conversions for digital marketing campaigns using ensemble learning techniques. Built a modular preprocessing and training pipeline with feature engineering, hyperparameter tuning, threshold optimization, and interactive Streamlit dashboard deployment. Achieved high ROC-AUC performance using XGBoost, LightGBM, and Random Forest ensemble models.

---

# 📬 Contact

Feel free to connect for:

* Machine Learning
* Data Science
* AI Engineering
* Analytics Projects

## GitHub

(https://github.com/ShaunakKathavate)
