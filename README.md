# 📊 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-78.5%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC-83%25-blue)

## 🎯 Problem
Predict which telecom customers will churn and identify why.

**Key result:** 83% AUC | $2.8M annual loss at risk

## 📁 Files in This Repository

| File | Description |
|------|-------------|
| `app/streamlit_app.py` | Interactive churn predictor app |
| `notebooks/01_complete_analysis.py` | Full analysis pipeline |
| `data/telco_churn_cleaned.csv` | Cleaned dataset for Tableau |
| `data/churn_by_*.csv` | Pre-aggregated views for dashboards |
| `models/churn_model.pkl` | Trained XGBoost model |
| `images/dashboard.pdf` | Tableau dashboard PDF export |

## 🔍 Top Findings

| Factor | Impact on Churn |
|--------|-----------------|
| Fiber optic internet | +31% |
| Month-to-month contract | +25% |
| Electronic check payment | +2% |

## 💰 Business Impact

- **Current churn rate:** 26.6%
- **Annual loss:** $2.8M
- **Potential savings:** $500K+ with targeted retention

## 🚀 Run the App Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
