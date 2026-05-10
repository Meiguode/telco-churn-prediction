#!/usr/bin/env python3
"""
Telco Customer Churn Analysis - Data Preparation for Tableau/PowerBI
FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import os

warnings.filterwarnings('ignore')

# Create folders
os.makedirs('../data', exist_ok=True)
os.makedirs('../models', exist_ok=True)

print("=" * 60)
print("TELCO CUSTOMER CHURN - DATA PREPARATION")
print("=" * 60)

# Step 1: Load and clean data
print("\n📂 Loading data...")
df = pd.read_csv('../data/Telco-Customer-Churn.csv')

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Remove customerID
df = df.drop('customerID', axis=1)

print(f"✓ Cleaned data: {len(df)} rows, {len(df.columns)} columns")

# Step 2: Create features for visualization
print("\n📊 Creating features for visualization...")

# Create numeric churn column
df['Churn_Num'] = (df['Churn'] == 'Yes').astype(int)

# Create tenure groups
df['TenureGroup_Num'] = pd.cut(df['tenure'], 
                                bins=[0, 3, 6, 12, 24, 72], 
                                labels=[1, 2, 3, 4, 5])

# Create useful calculated columns
df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsHighRisk'] = ((df['Contract'] == 'Month-to-month') & 
                    (df['PaymentMethod'] == 'Electronic check')).astype(int)

# Count services used
service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['NumServices'] = df[service_cols].apply(lambda x: (x != 'No').sum(), axis=1)

print("✓ Created new features")

# Step 3: Export data for Tableau/PowerBI
print("\n💾 Exporting data for visualization...")
df.to_csv('../data/telco_churn_cleaned.csv', index=False)
print("✓ Saved: ../data/telco_churn_cleaned.csv")

# Step 4: Create aggregated datasets
print("\n📈 Creating aggregated datasets...")

# Churn by contract
churn_by_contract = df.groupby('Contract').agg({
    'Churn_Num': 'mean',
    'tenure': 'count'
}).rename(columns={'Churn_Num': 'ChurnRate', 'tenure': 'CustomerCount'})
churn_by_contract.to_csv('../data/churn_by_contract.csv')
print("✓ Saved: churn_by_contract.csv")

# Churn by payment method
churn_by_payment = df.groupby('PaymentMethod').agg({
    'Churn_Num': 'mean',
    'tenure': 'count'
}).rename(columns={'Churn_Num': 'ChurnRate', 'tenure': 'CustomerCount'})
churn_by_payment.to_csv('../data/churn_by_payment.csv')
print("✓ Saved: churn_by_payment.csv")

# Churn by tenure group
churn_by_tenure = df.groupby('TenureGroup_Num').agg({
    'Churn_Num': 'mean',
    'tenure': 'count'
}).rename(columns={'Churn_Num': 'ChurnRate', 'tenure': 'CustomerCount'})
churn_by_tenure.to_csv('../data/churn_by_tenure.csv')
print("✓ Saved: churn_by_tenure.csv")

# Churn by contract and payment method
churn_by_contract_payment = df.groupby(['Contract', 'PaymentMethod']).agg({
    'Churn_Num': 'mean',
    'tenure': 'count'
}).rename(columns={'Churn_Num': 'ChurnRate', 'tenure': 'CustomerCount'})
churn_by_contract_payment.to_csv('../data/churn_by_contract_payment.csv')
print("✓ Saved: churn_by_contract_payment.csv")

# Churn by internet service and security
churn_by_internet_security = df.groupby(['InternetService', 'OnlineSecurity']).agg({
    'Churn_Num': 'mean',
    'tenure': 'count'
}).rename(columns={'Churn_Num': 'ChurnRate', 'tenure': 'CustomerCount'})
churn_by_internet_security.to_csv('../data/churn_by_internet_security.csv')
print("✓ Saved: churn_by_internet_security.csv")

# Monthly charges distribution
df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], 
                                    bins=[0, 30, 50, 70, 90, 120], 
                                    labels=['<30', '30-50', '50-70', '70-90', '90+'])
churn_by_monthly_charges = df.groupby('MonthlyChargesGroup').agg({
    'Churn_Num': 'mean',
    'tenure': 'count'
}).rename(columns={'Churn_Num': 'ChurnRate', 'tenure': 'CustomerCount'})
churn_by_monthly_charges.to_csv('../data/churn_by_monthly_charges.csv')
print("✓ Saved: churn_by_monthly_charges.csv")

# Step 5: Build prediction model
print("\n🤖 Building prediction model...")

# Create a SEPARATE dataframe for modeling (don't reuse X from before)
model_df = df.copy()

# Select numeric features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 
                    'NumServices', 'AvgMonthlyCharges']

# Select categorical features (from the ORIGINAL dataframe)
categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'gender']

# Combine features
X_numeric = model_df[numeric_features]
X_categorical = model_df[categorical_features]

# One-hot encode categorical features
X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)

# Combine numeric and encoded categorical
X = pd.concat([X_numeric, X_categorical_encoded], axis=1)
y = model_df['Churn_Num']

print(f"✓ Created {X.shape[1]} features for modeling")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {len(X_train)} rows")
print(f"✓ Test set: {len(X_test)} rows")

# Train XGBoost model
print("\n🚀 Training XGBoost model...")
model = XGBClassifier(
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   • AUC Score: {auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print(f"\n📊 Top 10 Most Important Features:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['importance']:.4f} - {row['feature']}")

# Calculate business impact
churn_rate = y.mean()
total_customers = len(y)
clv = 1500  # Customer Lifetime Value
annual_loss = total_customers * churn_rate * clv

print(f"\n💰 Business Impact Analysis:")
print(f"   • Current churn rate: {churn_rate:.1%}")
print(f"   • Total customers: {total_customers:,}")
print(f"   • Estimated annual loss: ${annual_loss:,.0f}")

# Save model and artifacts
import joblib
joblib.dump(model, '../models/churn_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(list(X.columns), '../models/feature_names.pkl')

print("\n✓ Saved model to: ../models/churn_model.pkl")
print("✓ Saved scaler to: ../models/scaler.pkl")
print("✓ Saved feature names to: ../models/feature_names.pkl")

print("\n" + "=" * 60)
print("✅ COMPLETE SUCCESS!")
print("=" * 60)
print("\n📁 Files created for Tableau/PowerBI:")
print("   • telco_churn_cleaned.csv - Main dataset for dashboards")
print("   • churn_by_contract.csv - Pre-aggregated views")
print("   • churn_by_payment.csv")
print("   • churn_by_tenure.csv")
print("   • churn_by_contract_payment.csv")
print("   • churn_by_internet_security.csv")
print("   • churn_by_monthly_charges.csv")
print("\n📁 Model files saved:")
print("   • models/churn_model.pkl - For Streamlit app")
print("   • models/scaler.pkl")
print("   • models/feature_names.pkl")
print("\n💡 NEXT STEPS:")
print("   1. Open Tableau Public")
print("   2. Connect to 'telco_churn_cleaned.csv'")
print("   3. Create your dashboard visualizations")
print("   4. Run Streamlit app for interactive predictions")