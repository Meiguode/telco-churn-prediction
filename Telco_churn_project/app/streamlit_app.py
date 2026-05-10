import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Predictor")

# Load model
model = joblib.load('../models/churn_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
features = joblib.load('../models/feature_names.pkl')

st.markdown("### Enter customer details to predict churn risk")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 150.0, 70.0)
    total_charges = tenure * monthly_charges
    senior = st.selectbox("Senior Citizen", [0, 1])

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Calculate features
num_services = 3  # simplified
avg_monthly = total_charges / (tenure + 1)

if st.button("🔮 Predict Churn Risk"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'SeniorCitizen': [senior],
        'NumServices': [num_services],
        'AvgMonthlyCharges': [avg_monthly],
        'Contract_Month-to-month': [1 if contract == "Month-to-month" else 0],
        'Contract_One year': [1 if contract == "One year" else 0],
        'Contract_Two year': [1 if contract == "Two year" else 0],
        'PaymentMethod_Bank transfer (automatic)': [1 if payment == "Bank transfer (automatic)" else 0],
        'PaymentMethod_Credit card (automatic)': [1 if payment == "Credit card (automatic)" else 0],
        'PaymentMethod_Electronic check': [1 if payment == "Electronic check" else 0],
        'PaymentMethod_Mailed check': [1 if payment == "Mailed check" else 0],
        'InternetService_Fiber optic': [1 if internet == "Fiber optic" else 0],
        'InternetService_No': [1 if internet == "No" else 0],
        'gender_Male': [1 if gender == "Male" else 0]
    })
    
    # Align columns
    for col in features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[features]
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    
    # Show result
    st.markdown("---")
    if prob > 0.6:
        st.error(f"⚠️ **HIGH RISK: {prob:.1%}** chance of churning")
        st.markdown("**Recommendation:** Offer retention discount immediately")
    elif prob > 0.3:
        st.warning(f"⚠️ **MEDIUM RISK: {prob:.1%}** chance of churning")
        st.markdown("**Recommendation:** Send engagement email")
    else:
        st.success(f"✅ **LOW RISK: {prob:.1%}** chance of churning")
        st.markdown("**Recommendation:** Standard retention program")