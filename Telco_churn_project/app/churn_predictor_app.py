# ============================================
# TELCO CHURN PREDICTOR - INTERACTIVE DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #34495e;
        padding: 0.5rem;
    }
    .risk-high {
        background-color: #e74c3c;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background-color: #f39c12;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #27ae60;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 Telco Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for input
with st.sidebar:
    st.markdown("## 🎯 Customer Information")
    st.markdown("Enter customer details to predict churn risk")
    
    # Demographics
    st.markdown("### 👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    # Account Information
    st.markdown("### 📅 Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                  ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                                   "Credit card (automatic)"])
    
    # Services
    st.markdown("### 📡 Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    col1, col2 = st.columns(2)
    with col1:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with col2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)

# Main content area
col1, col2, col3 = st.columns(3)

# Simple risk score calculation (simplified for demo)
# In production, you'd load your trained model
risk_score = 0

# Rule-based risk assessment (for demo)
if contract == "Month-to-month":
    risk_score += 30
if payment_method == "Electronic check":
    risk_score += 20
if tenure < 6:
    risk_score += 20
if monthly_charges > 80:
    risk_score += 15
if internet_service == "Fiber optic" and online_security != "Yes":
    risk_score += 10
if paperless_billing == "Yes":
    risk_score += 5

risk_score = min(risk_score, 100)

with col1:
    st.markdown("### 📊 Churn Risk Score")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        title = {'text': "Risk Percentage"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if risk_score > 60 else "orange" if risk_score > 30 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Risk Category")
    if risk_score >= 60:
        st.markdown('<div class="risk-high"><h2>⚠️ HIGH RISK</h2><p>Immediate retention action needed</p></div>', unsafe_allow_html=True)
        recommendation = "🚨 Offer retention discount + free add-on service immediately"
    elif risk_score >= 30:
        st.markdown('<div class="risk-medium"><h2>⚠️ MEDIUM RISK</h2><p>Monitor closely</p></div>', unsafe_allow_html=True)
        recommendation = "📧 Send engagement email + check-in call next month"
    else:
        st.markdown('<div class="risk-low"><h2>✅ LOW RISK</h2><p>Standard retention program</p></div>', unsafe_allow_html=True)
        recommendation = "👍 Continue normal customer success activities"
    
    st.markdown(f"**Recommended Action:** {recommendation}")

with col3:
    st.markdown("### 💰 Business Impact")
    clv = 1500
    if risk_score >= 60:
        potential_loss = clv * 0.8
    elif risk_score >= 30:
        potential_loss = clv * 0.4
    else:
        potential_loss = clv * 0.1
    
    st.metric("Customer Lifetime Value", f"${clv:,.0f}")
    st.metric("Potential Loss if Churned", f"${potential_loss:,.0f}")
    st.metric("Retention Offer Budget", "$50")

st.markdown("---")

# Detailed analysis section
st.markdown("## 📈 Detailed Analysis")

# Create feature importance chart
fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Risk Factors", "Savings Opportunity"))

# Risk factors
risk_factors = pd.DataFrame({
    'Factor': ['Month-to-month contract', 'Electronic check', 'Short tenure', 'High monthly charges', 'No online security'],
    'Impact': [30, 20, 20, 15, 10]
})
fig.add_trace(go.Bar(x=risk_factors['Impact'], y=risk_factors['Factor'], 
                     orientation='h', marker_color='coral', text=risk_factors['Impact'], 
                     textposition='outside'), row=1, col=1)

# Savings calculation
targeted_customers = 1000
reduction_rate = 0.25
savings = targeted_customers * reduction_rate * clv
cost = targeted_customers * 50
net_savings = savings - cost

savings_data = pd.DataFrame({
    'Category': ['Campaign Cost', 'Expected Savings', 'Net Benefit'],
    'Amount': [cost, savings, net_savings]
})
fig.add_trace(go.Bar(x=savings_data['Category'], y=savings_data['Amount'], 
                     marker_color=['#e74c3c', '#2ecc71', '#3498db'], 
                     text=savings_data['Amount'].apply(lambda x: f'${x:,.0f}'), 
                     textposition='outside'), row=1, col=2)

fig.update_layout(height=500, showlegend=False, title_text="Business Case for Retention Campaign")
fig.update_xaxes(title_text="Impact Score", row=1, col=1)
fig.update_xaxes(title_text="Amount ($)", row=1, col=2)
st.plotly_chart(fig, use_container_width=True)

# Recommendations based on risk factors
st.markdown("## 💡 Personalized Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Immediate Actions")
    if contract == "Month-to-month":
        st.write("• **Convert to annual contract** - Offer 2 months free for 1-year commitment")
    if payment_method == "Electronic check":
        st.write("• **Enable autopay** - Offer $10/month discount for automatic payments")
    if tenure < 6:
        st.write("• **New customer onboarding** - Schedule welcome call and product tutorial")
    if monthly_charges > 80:
        st.write("• **Loyalty discount** - 15% off next 3 months for high-value customers")

with col2:
    st.markdown("### 📊 Long-term Strategy")
    st.write("• Implement quarterly customer health scores")
    st.write("• Create churn prediction alerts for customer success team")
    st.write("• A/B test retention offers on high-risk segments")
    st.write("• Build customer feedback loop for service improvements")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    Built with ❤️ using Streamlit | Powered by Machine Learning | © 2024
</div>
""", unsafe_allow_html=True)
