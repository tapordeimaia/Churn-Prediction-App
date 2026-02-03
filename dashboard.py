import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the model and the scaler
model = load_model('churn_model.keras')
scaler = joblib.load('scaler.pkl')

# 2. Title and Description
st.title("Customer Churn Prediction")
st.write("Enter customer details below to predict if they will leave the service.")

# 3. Create the Input Format
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Months with Company (Tenure)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# 4. Predict Button

if st.button("Predict Churn Risk"):
    input_data = {
        'SeniorCitizen': 0, 'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'gender_Female': 0, 'gender_Male': 1,  # Default to Male for demo
        'Partner_No': 1, 'Partner_Yes': 0,
        'Dependents_No': 1, 'Dependents_Yes': 0,
        'PhoneService_No': 0, 'PhoneService_Yes': 1,
        'MultipleLines_No': 1, 'MultipleLines_No phone service': 0, 'MultipleLines_Yes': 0,
        'InternetService_DSL': 0, 'InternetService_Fiber optic': 0, 'InternetService_No': 0,
        'OnlineSecurity_No': 1, 'OnlineSecurity_No internet service': 0, 'OnlineSecurity_Yes': 0,
        'OnlineBackup_No': 1, 'OnlineBackup_No internet service': 0, 'OnlineBackup_Yes': 0,
        'DeviceProtection_No': 1, 'DeviceProtection_No internet service': 0, 'DeviceProtection_Yes': 0,
        'TechSupport_No': 1, 'TechSupport_No internet service': 0, 'TechSupport_Yes': 0,
        'StreamingTV_No': 1, 'StreamingTV_No internet service': 0, 'StreamingTV_Yes': 0,
        'StreamingMovies_No': 1, 'StreamingMovies_No internet service': 0, 'StreamingMovies_Yes': 0,
        'Contract_Month-to-month': 0, 'Contract_One year': 0, 'Contract_Two year': 0,
        'PaperlessBilling_No': 0, 'PaperlessBilling_Yes': 0,
        'PaymentMethod_Bank transfer (automatic)': 0, 'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0, 'PaymentMethod_Mailed check': 0
    }

    if contract == "Month-to-month":
        input_data['Contract_Month-to-month'] = 1
    elif contract == "One year":
        input_data['Contract_One year'] = 1
    else:
        input_data['Contract_Two year'] = 1

    if internet_service == "DSL":
        input_data['InternetService_DSL'] = 1
    elif internet_service == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1
    else:
        input_data['InternetService_No'] = 1

    if "Bank" in payment_method:
        input_data['PaymentMethod_Bank transfer (automatic)'] = 1
    elif "Credit" in payment_method:
        input_data['PaymentMethod_Credit card (automatic)'] = 1
    elif "Electronic" in payment_method:
        input_data['PaymentMethod_Electronic check'] = 1
    else:
        input_data['PaymentMethod_Mailed check'] = 1

    if paperless == "Yes":
        input_data['PaperlessBilling_Yes'] = 1
    else:
        input_data['PaperlessBilling_No'] = 1

    # Create DataFrame and Scale
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction_prob = model.predict(input_scaled)[0][0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction_prob > 0.5:
        st.error(f"High Risk {prediction_prob:.2%} probability of Churning")
    else:
        st.success(f"Low Risk {prediction_prob:.2%} probability of Churning")