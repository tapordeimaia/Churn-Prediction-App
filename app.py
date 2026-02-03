import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the saved Model and Scaler
# This simulates starting up a server that has the model ready
model = load_model('churn_model.keras')
scaler = joblib.load('scaler.pkl')

def predict_churn(new_data):
    # 2. Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([new_data])

    # 3. Scale the data
    input_scaled = scaler.transform(input_df)

    # 4. Make the prediction
    prediction_prob = model.predict(input_scaled)
    prediction = (prediction_prob > 0.5).astype(int)[0][0]

    return prediction, prediction_prob[0][0]

new_customer = {
    'SeniorCitizen': 0, 'tenure': 12, 'MonthlyCharges': 70.5, 'TotalCharges': 840.0,
    'gender_Female': 1, 'gender_Male': 0,
    'Partner_No': 0, 'Partner_Yes': 1,
    'Dependents_No': 1, 'Dependents_Yes': 0,
    'PhoneService_No': 0, 'PhoneService_Yes': 1,
    'MultipleLines_No': 0, 'MultipleLines_No phone service': 0, 'MultipleLines_Yes': 1,
    'InternetService_DSL': 0, 'InternetService_Fiber optic': 1, 'InternetService_No': 0,
    'OnlineSecurity_No': 1, 'OnlineSecurity_No internet service': 0, 'OnlineSecurity_Yes': 0,
    'OnlineBackup_No': 1, 'OnlineBackup_No internet service': 0, 'OnlineBackup_Yes': 0,
    'DeviceProtection_No': 1, 'DeviceProtection_No internet service': 0, 'DeviceProtection_Yes': 0,
    'TechSupport_No': 1, 'TechSupport_No internet service': 0, 'TechSupport_Yes': 0,
    'StreamingTV_No': 0, 'StreamingTV_No internet service': 0, 'StreamingTV_Yes': 1,
    'StreamingMovies_No': 0, 'StreamingMovies_No internet service': 0, 'StreamingMovies_Yes': 1,
    'Contract_Month-to-month': 1, 'Contract_One year': 0, 'Contract_Two year': 0,
    'PaperlessBilling_No': 0, 'PaperlessBilling_Yes': 1,
    'PaymentMethod_Bank transfer (automatic)': 0, 'PaymentMethod_Credit card (automatic)': 0,
    'PaymentMethod_Electronic check': 1, 'PaymentMethod_Mailed check': 0
}

result, probability = predict_churn(new_customer)

print("\n --- PREDICTION RESULT ---")
print(f"Churn Probability: {probability:.2%}")
if (result == 1) :
    print("WARNING: This customer is likely to CHURN!")
else:
    print("This customer is likey to STAY")