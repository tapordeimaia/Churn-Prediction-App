# 📱 Customer Churn Prediction App

A Machine Learning application that predicts whether a customer will churn (leave the service) based on their subscription details.

## 🚀 Overview
This project uses **TensorFlow** to build a deep learning neural network trained on the Telco Customer Churn dataset. The model is deployed via a **Streamlit** web application, allowing users to input customer demographics and service details to receive a real-time churn risk assessment.

## 🛠️ Technologies Used
* **Python**
* **TensorFlow / Keras** (Deep Learning)
* **scikit-learn** (Data Preprocessing & Scaling)
* **Pandas & NumPy** (Data Manipulation)
* **Streamlit** (Web Interface)

## 📊 Model Performance
* **Accuracy:** 79%
* **Precision (Non-Churn):** 84%
* **Recall (Churn):** 55%

## 💻 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/tapordeimaia/churn-prediction-app.git](https://github.com/tapordeimaia/churn-prediction-app.git)
    cd churn-prediction-app
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run dashboard.py
    ```
