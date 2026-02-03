import pandas as pd

# --- 1. READING THE DATA ---

df = pd.read_csv('data.csv')

# --- 2. CLEANING THE DATA ---

# Drop the customerID column (it is not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric values
# 'coerce' -> if you find a null string or error, turn it into a NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill NaN values with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# --- 3. ENCODING ---

# Convert the Target (Churn) manually to 0 and 1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert all other categorical column into numbers automatically
# pd.get_dummies -> finds all text columns and turns them into 1s and 0s
df = pd.get_dummies(df, dtype=int)

print(df.head())
print(df.dtypes)

# --- 4. SPLITTING AND SCALING ---

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate Features (X) and Target (y)
# X = everything except "Churn"
# y = only "Churn"
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into Training (80%) and Testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Data Shape: ", X_train.shape)
print("Test Data Shape: ", X_test.shape)

# --- 5. BUILDING AND TRAINING THE NEURAL NETWORK ---

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Defining the model structure
model = Sequential([
    # Input layer + 1st Hidden Layer (32 neurons)
    Dense(32, activation='relu', input_shape=(45,)),

    # 2nd Hidden Layer (16 neurons)
    Dense(16, activation='relu'),

    # Output Layer (1 neuron for binary classification)
    Dense(1, activation='sigmoid')
])

# 2. Compile the model
# Optimizer: 'adam' (adjusts the learning rate automatically)
# Loss: 'binary_crossentropy' (standard math for Yes/No problems)
# Metrics: 'accuracy' (to see how many we get right)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Train the model
# epochs = 50 => it will go through the entire dataset 50 times
print("Starting training...")
history = model.fit(X_train, y_train, epochs = 50, batch_size = 32,  validation_split=0.2)

# --- 6. EVALUATION ---
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Evaluate on the Test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

# 2. Make predictions
# The model gives probabilities (e.g., 0.85). We need to convert them to 0 or 1.
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# 3. Print the detailed report
print("\n--Classification Report--")
print(classification_report(y_test, y_pred))

print("\n--Confusion Matrix--")
print(confusion_matrix(y_test, y_pred))

# --- 7. SAVING THE MODEL AND SCALER ---

import joblib

# 1. Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# 2. Save the model
model.save('churn_model.keras')

print("\nModel and Scaler saved successfully!")



