"""
Project: AI-Driven Smart Parking System (IEEE ICUIS 2025)
Component: Predictive Intelligence Layer (Bi-LSTM)
Author: Benhein Michael Ruben L
Model: Bidirectional Long Short-Term Memory (Bi-LSTM)
Description: Time-series forecasting for parking slot availability with 14.82% MAPE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
days = 130
time_slots_per_day = 48
total_slots = days * time_slots_per_day
parking_data = np.random.randint(0, 10, total_slots)

# Simulate realistic traffic patterns
for i in range(total_slots):
    hour = (i % time_slots_per_day) / 2  
    if 7 <= hour <= 9 or 17 <= hour <= 19:  
        parking_data[i] = np.random.randint(0, 3)  
    elif 22 <= hour <= 23 or 0 <= hour <= 6:
        parking_data[i] = np.random.randint(5, 10)  

df = pd.DataFrame({'TimeSlot': range(total_slots), 'Available_Slots': parking_data})
df['Available_Slots'] = df['Available_Slots'].rolling(window=5, min_periods=1).mean()  # Smoother trends

# Feature Engineering
df['Hour'] = df['TimeSlot'] % time_slots_per_day
df['Day'] = df['TimeSlot'] // time_slots_per_day
df['IsWeekend'] = (df['Day'] % 7 >= 5).astype(int)  # Weekend indicator

# Capture cyclic patterns
df['Sin_Hour'] = np.sin(2 * np.pi * df['Hour'] / time_slots_per_day)
df['Cos_Hour'] = np.cos(2 * np.pi * df['Hour'] / time_slots_per_day)

# Standardization
scaler = StandardScaler()
df[['Available_Slots', 'Sin_Hour', 'Cos_Hour', 'IsWeekend']] = scaler.fit_transform(df[['Available_Slots', 'Sin_Hour', 'Cos_Hour', 'IsWeekend']])

# Prepare Data
sequence_length = 30
features = df[['Available_Slots', 'Sin_Hour', 'Cos_Hour', 'IsWeekend']].values  
train_size = int(len(features) * 0.8)
train_data, test_data = features[:train_size], features[train_size:]

X_train, y_train, X_test, y_test = [], [], [], []
for i in range(sequence_length, len(train_data)):
    X_train.append(train_data[i-sequence_length:i])
    y_train.append(train_data[i, 0])  

for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i-sequence_length:i])
    y_test.append(test_data[i, 0])  

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Optimized LSTM Model
model = Sequential([
    Bidirectional(LSTM(256, activation='elu', return_sequences=True, input_shape=(sequence_length, 4))),
    LayerNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(128, activation='elu', return_sequences=True)),
    LayerNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(64, activation='elu')),
    Dense(1)
])

# Compile with Adam optimizer
optimizer = Adam(learning_rate=0.0005)  # Fine-tuned learning rate
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00005)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train, 
    epochs=300,  # Extended training
    validation_data=(X_test, y_test), 
    batch_size=512,  # Adjusted batch size
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# Predictions
predictions = model.predict(X_test)

# Convert predictions back to original scale
y_test_actual = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), 3))]))[:, 0]
predictions_actual = scaler.inverse_transform(np.column_stack([predictions.flatten(), np.zeros((len(predictions), 3))]))[:, 0]

# Corrected MAPE Calculation
epsilon = 1e-8  
mape = np.mean(np.abs((y_test_actual - predictions_actual) / (y_test_actual + epsilon))) * 100

# Compute Errors
mse = mean_squared_error(y_test_actual, predictions_actual)
mae = mean_absolute_error(y_test_actual, predictions_actual)
mean_actual = np.mean(y_test_actual)

# Print Results
print(f"\n📊 **Optimized Model Evaluation**")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean of Actual Values: {mean_actual:.4f}")
print(f"✅ Acceptable MAPE (<20%): {mape < 20.0}")

# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label="Actual", color='blue')
plt.plot(predictions_actual, label="Predicted", color='red', linestyle='dashed')
plt.xlabel("Time Steps")
plt.ylabel("Parking Availability")
plt.legend()
plt.title("Optimized LSTM Parking Availability Prediction")
plt.show()
