# Description: Uses a Long Short-Term Memory (LSTM) neural network to predict
# the closing stock price of Apple Inc. (AAPL) using the past 60 days of prices.

import math
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use('fivethirtyeight')

# ---- Load data ----
raw = yf.download('AAPL', start='2012-01-01', end='2024-12-31')

# yfinance >= 0.2 returns a MultiIndex column structure — flatten it
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

print("Data shape:", df.shape)
print(df.head())

# ---- Visualize closing price history ----
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.tight_layout()
plt.show()

# ---- Prepare data ----
data = df.filter(['Close'])
dataset = data.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

training_data_len = int(len(scaled_data) * 0.8)

# Build training sequences
train_data = scaled_data[:training_data_len, :]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ---- Build LSTM model ----
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1),
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ---- Train ----
model.fit(x_train, y_train, batch_size=32, epochs=5)

# ---- Evaluate ----
test_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")

# ---- Visualize predictions ----
train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('LSTM Stock Price Prediction — AAPL')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'], label='Training Data')
plt.plot(valid['Close'], label='Actual Price')
plt.plot(valid['Predictions'], label='Predicted Price')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print(valid.tail(10))

# ---- Predict next day ----
last_60_days = data[-60:].values
last_60_scaled = scaler.transform(last_60_days)
X_next = np.array([last_60_scaled])
X_next = np.reshape(X_next, (X_next.shape[0], X_next.shape[1], 1))

pred_price = model.predict(X_next)
pred_price = scaler.inverse_transform(pred_price)
print(f"\nPredicted closing price for next trading day: ${pred_price[0][0]:.2f}")
