# financial_data_forecasting_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
data = pd.read_csv('cleaned_financial_data.csv', parse_dates=['Date'], index_col='Date')

# Use closing prices and additional features for forecasting
features = ['Close_TSLA', 'Volume_TSLA', 'High_TSLA', 'Low_TSLA']
data_tsla = data[features]

# Divide the dataset into training and testing sets
train_data, test_data = train_test_split(data_tsla, test_size=0.2, shuffle=False)

# ARIMA Model for Univariate Time Series with Statsmodels
arima_model = ARIMA(train_data['Close_TSLA'], order=(5, 1, 0))  # You might need to tune the order
arima_model_fit = arima_model.fit()
forecast_arima = arima_model_fit.forecast(steps=len(test_data))

# Prepare the Data for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # predict the Close_TSLA
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60  # 60 days sequence
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Build and Train the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Forecast Future Stock Prices with LSTM
predictions_lstm = model.predict(X_test)
predictions_lstm = scaler.inverse_transform(np.concatenate((predictions_lstm, np.zeros((predictions_lstm.shape[0], len(features) - 1))), axis=1))[:, 0]

# Calculate Evaluation Metrics for ARIMA and LSTM
mae_arima = mean_absolute_error(test_data['Close_TSLA'], forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test_data['Close_TSLA'], forecast_arima))
mape_arima = mean_absolute_percentage_error(test_data['Close_TSLA'], forecast_arima)

print(f"ARIMA - Mean Absolute Error (MAE): {mae_arima}")
print(f"ARIMA - Root Mean Squared Error (RMSE): {rmse_arima}")
print(f"ARIMA - Mean Absolute Percentage Error (MAPE): {mape_arima}")

mae_lstm = mean_absolute_error(test_data['Close_TSLA'][seq_length:], predictions_lstm)
rmse_lstm = np.sqrt(mean_squared_error(test_data['Close_TSLA'][seq_length:], predictions_lstm))
mape_lstm = mean_absolute_percentage_error(test_data['Close_TSLA'][seq_length:], predictions_lstm)

print(f"LSTM - Mean Absolute Error (MAE): {mae_lstm}")
print(f"LSTM - Root Mean Squared Error (RMSE): {rmse_lstm}")
print(f"LSTM - Mean Absolute Percentage Error (MAPE): {mape_lstm}")

# Plot the results for ARIMA
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Close_TSLA'], label='Test Data')
plt.plot(test_data.index, forecast_arima, label='ARIMA Forecast', color='red')
plt.title('Tesla Stock Price Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the results for LSTM
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[seq_length:], test_data['Close_TSLA'][seq_length:], label='Test Data')
plt.plot(test_data.index[seq_length:], predictions_lstm, label='LSTM Forecast', color='blue')
plt.title('Tesla Stock Price Forecast using LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
