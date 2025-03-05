import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load your dataset
# Assuming you have your historical data in a DataFrame called 'data'
data = pd.read_csv('../data/cleaned_financial_data.csv')

# Ensure the Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the Date column as the index
data.set_index('Date', inplace=True)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Prepare the data for the LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Example sequence length
X_train, y_train = create_sequences(train_data['Close_TSLA'].values, seq_length)
X_test, y_test = create_sequences(test_data['Close_TSLA'].values, seq_length)

# Reshape the data for the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Generate future timestamps (e.g., for the next 12 months)
future_timestamps = pd.date_range(start=test_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')

# Create an empty array to store the predictions
future_predictions = []

# Use the last available data point for predictions
last_data_point = X_test[-1]

# Forecast for each future timestamp
for _ in future_timestamps:
    # Predict the next value
    next_value = model.predict(last_data_point.reshape(1, last_data_point.shape[0], 1))
    future_predictions.append(next_value[0][0])

    # Update the last_data_point with the new prediction
    last_data_point = np.append(last_data_point[1:], next_value[0][0]).reshape(-1, 1)

# Convert predictions to a DataFrame
future_predictions_df = pd.DataFrame(data=future_predictions, index=future_timestamps, columns=['Predicted_Close'])

