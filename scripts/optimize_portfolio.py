import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load your dataset
data = pd.read_csv('../data/cleaned_financial_data.csv')

# Ensure the Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the Date column as the index
data.set_index('Date', inplace=True)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Example sequence length

# Prepare the data for BND
X_train_bnd, y_train_bnd = create_sequences(train_data['Close_BND'].values, seq_length)
X_test_bnd, y_test_bnd = create_sequences(test_data['Close_BND'].values, seq_length)

# Reshape the data for the LSTM model
X_train_bnd = np.reshape(X_train_bnd, (X_train_bnd.shape[0], X_train_bnd.shape[1], 1))
X_test_bnd = np.reshape(X_test_bnd, (X_test_bnd.shape[0], X_test_bnd.shape[1], 1))

# Define the LSTM model for BND
model_bnd = Sequential()
model_bnd.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_bnd.shape[1], X_train_bnd.shape[2])))
model_bnd.add(LSTM(units=50))
model_bnd.add(Dropout(0.2))
model_bnd.add(Dense(1))

model_bnd.compile(optimizer='adam', loss='mean_squared_error')
model_bnd.summary()

# Train the model for BND
history_bnd = model_bnd.fit(X_train_bnd, y_train_bnd, epochs=20, batch_size=32, validation_split=0.2)

# Generate future timestamps (e.g., for the next 12 months)
future_timestamps = pd.date_range(start=test_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')

# Generate future predictions for BND
future_predictions_bnd = []
last_data_point_bnd = X_test_bnd[-1]

for _ in future_timestamps:
    next_value_bnd = model_bnd.predict(last_data_point_bnd.reshape(1, last_data_point_bnd.shape[0], 1))
    future_predictions_bnd.append(next_value_bnd[0][0])
    last_data_point_bnd = np.append(last_data_point_bnd[1:], next_value_bnd[0][0]).reshape(-1, 1)

future_predictions_df_bnd = pd.DataFrame(data=future_predictions_bnd, index=future_timestamps, columns=['Predicted_Close_BND'])


# Prepare the data for SPY
X_train_spy, y_train_spy = create_sequences(train_data['Close_SPY'].values, seq_length)
X_test_spy, y_test_spy = create_sequences(test_data['Close_SPY'].values, seq_length)

# Reshape the data for the LSTM model
X_train_spy = np.reshape(X_train_spy, (X_train_spy.shape[0], X_train_spy.shape[1], 1))
X_test_spy = np.reshape(X_test_spy, (X_test_spy.shape[0], X_test_spy.shape[1], 1))

# Define the LSTM model for SPY
model_spy = Sequential()
model_spy.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_spy.shape[1], X_train_spy.shape[2])))
model_spy.add(LSTM(units=50))
model_spy.add(Dropout(0.2))
model_spy.add(Dense(1))

model_spy.compile(optimizer='adam', loss='mean_squared_error')
model_spy.summary()

# Train the model for SPY
history_spy = model_spy.fit(X_train_spy, y_train_spy, epochs=20, batch_size=32, validation_split=0.2)

# Generate future predictions for SPY
future_predictions_spy = []
last_data_point_spy = X_test_spy[-1]

for _ in future_timestamps:
    next_value_spy = model_spy.predict(last_data_point_spy.reshape(1, last_data_point_spy.shape[0], 1))
    future_predictions_spy.append(next_value_spy[0][0])
    last_data_point_spy = np.append(last_data_point_spy[1:], next_value_spy[0][0]).reshape(-1, 1)

future_predictions_df_spy = pd.DataFrame(data=future_predictions_spy, index=future_timestamps, columns=['Predicted_Close_SPY'])


# Prepare the data for TSLA
X_train_tsla, y_train_tsla = create_sequences(train_data['Close_TSLA'].values, seq_length)
X_test_tsla, y_test_tsla = create_sequences(test_data['Close_TSLA'].values, seq_length)

# Reshape the data for the LSTM model
X_train_tsla = np.reshape(X_train_tsla, (X_train_tsla.shape[0], X_train_tsla.shape[1], 1))
X_test_tsla = np.reshape(X_test_tsla, (X_test_tsla.shape[0], X_test_tsla.shape[1], 1))

# Define the LSTM model for TSLA
model_tsla = Sequential()
model_tsla.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_tsla.shape[1], X_train_tsla.shape[2])))
model_tsla.add(LSTM(units=50))
model_tsla.add(Dropout(0.2))
model_tsla.add(Dense(1))

model_tsla.compile(optimizer='adam', loss='mean_squared_error')
model_tsla.summary()

# Train the model for TSLA
history_tsla = model_tsla.fit(X_train_tsla, y_train_tsla, epochs=20, batch_size=32, validation_split=0.2)

# Generate future timestamps (e.g., for the next 12 months)
future_timestamps = pd.date_range(start=test_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')

# Generate future predictions for TSLA
future_predictions_tsla = []
last_data_point_tsla = X_test_tsla[-1]

for _ in future_timestamps:
    next_value_tsla = model_tsla.predict(last_data_point_tsla.reshape(1, last_data_point_tsla.shape[0], 1))
    future_predictions_tsla.append(next_value_tsla[0][0])
    last_data_point_tsla = np.append(last_data_point_tsla[1:], next_value_tsla[0][0]).reshape(-1, 1)

future_predictions_df_tsla = pd.DataFrame(data=future_predictions_tsla, index=future_timestamps, columns=['Predicted_Close_TSLA'])

