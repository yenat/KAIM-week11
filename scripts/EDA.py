# financial_data_analysis_eda.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np

# Load the dataset
data = pd.read_csv('cleaned_financial_data.csv', parse_dates=['Date'], index_col='Date')

# Select relevant columns
features = ['Close_TSLA', 'Volume_TSLA', 'High_TSLA', 'Low_TSLA']
data_tsla = data[features]

# Check basic statistics
print(data_tsla.describe())

# Ensure appropriate data types and handle missing values
print(data_tsla.dtypes)
print(data_tsla.isnull().sum())
data_tsla.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = StandardScaler()
data_tsla_scaled = pd.DataFrame(scaler.fit_transform(data_tsla), columns=data_tsla.columns)

# Visualize the closing price over time
plt.figure(figsize=(12, 6))
plt.plot(data_tsla['Close_TSLA'], label='TSLA')
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate and plot the daily percentage change
data_tsla['Daily_Pct_Change'] = data_tsla['Close_TSLA'].pct_change()
plt.figure(figsize=(12, 6))
plt.plot(data_tsla['Daily_Pct_Change'], label='TSLA')
plt.title('Daily Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend()
plt.show()

# Analyze volatility
data_tsla['Rolling_Mean'] = data_tsla['Close_TSLA'].rolling(window=21).mean()
data_tsla['Rolling_Std'] = data_tsla['Close_TSLA'].rolling(window=21).std()
plt.figure(figsize=(12, 6))
plt.plot(data_tsla['Close_TSLA'], label='Close')
plt.plot(data_tsla['Rolling_Mean'], label='21-day Rolling Mean')
plt.plot(data_tsla['Rolling_Std'], label='21-day Rolling Std')
plt.title('Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Perform outlier detection
outliers = data_tsla[(data_tsla['Daily_Pct_Change'] > 0.05) | (data_tsla['Daily_Pct_Change'] < -0.05)]
plt.figure(figsize=(12, 6))
plt.plot(data_tsla['Daily_Pct_Change'], label='Daily Pct Change')
plt.scatter(outliers.index, outliers['Daily_Pct_Change'], color='red', label='Outliers')
plt.title('Outliers in Daily Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.legend()
plt.show()

# Decompose the time series into trend, seasonal, and residual components
decomposition = sm.tsa.seasonal_decompose(data_tsla['Close_TSLA'], model='additive', period=252)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# Calculate Value at Risk (VaR) and Sharpe Ratio
VaR = data_tsla['Daily_Pct_Change'].quantile(0.05)
sharpe_ratio = data_tsla['Daily_Pct_Change'].mean() / data_tsla['Daily_Pct_Change'].std() * (252 ** 0.5)
print(f"Value at Risk (5% quantile): {VaR}")
print(f"Sharpe Ratio: {sharpe_ratio}")

# Document key insights
insights = {
    "Overall Direction": "Summarize Tesla's stock price direction.",
    "Daily Returns Impact": "Discuss fluctuations and their implications.",
    "VaR": VaR,
    "Sharpe Ratio": sharpe_ratio
}
print(insights)
