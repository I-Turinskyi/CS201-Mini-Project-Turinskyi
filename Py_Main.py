import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_filename = "aapl_us_d.csv"

df = pd.read_csv(csv_filename, parse_dates=['Date'], index_col='Date')

df.columns = df.columns.str.capitalize()

df = df.sort_index()

df = df.loc['2022-01-01':'2026-01-01']

# Retain required OHLCV columns and drop missing values
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(df)

# Calculating Moving Averages
short_window = 20
long_window = 50

# Calculating 20-day and 50-day Simple Moving Averages of Close prices
df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

# Setting position to 1 (In Market) when Short SMA > Long SMA, else 0 (Cash)
df['Position'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)

# Detecting position state changes and making Stock Signals: +1 = Buy , 0 = Hold, -1 = Sell
df['Signal'] = df['Position'].diff()

# Daily market percentage return
df['Market_Returns'] = df['Close'].pct_change()

# Finding returns from the chosen strategy, shifting 1 day ahead
df['Strategy_Returns'] = df['Market_Returns'] * df['Position'].shift(1)

# Compounded cumulative growth over time for the Market(basic holding of a stock) and for our Strategy
df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod() - 1
df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod() - 1
