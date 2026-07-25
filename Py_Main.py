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

# Calculating Moving Averages
short_window = 4
long_window = 100

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

# Getting the final overall change/effect of the strategy:
market_perf = df['Cumulative_Market'].iloc[-1] * 100
strategy_perf = df['Cumulative_Strategy'].iloc[-1] * 100
# 2. Counting Buy and Sell events
num_buys  = (df['Signal'] == 1).sum()
num_sells = (df['Signal'] == -1).sum()

print("       STRATEGY PERFORMANCE SUMMARY       ")
print(f"Target Asset:                AAPL")
print(f"Time Horizon:                2022 - 2026")
print(f"Buy & Hold Market Return:    {market_perf:.2f}%")
print(f"SMA Crossover Strategy:      {strategy_perf:.2f}%")
print(f"Total Buy Orders:            {num_buys}")
print(f"Total Sell Orders:           {num_sells}")


# Plotting the strategy execution chart
plt.figure(figsize=(12, 6))

# Primary price curve and trendlines
plt.plot(df.index, df['Close'], label='AAPL Close Price', alpha=0.35, color='gray')
plt.plot(df.index, df['SMA_short'], label=f'SMA {short_window} (Short)', color='blue', linewidth=1.5)
plt.plot(df.index, df['SMA_long'], label=f'SMA {long_window} (Long)', color='orange', linewidth=1.5)

# Separate Buy (+1) and Sell (-1) markers
buy_signals = df[df['Signal'] == 1]
sell_signals = df[df['Signal'] == -1]

plt.plot(buy_signals.index, buy_signals['SMA_short'], '^', markersize=9, color='green', label='Buy Signal', lw=0)
plt.plot(sell_signals.index, sell_signals['SMA_short'], 'v', markersize=9, color='red', label='Sell Signal', lw=0)

plt.title('AAPL Moving Average Crossover Strategy (Offline Data Execution)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()