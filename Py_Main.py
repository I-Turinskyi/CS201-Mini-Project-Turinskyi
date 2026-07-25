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