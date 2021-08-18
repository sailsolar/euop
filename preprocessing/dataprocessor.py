import sys

import pandas as pd
from datetime import datetime
import pandas_datareader as web

import blackSholes as bS

# set initial parameters
company = 'TWTR'
expiry = '17-07-2015'
strike_price = 26
start = '2015-07-01 0:0:1.1'
end = '2015-07-17 0:0:1.1'

starting = datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f")
ending = datetime.strptime(end, "%Y-%m-%d %H:%M:%S.%f")
today = datetime.now()

# Fetch dataset from Yahoo Finance API
# df = web.DataReader(company, 'yahoo', starting, today)
df = pd.read_csv('../spy.csv')

# Sorting data by date
df = df.sort_values(by="Date", ignore_index=True)

# Drop any NAN values in the dataframe - 2 samples dropped from our data
# count = df.isnull().sum()
df.dropna(inplace=True)

# Volatility calculation as std of stock returns over the period of days the market operated for the data sample
days = len(df.Date.unique())
volatility_constant = df['StockPrice'].std()/days



# Calculate best before
def convert_to_datetime(s):
    return s.days


df['Date'] = df['Date'].map(lambda s: datetime.strptime(s, "%Y-%m-%d"))
df['ExpirationDate'] = df['ExpirationDate'].map(lambda s: datetime.strptime(s, "%Y-%m-%d"))
df['BestBefore'] = abs(df['ExpirationDate']-df['Date']).map(convert_to_datetime)

# Getting the risk-free factor as the 10-year US treasury yield from Yahoo Finance API
riskfree_rate = (web.DataReader("^TNX", 'yahoo', today.replace(day=today.day-1), today)['Close'].iloc[-1])/100


# Preparing Final Data
def calculate_blacksholes(x):
    return bS.EuropeanCall(x[4], x[3], volatility_constant, x[8], riskfree_rate).price


df['bsValues'] = df.apply(calculate_blacksholes, axis=1)
# df1 = df[df.isna().any(axis=1)]
df.dropna(inplace=True)

# df["bsValues"].fillna("0.01", inplace = True)

df.to_csv('raw.csv', index=False)
