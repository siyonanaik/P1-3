import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# TO DO: Add more financial calculations as needed
# Author: P1-3 Team
# -------------------------------------------------------------------------


#------------------------------------START OF THAWZIN PART-----------------------------------

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI) for a given stock price data.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

#------------------------------------END OF THAWZIN PART-------------------------------------



#------------------------------------START OF SIYONA PART------------------------------------
# (Currently empty — Siyona can add functions here)
#------------------------------------END OF SIYONA PART--------------------------------------



#------------------------------------START OF KAI REI PART-----------------------------------
# def calculate_daily_returns(data):
#     """Example placeholder for daily returns calculation"""
#     return data['Close'].pct_change()
#------------------------------------END OF KAI REI PART-------------------------------------



#------------------------------------START OF WYNN PART--------------------------------------
# def calculate_max_profit(data):
#     """Example placeholder for max profit calculation"""
#     pass
#------------------------------------END OF WYNN PART----------------------------------------



#------------------------------------START OF YUAN WEI PART----------------------------------
def bollinger_bands(data, window=5, k=2):
    '''
    Calculates Bollinger Bands for given price data/

    '''
    prices = np.asarray(data['Close'])
    n = len(prices)

    # Creating empty array to hold results
    sma = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    # Sliding window calculation
    for i in range(window - 1, n):
        window_data = prices[i - window + 1 : i + 1] # Getting first 20 elements
        mean = np.mean(window_data)
        std = np.std(window_data)

        sma[i] = mean
        upper_band[i] = mean + k * std
        lower_band[i] = mean - k * std

    return pd.DataFrame({
        'SMA': sma,
        'UpperBand': upper_band,
        'LowerBand': lower_band
    }, index=data.index)

def plot_bollinger_bands(data):
    bands = bollinger_bands(data=data, window=5, k=2)

    plt.plot(bands['SMA'], label="SMA", color='orange')
    plt.plot(bands['UpperBand'], label="Upper Band", color='green', linestyle='-')
    plt.fill_between(data.index, bands['UpperBand'], bands['LowerBand'], color='grey', alpha=0.4)
    plt.plot(bands['LowerBand'], label="Lower Band", color='red', linestyle='-')

def plot_trends(data):
    for i in range(1, len(data)):
        color = "green" if data["Close"].iloc[i] > data["Close"].iloc[i-1] else \
                "red" if data["Close"].iloc[i] < data["Close"].iloc[i-1] else "grey"
        plt.plot(data.index[i-1:i+1], data["Close"].iloc[i-1:i+1], color=color, linewidth=1.8)
#------------------------------------END OF YUAN WEI PART------------------------------------



