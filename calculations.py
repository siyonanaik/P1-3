import pandas as pd
import csv


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

def simple_moving_average(prices, period):
    """
    Calculate simple moving average (SMA) for a list of prices.
    
    Parameters:
    - prices: list of floats
    - period: int, window size
    
    Returns:
    - list of SMA values (None for positions with insufficient data)
    """
    sma = []
    window_sum = 0
    for i in range(len(prices)):
        window_sum += prices[i]
        if i >= period:
            window_sum -= prices[i - period]
            sma.append(window_sum / period)
        elif i == period - 1:
            sma.append(window_sum / period)
        else:
            sma.append(None)  # Not enough data points yet
    return sma


# ema_utils.py

def exponential_moving_average(prices, period):
    """
    Calculate Exponential Moving Average (EMA) for a list of prices.
    
    Parameters:
    - prices: list of floats
    - period: int, smoothing period
    
    Returns:
    - list of EMA values (None for initial positions with insufficient data)
    """
    ema = []
    multiplier = 2 / (period + 1)
    for i in range(len(prices)):
        if i < period - 1:
            ema.append(None)
        elif i == period - 1:
            sma = sum(prices[:period]) / period
            ema.append(sma)
        else:
            ema_today = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_today)
    return ema




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
# def analyze_trends(data):
#     """Example placeholder for trend analysis calculation"""
#     pass
#------------------------------------END OF YUAN WEI PART------------------------------------



