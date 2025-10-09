import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
import yfinance as yf


# -------------------------------------------------------------------------
# TO DO: Add more financial calculations as needed
# Author: P1-3 Team
# -------------------------------------------------------------------------


#------------------------------------START OF THAWZIN PART-----------------------------------

# calculate rsi using j.welles wilder method
def calculate_rsi(df: pd.DataFrame, periods: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) using J. Welles Wilder's method.
    Uses numpy for robust array iteration, avoiding iloc enlargement errors.
    """
    if 'Close' not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    # 1. Calculate price differences (deltas)
    delta = df['Close'].diff()
    
    # 2. Separate gains (up moves) and losses (down moves)
    gain = delta.clip(lower=0).fillna(0)
    loss = (-delta).clip(lower=0).fillna(0)
    
    # --- Start of Array-Based Calculation ---
    
    # 3. Initialize NumPy arrays to hold averages and RSI result
    avg_gain = np.zeros_like(gain.values)
    avg_loss = np.zeros_like(loss.values)
    rsi = np.full_like(gain.values, np.nan) # Fill with NaN initially

    # 4. Calculate the initial average (SMA for the first 'periods')
    # Note: We start the average from index 'periods'
    if len(gain) > periods:
        avg_gain[periods] = gain.iloc[1:periods+1].mean()
        avg_loss[periods] = loss.iloc[1:periods+1].mean()
    else:
        # Not enough data, return NaN
        return pd.Series(rsi, index=df.index)

    # 5. Apply Wilder's Smoothing (EMA) for subsequent periods
    # Formula: NewAvg = (PrevAvg * (N - 1) + CurrentValue) / N
    for i in range(periods + 1, len(df)):
        avg_gain[i] = (avg_gain[i-1] * (periods - 1) + gain.iloc[i]) / periods
        avg_loss[i] = (avg_loss[i-1] * (periods - 1) + loss.iloc[i]) / periods

    # 6. Calculate RS and RSI only for the smoothed values
    rs = avg_gain[periods:] / avg_loss[periods:]
    rsi[periods:] = 100 - (100 / (1 + rs))
    
    # 7. Convert the NumPy array back to a Pandas Series with correct index
    return pd.Series(rsi, index=df.index)

#------------------------------------END OF THAWZIN PART-------------------------------------



#------------------------------------START OF SIYONA PART------------------------------------


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
def calculate_daily_returns(close):
    """
    Calculate daily returns for each day.
    
    Parameters:
    - close: list of close values
    
    Returns:
    - list of daily returns (%)
    """

    # Input validation
    if not close:
        raise ValueError("Close prices list cannot be empty")
    
    if any(price <= 0 for price in close):
        negative_prices = [price for price in close if price < 0]
        zero_prices = [price for price in close if price == 0]
        
        if negative_prices:
            raise ValueError(f"Stock prices cannot be negative. Found negative prices: {negative_prices}")
        if zero_prices:
            raise ZeroDivisionError(f"Stock prices cannot be zero (division by zero). Found zero prices: {zero_prices}")


    # Day 1 daily return = 0
    dailyreturn_lst = [0]

    for i in range(1, len(close)): 
        dailyreturn_lst.append(((close[i] - close[i-1]) / close[i-1]) * 100)

    return dailyreturn_lst

def calculate_true_range(high, low, close):
    """
    Calculate true range for each day. (needed to calculate average true range)
    
    Parameters:
    - high: list of high values
    - low: list of low values
    - close: list of close values
    
    Returns:
    - list of true range values
    """
    # Input validation
    if not high or not low or not close:
        raise ValueError("All price lists (high, low, close) must be non-empty")
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError(f"All price lists must have the same length. High: {len(high)}, Low: {len(low)}, Close: {len(close)}")
    
    # Check for negative prices in all lists
    all_prices = high + low + close
    if any(price < 0 for price in all_prices):
        negative_high = [price for price in high if price < 0]
        negative_low = [price for price in low if price < 0]
        negative_close = [price for price in close if price < 0]
        
        error_msg = "Stock prices cannot be negative. "
        if negative_high:
            error_msg += f"Negative high prices: {negative_high}. "
        if negative_low:
            error_msg += f"Negative low prices: {negative_low}. "
        if negative_close:
            error_msg += f"Negative close prices: {negative_close}."
        
        raise ValueError(error_msg.strip())
    
    # Check that high >= low, low <= high, high >= close for each day
    for i in range(len(high)):
        if high[i] < low[i]:
            raise ValueError(f"High price ({high[i]}) cannot be less than low price ({low[i]}) on day {i}")
        if low[i] > close[i]:
            raise ValueError(f"Low price ({low[i]}) cannot be more than close price ({close[i]}) on day {i}")
        if close[i] > high[i]:
            raise ValueError(f"Close price ({close[i]}) cannot be more than high price ({high[i]}) on day {i}")
    
    # Check for zero prices
    close_zero_prices = [price for price in close if price == 0]
    high_zero_prices = [price for price in high if price == 0]
    low_zero_prices = [price for price in low if price == 0]
    if close_zero_prices or high_zero_prices or low_zero_prices:
        raise ValueError(f"Stock prices cannot be zero. Found zero prices: {close_zero_prices + high_zero_prices + low_zero_prices}")


    # Day 1 TR = High - Low, since there are no previous values
    tr_day_1 = high[0] - low[0]
    tr_lst = [tr_day_1]

    for i in range(1, len(close)):
        tr_1 = high[i] - low[i]
        tr_2 = abs(high[i] - close[i-1])
        tr_3 = abs(low[i] - close[i-1])
        tr = max(tr_1, tr_2, tr_3)
        tr_lst.append(tr)
    
    return tr_lst

def calculate_average_true_range(tr_lst):
    """
    Calculate average true range for each day.
    
    Parameters:
    - tr_lst: list of true range values
    
    Returns:
    - list of average true range values
    """
    # Input validation
    if not tr_lst:
        raise ValueError("True range list cannot be empty")
    
    if any(tr < 0 for tr in tr_lst):
        negative_tr = [tr for tr in tr_lst if tr < 0]
        raise ValueError(f"True range values cannot be negative. Found negative values: {negative_tr}")


    # 7 Days
    time_period = 7
    # Day 1 ATR = Day 1 TR
    atr_lst = [tr_lst[0]]
    # previous_atr = Day 1 TR
    previous_atr = tr_lst[0]

    for i in range(1, len(tr_lst)):
        if i < (time_period - 1):
            # For Days 1 to time_period, ATR is calculated as Day 1: day_1_tr / 1, Day 2: (day_1_tr + day_2_tr) / 2, Day 3: (day_1_tr + day_2_tr + day_3_tr) / 3 ...
            atr = ((previous_atr * i) + tr_lst[i]) / (i + 1)
        else:  
            # Standard ATR formula after initial period
            atr = ((previous_atr * (time_period - 1)) + tr_lst[i]) / time_period

        atr_lst.append(atr)      
        previous_atr = atr
    
    return atr_lst
#------------------------------------END OF KAI REI PART-------------------------------------



#------------------------------------START OF WYNN PART--------------------------------------

# Max profit calculation
def max_profit_with_days(prices, dates):
    """
    Calculate maximum profit for a given stock price period 

    Parameters:
    - Prices : list of price values 
    - Dates : list of corresponding dates
    """

    profit = 0
    transactions = []  # to store transaction details
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
            transactions.append({
                "Buy Date": dates[i - 1],
                "Buy Price": prices[i - 1],
                "Sell Date": dates[i],
                "Sell Price": prices[i],
                "Profit": prices[i] - prices[i - 1]
            })
    return profit, transactions



#------------------------------------END OF WYNN PART----------------------------------------



#------------------------------------START OF YUAN WEI PART----------------------------------
def bollinger_bands(data: pd.DataFrame, window: int = 5, k: int = 2) -> pd.DataFrame:
    '''
    Calculates Bollinger Bands for given price data

    Parameters:
    data: A stock's dataframe containing Close price
    window: Bollinger Band lookback period / number of data points, Default = 5
    k: Standard Deviation Value, Default = 2

    Returns:
    pandas.DataFrame consisting of SMA, UpperBand, LowerBand values


    '''
    prices = np.asarray(data['Close'])
    n = len(prices)

    # Creating empty array to hold results
    sma = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    # Sliding window calculation
    for i in range(window - 1, n):
        window_data = prices[i - window + 1 : i + 1] # Getting first window elements
        
        mean = sum(window_data) / window
        variance = sum((x-mean) ** 2 for x in window_data) / window
        std = variance ** 0.5

        sma[i] = mean
        upper_band[i] = mean + k * std
        lower_band[i] = mean - k * std

    return pd.DataFrame({
        'SMA': sma,
        'UpperBand': upper_band,
        'LowerBand': lower_band
    }, index=data.index)

#------------------------------------END OF YUAN WEI PART------------------------------------



