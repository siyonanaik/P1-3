import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# TO DO: Add more financial calculations as needed
# Author: P1-3 Team
# -------------------------------------------------------------------------


#------------------------------------START OF THAWZIN PART-----------------------------------

# Ensure this function replaces the one in your utility section
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
    time_period = 7
    atr_lst = [tr_lst[0]]
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
# Load data
file_path = "nvidia_cleaned.csv"   
df = pd.read_csv(file_path)

# Max profit calculation
def max_profit_with_days(prices, dates):
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


prices = df["Close"].values
dates = pd.to_datetime(df["Date"]).values  

total_profit, transactions = max_profit_with_days(prices, dates)
print(f"Total Max Profit: {total_profit:.2f}")
print(f"Number of transactions: {len(transactions)}")


transactions_df = pd.DataFrame(transactions)

# Sorting
transactions_df = transactions_df.sort_values(by="Buy Date")

# Find top 5 profits
top5_df = transactions_df.sort_values(by="Profit", ascending=False).head(5)

# Plot profits
plt.figure(figsize=(12,6))
plt.plot(transactions_df["Buy Date"], transactions_df["Profit"],
        linestyle='-', color='lightpink', label="Total Profits")

# Highlight top 5 profits
plt.scatter(top5_df["Buy Date"], top5_df["Profit"], 
            color="lightblue", s=100, label="Top 5 Profits")

# Writting top 5
for _, row in top5_df.iterrows():
    plt.annotate(f"{row['Profit']:.2f}",
                 (row['Buy Date'], row['Profit']),
                 textcoords="offset points", xytext=(0,10), ha='center',
                 fontsize=9, color="blue")

# Labels and formatting
plt.xlabel("Buy Date")
plt.ylabel("Profit")
plt.title("Profit per Transaction")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
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

#------------------------------------END OF YUAN WEI PART------------------------------------



