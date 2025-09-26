import pandas as pd
import csv
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
                 fontsize=9, color="black")

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
# def analyze_trends(data):
#     """Example placeholder for trend analysis calculation"""
#     pass
#------------------------------------END OF YUAN WEI PART------------------------------------



