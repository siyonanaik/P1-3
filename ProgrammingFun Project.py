import pandas as pd
import matplotlib.pyplot as plt

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