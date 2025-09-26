import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Load Data ---
dates, closes = [], []
with open("nvidia_cleaned.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dates.append(datetime.strptime(row["Date"], "%Y-%m-%d"))
        closes.append(float(row["Close"]))

# --- Sliding Window 5-Day SMA ---
window = 5
sma = []
for i in range(len(closes)):
    if i + 1 >= window:
        avg = sum(closes[i - window + 1:i + 1]) / window
        sma.append(avg)
    else:
        sma.append(None)

# --- Plot 12-Month Chunks (3 graphs) ---
chunk_start = dates[0]
chunk_duration = timedelta(days=365)  # 12 months ~ 365 days
chunk_number = 1

while chunk_start < dates[-1] and chunk_number <= 3:
    chunk_end = chunk_start + chunk_duration
    # Get indices in this chunk
    indices = [i for i, d in enumerate(dates) if chunk_start <= d <= chunk_end]
    if not indices:
        chunk_start = chunk_end
        chunk_number += 1
        continue

    chunk_dates = [dates[i] for i in indices]
    chunk_closes = [closes[i] for i in indices]
    chunk_sma = [sma[i] for i in indices]

    # --- Plot this chunk ---
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(chunk_dates, chunk_closes, label="Close Price", color="blue")
    ax.plot(chunk_dates, chunk_sma, label="5-Day SMA", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"NVIDIA Closing Price vs 5-Day SMA (Year {chunk_number})")
    ax.legend()
    fig.autofmt_xdate()  # formats and rotates dates
    plt.tight_layout()
    plt.show()

    # Move to next chunk
    chunk_start = chunk_end
    chunk_number += 1
