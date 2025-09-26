import pandas as pd
import math
from datetime import timedelta

# --- Function to compute SMA using pandas ---
def compute_sma(df, window=5):
    df["SMA"] = df["Close"].rolling(window=window).mean()
    return df

# --- Helper to compare lists with NaNs ---
def compare_lists_with_nan(list1, list2):
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if a is None or (isinstance(a, float) and math.isnan(a)):
            if b is not None:
                return False
        else:
            if a != b:
                return False
    return True

# --- Test Case 1: Basic 5-day SMA ---
data1 = pd.DataFrame({
    "Date": pd.date_range("2025-01-01", periods=5),
    "Close": [10, 20, 30, 40, 50]
})
result1 = compute_sma(data1)
expected1 = [None, None, None, None, 30.0]
assert compare_lists_with_nan(result1["SMA"].tolist(), expected1)

# --- Test Case 2: Less than 5 days ---
data2 = pd.DataFrame({
    "Date": pd.date_range("2025-01-01", periods=3),
    "Close": [100, 200, 300]
})
result2 = compute_sma(data2)
expected2 = [None, None, None]
assert compare_lists_with_nan(result2["SMA"].tolist(), expected2)

# --- Test Case 3: Constant prices ---
data3 = pd.DataFrame({
    "Date": pd.date_range("2025-01-01", periods=6),
    "Close": [50, 50, 50, 50, 50, 50]
})
result3 = compute_sma(data3)
expected3 = [None, None, None, None, 50.0, 50.0]
assert compare_lists_with_nan(result3["SMA"].tolist(), expected3)

# --- Test Case 4: Negative values ---
data4 = pd.DataFrame({
    "Date": pd.date_range("2025-01-01", periods=5),
    "Close": [-10, 0, 10, 20, 30]
})
result4 = compute_sma(data4)
expected4 = [None, None, None, None, 10.0]
assert compare_lists_with_nan(result4["SMA"].tolist(), expected4)

# --- Test Case 5: 6-month chunk logic validation ---
dates = pd.date_range("2025-01-01", periods=400)  # >1 year
closes = list(range(400))
data5 = pd.DataFrame({"Date": dates, "Close": closes})
data5 = compute_sma(data5)

chunk_start = dates[0]
chunk_duration = timedelta(days=182)
chunk_end = chunk_start + chunk_duration
indices = [i for i, d in enumerate(dates) if chunk_start <= d <= chunk_end]
assert indices[0] == 0
assert indices[-1] <= 182

print("All 5 pandas-based test cases passed ✅")
