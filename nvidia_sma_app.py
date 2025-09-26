import csv
import pandas as pd
import plotly.express as px
import streamlit as st
from sma_utils import simple_moving_average  # <-- import the function

st.title("NVIDIA Stock Price with SMA (Periods: 50, 100, 200) - Manual Calculation")

# ---- Load dataset manually ----
file_path = 'nvidia_cleaned.csv'
dates = []
close_prices = []

with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        dates.append(row['Date'])
        close_prices.append(float(row['Close']))

# ---- Calculate SMAs using imported function ----
sma_50 = simple_moving_average(close_prices, 50)
sma_100 = simple_moving_average(close_prices, 100)
sma_200 = simple_moving_average(close_prices, 200)

# ---- Save CSV ----
output_path = 'nvidia_sma_manual.csv'
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Close', 'SMA_50', 'SMA_100', 'SMA_200'])
    for i in range(len(dates)):
        writer.writerow([dates[i], close_prices[i], sma_50[i], sma_100[i], sma_200[i]])


# ---- Create DataFrame ----
data = pd.DataFrame({
    'Date': pd.to_datetime(dates),
    'Close': close_prices,
    'SMA_50': sma_50,
    'SMA_100': sma_100,
    'SMA_200': sma_200
})

# ---- Melt for Plotly ----
df_long = data.melt(
    id_vars=['Date'],
    value_vars=['Close', 'SMA_50', 'SMA_100', 'SMA_200'],
    var_name='Series',
    value_name='Price'
)

# ---- Plotly Express ----
fig = px.line(
    df_long,
    x='Date',
    y='Price',
    color='Series',
    title="NVIDIA Stock Price with SMA (50, 100, 200)"
)
fig.update_traces(line=dict(width=1))

# ---- X-axis: show start and end dates ----
tick_dates = pd.date_range(start=data['Date'].iloc[0],
                           end=data['Date'].iloc[-1],
                           periods=10)

fig.update_xaxes(
    tickmode='array',
    tickvals=tick_dates,
    tickformat="%Y-%m-%d",
    tickangle=45
)

# ---- Display in Streamlit ----
st.plotly_chart(fig, use_container_width=True)
