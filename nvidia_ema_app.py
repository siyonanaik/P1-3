import csv
import pandas as pd
import plotly.express as px
import streamlit as st
from ema_utils import exponential_moving_average

st.title("NVIDIA Stock Price with EMA (Periods: 12, 26, 50, 200)")

# ---- Load dataset manually ----
file_path = 'nvidia_cleaned.csv'
dates = []
close_prices = []

with open(file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        dates.append(row['Date'])
        close_prices.append(float(row['Close']))

# ---- EMA periods ----
periods = [12, 26, 50, 200]

# ---- Calculate EMAs ----
ema_dict = {f"EMA_{p}": exponential_moving_average(close_prices, p) for p in periods}

# ---- Save CSV ----
output_path = 'nvidia_ema.csv'
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['Date', 'Close'] + list(ema_dict.keys())
    writer.writerow(header)
    for i in range(len(dates)):
        row = [dates[i], close_prices[i]] + [ema_dict[key][i] for key in ema_dict]
        writer.writerow(row)


# ---- Create DataFrame ----
data = pd.DataFrame({'Date': pd.to_datetime(dates), 'Close': close_prices})
for key in ema_dict:
    data[key] = ema_dict[key]

# ---- Melt for Plotly ----
df_long = data.melt(
    id_vars=['Date'],
    value_vars=['Close'] + list(ema_dict.keys()),
    var_name='Series',
    value_name='Price'
)

# ---- Plotly Express ----
color_map = {
    'Close': 'orange',      # bright orange for close price
    'EMA_12': 'cyan',      # bright cyan
    'EMA_26': 'magenta',   # bright magenta
    'EMA_50': 'lime',      # bright green
    'EMA_200': 'yellow'    # bright yellow
}
fig = px.line(df_long, x='Date', y='Price', color='Series', color_discrete_map=color_map)
fig.update_traces(line=dict(width=1))

# ---- X-axis: show start and end dates ----
tick_dates = pd.date_range(start=data['Date'].iloc[0],
                           end=data['Date'].iloc[-1],
                           periods=10)
fig.update_xaxes(tickmode='array', tickvals=tick_dates, tickformat="%Y-%m-%d", tickangle=45)

# ---- Display in Streamlit ----
st.plotly_chart(fig, use_container_width=True)
