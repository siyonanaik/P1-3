import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta


# --- Streamlit page configuration ---
st.set_page_config(page_title="Stock SMA & EMA Viewer", layout="wide")
st.title("📈 Stock SMA & EMA Viewer")

# --- Ticker input ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").strip().upper()

if ticker:
    # --- Fetch data ---
    # start_date = "2023-01-01"
    # end_date = datetime.today().strftime("%Y-%m-%d")

    # end date = today's date
    end_date = datetime.now()

    # start date = 3 years ago from today
    start_date = end_date - timedelta(days=3*365)

    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        st.error(f"No data found for ticker {ticker}. Please check the symbol.")
    else:
        # Reset index
        df.reset_index(inplace=True)

        # --- Calculate SMA using rolling().mean() ---
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        # --- Calculate EMA ---
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # --- SMA Chart ---
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], mode='lines', name='SMA 50'))
        fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA100'], mode='lines', name='SMA 100'))
        fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], mode='lines', name='SMA 200'))
        fig_sma.update_layout(
            title=f"{ticker} SMA Chart (from {start_date})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_sma, use_container_width=True)

        # --- EMA Chart ---
        fig_ema = go.Figure()
        fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['EMA12'], mode='lines', name='EMA 12'))
        fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['EMA26'], mode='lines', name='EMA 26'))
        fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], mode='lines', name='EMA 50'))
        fig_ema.add_trace(go.Scatter(x=df['Date'], y=df['EMA200'], mode='lines', name='EMA 200'))
        fig_ema.update_layout(
            title=f"{ticker} EMA Chart (from {start_date})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_ema, use_container_width=True)
