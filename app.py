import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from calculations import *
from apihandler import *

# Set up the page configuration
st.set_page_config(
    page_title="Financial Trend Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Dashboard UI (Frontend) ---

st.title("Financial Trend Analysis Dashboard")

st.markdown("""
Welcome to the P1-3's historical & live financial data analysis dashboard! This tool allows you
to analyze stock tickers and get AI-powered insights.
""")

# --- Sidebar Menu for Navigation ---
st.sidebar.title("Dashboard Menu")
dashboard_selection = st.sidebar.radio(
    "Select a dashboard view:",
    ("Stock Price and RSI", "SMA & EMA", "Daily Returns", "Max Profit Calculation", "Trends Analysis")
)

# --- Display different dashboards based on selection For different Team Members---

#------------------------------------START OF THAW ZIN PART-------------------------------
if dashboard_selection == "Stock Price and RSI":
    st.markdown("---")
    st.header("Live Stock Ticker Analysis")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", "MSFT").upper()

    if ticker_symbol:
        try:
            # Fetch data for the last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            with st.spinner(f"Fetching data for {ticker_symbol}..."):
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            if stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                # --- Stock Price and RSI section ---
                st.subheader(f"Historical Price for {ticker_symbol}")
                fig_price = px.line(
                    stock_data,
                    x=stock_data.index,
                    y='Close',
                    title=f"Historical Price of {ticker_symbol}",
                    labels={'Close': 'Price ($)'}
                )
                st.plotly_chart(fig_price, use_container_width=True)

                # Calculate and plot RSI
                stock_data['RSI'] = calculate_rsi(stock_data)
                
                st.subheader(f"Relative Strength Index (RSI) for {ticker_symbol}")
                fig_rsi = px.line(
                    stock_data,
                    x=stock_data.index,
                    y='RSI',
                    title=f"RSI for {ticker_symbol}",
                    labels={'RSI': 'Value'},
                )
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                st.plotly_chart(fig_rsi, use_container_width=True)

                # --- RSI explanation with API call ---
                st.markdown("---")
                st.write("### What is RSI?")
                with st.spinner("Fetching RSI explanation from Hugging Face..."):
                    rsi_explanation = call_huggingface_api("What is the Relative Strength Index (RSI)? Explain it simply for a beginner.")
                st.info(rsi_explanation)

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")
            
#------------------------------------END OF THAW ZIN PART------------------------------------
            
            
#------------------------------------START OF SIYONA PART------------------------------------

elif dashboard_selection == "SMA & EMA":
    st.markdown("---")
    st.header("SMA & EMA Dashboard")
    st.info("The SMA & EMA dashboard is not yet implemented.")

#------------------------------------END OF SIYONA PART------------------------------------

#------------------------------------START OF KAI REI PART----------------------------------

elif dashboard_selection == "Daily Returns":
    st.markdown("---")
    st.header("Daily Returns Dashboard")
    st.info("The Daily Returns dashboard is not yet implemented.")

#------------------------------------END OF KAI REI PART------------------------------------

#------------------------------------START OF WYNN PART-------------------------------------

elif dashboard_selection == "Max Profit Calculation":
    st.markdown("---")
    st.header("Max Profit Calculation Dashboard")
    st.info("The Max Profit Calculation dashboard is not yet implemented.")
    
#------------------------------------END OF WYNN PART---------------------------------------

#------------------------------------START OF YUAN WEI PART---------------------------------

elif dashboard_selection == "Trends Analysis":
    st.markdown("---")
    st.header("Trends Analysis Dashboard")

    user_ticker = st.text_input("Enter a Ticker Symbol: (e.g. AAPL, GOOG ...)").upper()

    def plot_bollinger_bands(data):
        bands = bollinger_bands(data=data, window=5, k=2)

        plt.plot(bands['SMA'], label="SMA", color='orange')
        plt.plot(bands['UpperBand'], label="Upper Band", color='green', linestyle='-')
        plt.fill_between(data.index, bands['UpperBand'], bands['LowerBand'], color='grey', alpha=0.4)
        plt.plot(bands['LowerBand'], label="Lower Band", color='red', linestyle='-')

    def plot_trends(data):
        for i in range(1, len(data)):
            color = "green" if data["Close"].iloc[i] > data["Close"].iloc[i-1] else \
                    "red" if data["Close"].iloc[i] < data["Close"].iloc[i-1] else "grey"
            plt.plot(data.index[i-1:i+1], data["Close"].iloc[i-1:i+1], color=color, linewidth=1.8)

    if user_ticker:
        # Getting 3 years of data from yfinance using 
        # user given ticker
        ticker = yf.Ticker(user_ticker)
        data = ticker.history(period="3Y")

        indicators_plot = {
            "Bollinger Bands": plot_bollinger_bands,
            "Trends": plot_trends
        }

        selected_options = st.multiselect(
            "Select indicators to display:",
            ["Bollinger Bands", "Trends"]
        )

        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label="Price", color='blue')

        for option in selected_options:
            if option in indicators_plot:
                indicators_plot[option](data)


        # Plotting closing price
        plt.title(f"Closing price of {user_ticker}")
        plt.legend()
        fig = plt.gcf()       

        # Turning static plot to interactive
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=1000)
#------------------------------------END OF YUAN WEI PART-----------------------------------

