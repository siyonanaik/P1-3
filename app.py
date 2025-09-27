import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpld3
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from calculations import *
from apihandler import *
import csv

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
    st.header("Simple Moving Average (SMA) & Exponential Moving Average (EMA)")
    st.subheader("NVIDIA Stock Price with SMA (Periods: 50, 100, 200)")

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
        # title="NVIDIA Stock Price with SMA (50, 100, 200)"
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

    st.info("The Simple Moving Average (SMA) is a widely used technical indicator that calculates the average of a stock’s prices over a specific number of periods. By smoothing out daily price fluctuations, the SMA helps traders and analysts identify the overall trend of a stock, making it easier to distinguish short-term noise from meaningful movements. Commonly used SMA periods include 50-day, which reflects short-term trends, 100-day for medium-term trends, and 200-day for long-term trends, often serving as key levels of support or resistance in market analysis.")

    # EMA Calculation 

    st.subheader("NVIDIA Stock Price with EMA (Periods: 12, 26, 50, 200)")

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

    st.info("The Exponential Moving Average (EMA), on the other hand, places greater weight on recent prices, allowing it to respond more quickly to changes in market direction. This responsiveness makes the EMA particularly useful for detecting trends and reversals sooner than the SMA. Typical EMA periods include 12-day and 26-day for short-term trends, which are often used in combination to generate trading signals, as well as 50-day and 200-day EMAs that help identify intermediate and long-term market trends. By choosing the appropriate EMA periods, traders can effectively balance sensitivity to recent price movements with the overall trend of the market.")

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

    def plot_bollinger_bands(data, ax):
        bands = bollinger_bands(data=data, window=5, k=2)

        plt.plot(bands['SMA'], label="SMA", color='orange')
        plt.plot(bands['UpperBand'], label="Upper Band", color='green', linestyle='-')
        plt.fill_between(data.index, bands['UpperBand'], bands['LowerBand'], color='grey', alpha=0.4)
        plt.plot(bands['LowerBand'], label="Lower Band", color='red', linestyle='-')

    def plot_trends(data, ax):
        for i in range(1, len(data)):
            color = "green" if data["Close"].iloc[i] > data["Close"].iloc[i-1] else \
                    "red" if data["Close"].iloc[i] < data["Close"].iloc[i-1] else "grey"
            plt.plot(data.index[i-1:i+1], data["Close"].iloc[i-1:i+1], color=color, linewidth=1.8)

    def plot_candles(data, ax):
        base = data.index[0].toordinal()

        for idx, row in data.iterrows():

            x = idx.toordinal() - base
            # Candle color
            color = "green" if row.Close >= row.Open else "red"

            # Candle body
            lower = min(row.Open, row.Close)
            height = abs(row.Close - row.Open)
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.4, lower),
                    0.8,
                    height if height > 0 else 0.01,
                    facecolor=color,
                    edgecolor=color
                )
            )

            # Candle Wick
            ax.vlines(x, row.Low, row.High, color=color, linewidth=2)

        ax.set_xlim(-1, len(data))
        ax.set_xticks(range(0, len(data), max(1, len(data)//10)))
        ax.set_xticklabels(data.index.strftime("%Y-%m-%d")[::max(1, len(data)//10)])
        ax.autoscale_view()


    if user_ticker:
        # Getting 3 years of data from yfinance using 
        # user given ticker
        ticker = yf.Ticker(user_ticker)
        data = ticker.history(period="3Y")

        # Cleaning data
        # Reindexing to fill in weekends
        data.index = pd.to_datetime(data.index)
        full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(full_index)
        # Forward filling missing data
        data = data.ffill()

        indicators_plot = {
            "Bollinger Bands": plot_bollinger_bands,
            "Trends": plot_trends,
            "Candles": plot_candles
        }

        selected_options = st.multiselect(
            "Select indicators to display:",
            ["Bollinger Bands", "Trends", "Candles"]
        )

        #fig = plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(14, 8))
        if "Candles" not in selected_options:
            ax.plot(data['Close'], label="Price", color='blue')


        for option in selected_options:
            if option in indicators_plot:
                indicators_plot[option](data, ax)

        
        # Plotting closing price
        plt.title(f"Closing price of {user_ticker}")
        plt.legend()     

        # Turning static plot to interactive
        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=1000, width=1000)
        
#------------------------------------END OF YUAN WEI PART-----------------------------------

