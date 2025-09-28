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
import csv
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Set up the page configuration
st.set_page_config(
    page_title="Financial Trend Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Dashboard UI (Frontend) ---

st.title("Financial Trend Analysis Dashboard")

st.markdown("""
Welcome to the P1-3's live & historical financial data analysis dashboard! This tool allows you
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

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    # --- NEW TIME RANGE SELECTION LOGIC ---
    TIME_RANGES = {
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),      # Approximation
        "3M": timedelta(days=90),      # Approximation
        "1Y": timedelta(days=365),
        "3Y": timedelta(days=365 * 3), # Approximation
    }

    selected_range_label = st.radio(
        "Select Time Range:",
        options=list(TIME_RANGES.keys()),
        index=4, # Default to 1Y
        horizontal=True
    )
    
    if ticker_symbol:
        try:
            # Fetch data based on user selected range
            end_date = datetime.now()
            
            # Calculate start date based on selected label
            range_delta = TIME_RANGES.get(selected_range_label, timedelta(days=365))
            start_date = end_date - range_delta
            
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {selected_range_label}..."):
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            if stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                # --- Candlestick and Volume Subplot Section (Current Ticker) ---
                st.subheader(f"Candlestick Price and Volume for {ticker_symbol}")
                
                # Create subplots: 2 rows, shared X-axis, Price (row 1) is taller than Volume (row 2)
                fig_combined = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3]
                )
                
                # 1. Candlestick Chart (Row 1)
                candlestick = go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Price'
                )
                fig_combined.add_trace(candlestick, row=1, col=1)
                
                # 2. Volume Bar Chart (Row 2)
                # Determine bar color based on daily movement (Close > Open = Green; else Red)
                volume_colors = ['green' if stock_data['Close'].iloc[i] > stock_data['Open'].iloc[i] else 'red'
                                 for i in range(len(stock_data))]

                volume_bar = go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    marker_color=volume_colors,
                    name='Volume'
                )
                fig_combined.add_trace(volume_bar, row=2, col=1)

                # Update layout for a cleaner financial look
                fig_combined.update_layout(
                    title_text=f"Historical Price and Volume Analysis for {ticker_symbol}",
                    xaxis_rangeslider_visible=False, # Hide the main range slider
                    xaxis2_title="Date",
                    yaxis_title="Price ($)",
                    yaxis2_title="Volume",
                    height=700,
                    template='plotly_white'
                )
                
                # Finalize axis visibility and ranges
                fig_combined.update_xaxes(showgrid=True, row=1, col=1)
                fig_combined.update_yaxes(showgrid=True, row=1, col=1)
                fig_combined.update_yaxes(showgrid=True, row=2, col=1)
                
                st.plotly_chart(fig_combined, use_container_width=True)

                # --- RSI Calculation and Plot ---
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

                with st.spinner("Fetching RSI explanation..."):
                    try:
                        rsi_explanation =  WAITcall_huggingface_api(
                            "What is the Relative Strength Index (RSI)? Explain it simply for a beginner."
                        )
                    except Exception as e:
                        # Fallback hardcoded explanation
                        rsi_explanation = (
                            "The Relative Strength Index (RSI) is a popular momentum indicator used "
                            "in technical analysis. It measures the speed and size of recent price "
                            "changes to identify whether a stock is overbought or oversold. "
                            
                            "RSI values range from 0 to 100: generally, above 70 means overbought, "
                            "and below 30 means oversold."
                        )

                st.info(rsi_explanation)

                # --- Specific NVIDIA Chart (Reads from File) ---
                st.markdown("---")
                st.subheader("Candlestick Chart from Sample NVIDIA Data (Read from Preprocessed Dataset)")
                
                try:
                    # Load data directly from the external file
                    nvidia_df = pd.read_csv('nvidia_cleaned.csv')
                    
                    # Convert 'Date' column to datetime objects
                    nvidia_df['Date'] = pd.to_datetime(nvidia_df['Date'])

                    # Create the Candlestick chart for the sample data
                    fig_nvidia = go.Figure(data=[go.Candlestick(
                        x=nvidia_df['Date'],
                        open=nvidia_df['Open'],
                        high=nvidia_df['High'],
                        low=nvidia_df['Low'],
                        close=nvidia_df['Close'],
                        name='NVDA Price'
                    )])

                    # Customize the layout
                    fig_nvidia.update_layout(
                        title='NVIDIA Sample Data Candlestick (2022-09-12 to 2022-09-18)',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        xaxis_rangeslider_visible=True, 
                        template='plotly_white',
                        height=500,
                        hovermode='x unified',
                    )
                    
                    st.plotly_chart(fig_nvidia, use_container_width=True)

                except FileNotFoundError:
                    st.warning("⚠️ **File Not Found:** Skipping NVIDIA sample chart because 'nvidia_cleaned.csv' could not be loaded. Please ensure the file exists in the current directory.")
                except Exception as e:
                    st.error(f"Error loading NVIDIA sample data: {e}")


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
    # Plot profits
    maxprofitfig = plt.figure(figsize=(12,6))
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
    st.pyplot(maxprofitfig)
    
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

