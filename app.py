import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from calculations import *
from apihandler import *
from helper import *
import csv
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Display different dashboards based on selection For different Team Members---

#------------------------------------START OF THAW ZIN PART-------------------------------

# # --- SESSION STATE INITIALIZATION ---
# if 'stock_data_cache' not in st.session_state:
#     st.session_state['stock_data_cache'] = {} # Stores {ticker: 3Y_dataframe}
# if 'current_ticker' not in st.session_state:
#     st.session_state['current_ticker'] = ""


# if "chat_messages" not in st.session_state:
#     initial_message = """
#     Hello! I'm **FinSight** 🧠, your AI financial research and analysis assistant. I can help you understand stock data, technical indicators, and market trends.

#     Ask me about **Moving averages**, **Support/resistance**, **Relative Strength Index**or anything else related to technical analysis!
#     """
#     st.session_state["chat_messages"] = [{"role": "assistant", "content": initial_message}]

# # Set up the page configuration
# st.set_page_config(
#     page_title="Financial Trend Analysis Dashboard",
#     page_icon="📊",
#     layout="wide",
# )
 
# # --- Dashboard UI (Frontend) ---

# st.title("Financial Trend Analysis Dashboard")

# st.markdown("""
# Welcome to the P1-3's live & historical financial data analysis dashboard! This tool allows you
# to analyze stock tickers and get AI-powered insights.
# """)

# # --- Sidebar Menu for Navigation ---
# st.sidebar.title("Dashboard Menu")
# dashboard_selection = st.sidebar.radio(
#     "Select a dashboard view:",
#     ("Ticker Input" , "Trends Analysis" , "Daily Returns", "SMA & EMA",  "RSI Visualization & Explanation", "Max Profit Calculation")
# )


# if dashboard_selection == "Ticker Input":
#     st.markdown("---")
#     st.header("Please input Stock Ticker To view Various Analysis")

#     # Use session state to maintain the input value across reruns
#     ticker_symbol_input = st.text_input(
#         "Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", 
#         value=st.session_state.current_ticker
#     ).upper()
    
#     st.markdown("---")
#     # --- FIN SIGHT CHAT ASSISTANT UI ---
#     st.header("FinSight 💬")

#     # 1. DISPLAY ALL MESSAGES FROM HISTORY
#     # This loop renders all messages from previous runs.
#     for message in st.session_state.chat_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # 2. Handle new user input
#     if prompt := st.chat_input("Ask about any technical analysis...", key="ticker_input_chat"):
        
#         # --- IMPROVEMENT START ---
#         # A. Append user message to history immediately
#         st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
#         # B. EXPLICITLY DISPLAY THE NEW USER MESSAGE in the current run
#         # This makes it appear instantly before the API call starts.
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # --- IMPROVEMENT END ---
        
#         # 3. Get response from LLM
#         with st.chat_message("assistant"):
#             with st.spinner("Assistant is thinking..."):
#                 try:
#                     # Prepare the full conversation context for the LLM (Universalized prompt)
#                     full_prompt = "You are an expert technical analyst. Answer the user's question concisely. User: " + prompt
                    
#                     # Blocking API call happens here
#                     llm_response = call_huggingface_api(full_prompt)
                    
#                     # Display and append the response
#                     st.markdown(llm_response)
#                     st.session_state.chat_messages.append({"role": "assistant", "content": llm_response})
                    
#                 except Exception as e:
#                     error_message = "Sorry, I can't reach the technical analysis server right now. Please try again later."
#                     st.error(error_message)
#                     # Append the error message to the chat history
#                     st.session_state.chat_messages.append({"role": "assistant", "content": error_message})
            
#             # The st.rerun() is no longer strictly necessary if the user message is displayed immediately,
#             # but we keep it to ensure the latest state (especially after an error) is clean.
#             st.rerun() 

#     # --- Ticker Input and Data Fetching Logic (Original Code) ---
    
#     # Only proceed if the input has changed or is not empty
#     if ticker_symbol_input and ticker_symbol_input != st.session_state.current_ticker:
#         st.session_state.current_ticker = ticker_symbol_input # Update current ticker

#         # --- DATA FETCHING LOGIC (Max 3Y range) ---
#         MAX_TIME_RANGE = timedelta(days=365 * 3) # 3 Years
#         end_date = datetime.now()
#         start_date = end_date - MAX_TIME_RANGE
#         ticker = st.session_state.current_ticker
        
#         try:
#             with st.spinner(f"Fetching **3 years** of historical data for {ticker}..."):
#                 # Assume yf (yfinance) and datetime/timedelta are imported
#                 # Fetch data up to the maximum required range (3Y)
#                 stock_data_3y = yf.download(ticker, start=start_date, end=end_date, progress=False)

#             if stock_data_3y.empty:
#                 st.error(f"Could not find data for ticker: {ticker}. Please check the symbol and try again.")
#                 st.session_state.current_ticker = "" # Reset ticker on error
#                 st.session_state.stock_data_cache.pop(ticker, None) # Remove from cache
#             else:
#                 # Flatten columns if multi-level
#                 if isinstance(stock_data_3y.columns, pd.MultiIndex):
#                     # Assume pd (pandas) is imported
#                     stock_data_3y.columns = stock_data_3y.columns.get_level_values(0)
                
#                 # Store the full 3Y data in the cache under the ticker key
#                 st.session_state.stock_data_cache[ticker] = stock_data_3y
#                 st.success(f"Data for **{ticker}** (3Y range) fetched and cached successfully!")
                
#         except Exception as e:
#             st.error(f"An error occurred during data fetching for {ticker}: {e}")
#             st.session_state.current_ticker = "" # Reset ticker on error
#             st.session_state.stock_data_cache.pop(ticker, None) # Remove from cache
#         st.session_state.current_ticker = ticker_symbol_input # Update current ticker

#         # --- DATA FETCHING LOGIC (Max 3Y range) ---
#         MAX_TIME_RANGE = timedelta(days=365 * 3) # 3 Years
#         end_date = datetime.now()
#         start_date = end_date - MAX_TIME_RANGE
#         ticker = st.session_state.current_ticker
        
#         try:
#             with st.spinner(f"Fetching **3 years** of historical data for {ticker}..."):
#                 # Assume yf (yfinance) and datetime/timedelta are imported
#                 # Fetch data up to the maximum required range (3Y)
#                 stock_data_3y = yf.download(ticker, start=start_date, end=end_date, progress=False)

#             if stock_data_3y.empty:
#                 st.error(f"Could not find data for ticker: {ticker}. Please check the symbol and try again.")
#                 st.session_state.current_ticker = "" # Reset ticker on error
#                 st.session_state.stock_data_cache.pop(ticker, None) # Remove from cache
#             else:
#                 # Flatten columns if multi-level
#                 if isinstance(stock_data_3y.columns, pd.MultiIndex):
#                     # Assume pd (pandas) is imported
#                     stock_data_3y.columns = stock_data_3y.columns.get_level_values(0)
                
#                 # Store the full 3Y data in the cache under the ticker key
#                 st.session_state.stock_data_cache[ticker] = stock_data_3y
#                 st.success(f"Data for **{ticker}** (3Y range) fetched and cached successfully!")
                
#         except Exception as e:
#             st.error(f"An error occurred during data fetching for {ticker}: {e}")
#             st.session_state.current_ticker = "" # Reset ticker on error
#             st.session_state.stock_data_cache.pop(ticker, None) # Remove from cache
# # --- Stock Price and RSI Dashboard Block ---

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
    ("Trends Analysis", "Daily Returns", "SMA & EMA", "RSI Visualization & Explanation", "ATR", "Max Profit Calculation")
)


if dashboard_selection == "RSI Visualization & Explanation":
    st.markdown("---")
    
    # ticker_symbol = st.session_state.current_ticker 
    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    # if not ticker_symbol:
    #     st.warning("Please enter a stock ticker in the **Ticker Input** section first.")
    # elif ticker_symbol not in st.session_state.stock_data_cache:
    #     st.warning(f"No 3-year data found for **{ticker_symbol}**. Please re-enter the ticker in the **Ticker Input** section.")
    if ticker_symbol:
        try:
            # # --- DATA RETRIEVAL ---
            # full_stock_data = st.session_state.stock_data_cache[ticker_symbol]

            TIME_RANGES = {
                "1M": timedelta(days=30), 
                "3M": timedelta(days=90), 
                "6M": timedelta(days=180),
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), 
            }

            selected_range_label = st.radio(
                "Select Time Range:",
                options=list(TIME_RANGES.keys()),
                index=4, # Default to 1Y
                horizontal=True
            )
            # Fetch data based on user selected range
            end_date = datetime.now()
            
            # Calculate start date based on selected label
            range_delta = TIME_RANGES.get(selected_range_label, timedelta(days=365))
            start_date = end_date - range_delta
            
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {selected_range_label}..."):
                full_stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            if full_stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(full_stock_data.columns, pd.MultiIndex):
                    full_stock_data.columns = full_stock_data.columns.get_level_values(0)
            
            # # 1. Initialize session state variable for the widget's value (using the key name)
            # # This setup is correct and provides the default/initial value
            # if 'current_range_key' not in st.session_state:
            #     st.session_state.current_range_key = "1Y" 
                
            # # Ensure the stored value is valid
            # if st.session_state.current_range_key not in TIME_RANGES:
            #      st.session_state.current_range_key = "1Y"

                # Stock Data Handling
                data = {
                    'Date': full_stock_data.index,
                    'Open': full_stock_data['Open'],
                    'High': full_stock_data['High'],
                    'Low': full_stock_data['Low'],
                    'Close': full_stock_data['Close'],
                    'Volume': full_stock_data['Volume'],
                }

                df = pd.DataFrame(data)
                df = preprocess(df)


                # --- TAB DEFINITION ---
                tab1, tab2, tab3 = st.tabs(["📊 RSI Indicator", "💡 Indicator Explanation", "📰 Stock News Feed"])
                
                # --- WIDGET PLACEMENT (INSIDE TAB 1 ONLY) ---
                with tab1:
                    # 2. Place the st.radio widget inside Tab 1
                    st.radio( # <-- FIX: Removed the redundant 'index' argument
                        "Select Time Range for Data Analysis:",
                        options=list(TIME_RANGES.keys()),
                        horizontal=True,
                        key='current_range_key' # Streamlit will use st.session_state.current_range_key as the initial value
                    )
                    
                # --- DATA FILTERING LOGIC (Using the value captured from session state) ---
                current_range_label = st.session_state.current_range_key 
                range_delta = TIME_RANGES.get(current_range_label, timedelta(days=365))
                start_date_limit = datetime.now() - range_delta
                
                # Filter the full data for the selected range 
                stock_data = full_stock_data[full_stock_data.index >= start_date_limit.strftime('%Y-%m-%d')]

                if stock_data.empty:
                    st.error(f"No data available for the selected range: {current_range_label}. Try selecting a larger range.")
                else:

                    # --- FIXED RSI PERIOD LOGIC ---
                    fixed_rsi_period = 14
                    
                    # --- REMAINDER OF TAB 1 CONTENT (Continuation) ---
                    with tab1:
                        # --- RSI Calculation and Plot ---
                        st.subheader(f"Relative Strength Index (RSI) for {ticker_symbol}")
                        
                        # Calculate RSI on the *filtered* data using the fixed 14-day period
                        stock_data['RSI'] = calculate_rsi(stock_data, periods=fixed_rsi_period) 
                        
                        fig_rsi = px.line(
                            stock_data.dropna(), 
                            x=stock_data.dropna().index,
                            y='RSI',
                            title=f"RSI Trend (Period {fixed_rsi_period} Days) over Last {current_range_label}", 
                            labels={'RSI': 'Value'},
                        )
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_yaxes(range=[0, 100], title="RSI Value")
                        st.plotly_chart(fig_rsi, use_container_width=True)


                    with tab2:
                        st.header("What is the Relative Strength Index (RSI)?")
                        with st.spinner("Generating RSI explanation from Language Model..."):
                            try:
                                rsi_explanation = call_huggingface_api(
                                    "What is the Relative Strength Index (RSI)? Explain it simply for a beginner."
                                )
                                st.info(rsi_explanation)
                            except Exception as e:
                                st.info(
                                    """The **Relative Strength Index (RSI)** is a popular momentum indicator used in trading.         
                                    \nThink of it like a *speedometer for price movements*. It shows whether a stock or asset has been going up or down **too quickly**.  
                                    \n How it works:
                                    - RSI gives a value between **0 and 100**.  
                                    - **Above 70 → Overbought:** Price may have climbed too fast, and a pullback is possible.  
                                    - **Below 30 → Oversold:** Price may have dropped too fast, and a rebound is possible.  
                                    \n Key Notes:
                                    - The most common calculation uses a **14-day lookback period**.  
                                    - RSI doesn’t predict the future, but it helps traders spot when prices might be overheating or undervalued.  
                                    """
                                )

                    
                    with tab3:
                        # ... (News feed content) ...
                        st.header(f"Recent News for {ticker_symbol}")
                        with st.spinner(f"Fetching recent news for {ticker_symbol}..."):
                            stock_news = fetch_latest_news(ticker_symbol, limit=8)
                        if stock_news:
                            st.markdown("Click on a headline to open the full article.")
                            st.markdown("---")
                            for i, news_item in enumerate(stock_news):
                                col_index, col_source = st.columns([0.1, 0.9])
                                col_index.markdown(f"**{i+1}.**")
                                col_source.markdown(f"*{news_item['Source']}*")
                                with st.expander(f"**{news_item['Title']}**"):
                                    st.markdown(f"Read full article: [**{news_item['URL']}**]({news_item['URL']})")
                                st.markdown("---")
                        else:
                            st.warning("No recent news articles found for this ticker. Try a larger company or check the symbol.")
        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")
            # --- END: TABBED LAYOUT ---
#------------------------------------END OF THAW ZIN PART------------------------------------
            
            
#------------------------------------START OF SIYONA PART------------------------------------

elif dashboard_selection == "SMA & EMA":
    st.markdown("---") 
    st.header("Simple Moving Average (SMA) & Exponential Moving Average (EMA)")
    st.subheader("NVIDIA Stock Price with SMA (Periods: 50, 100, 200)")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    if ticker_symbol:
        try:
            # --- NEW TIME RANGE SELECTION LOGIC ---
            TIME_RANGES = {
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30),      # Approximation
                "3M": timedelta(days=90),      # Approximation
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), # Approximation
            }

            selected_range_label = st.radio(
                "Select Time Range:",
                options=list(TIME_RANGES.keys()),
                index=3, # Default to 1Y
                horizontal=True
            )

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

                # Stock Data Handling
                data = {
                    'Date': stock_data.index,
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)
                df = preprocess(df)

                # REST OF THE CODE GOES HERE ----------------------------------------------------------------------------------------------

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")

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
    st.header("Live Stock Ticker Analysis")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()
    
    if ticker_symbol:
        try:
            # --- NEW TIME RANGE SELECTION LOGIC ---
            TIME_RANGES = {
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30),      # Approximation
                "3M": timedelta(days=90),      # Approximation
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), # Approximation
            }

            selected_range_label = st.radio(
                "Select Time Range:",
                options=list(TIME_RANGES.keys()),
                index=3, # Default to 1Y
                horizontal=True
            )

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

                # Stock Data Handling
                data = {
                    'Date': stock_data.index,
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)
                df = preprocess(df)

                # --- Close and Close+Atr and ATR Section (Current Ticker) ---
                st.subheader(f"Average True Range (ATR) Analysis for {ticker_symbol}")
                
                # Create subplots: 2 rows, shared X-axis, Price (row 1) is taller than ATR (row 2)
                fig_combined = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3]
                )
                
                # 1. Close and Close+ATR Line Chart (Row 1)
                close_line = go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close'
                )
                fig_combined.add_trace(close_line, row=1, col=1)

                df['TR'] = calculate_true_range(df['High'], df['Low'], df['Close'])
                df['ATR'] = calculate_average_true_range(df['TR'])
                df['Close+ATR'] = [close + atr for close, atr in zip(df['Close'], df['ATR'])]

                close_atr_line = go.Scatter(
                    x=df['Date'],
                    y=df['Close+ATR'],
                    mode='lines',
                    name='Close+ATR'
                )
                fig_combined.add_trace(close_atr_line, row=1, col=1)
            

                # 2. ATR Line Chart (Row 2)
                atr_line = go.Scatter(
                    x=df['Date'],
                    y=df['ATR'],
                    mode='lines',
                    name='ATR'
                )
                fig_combined.add_trace(atr_line, row=2, col=1)

                # Update layout for a cleaner financial look
                fig_combined.update_layout(
                    title_text=f"Close Price with ATR Analysis for {ticker_symbol}",
                    xaxis_rangeslider_visible=False, # Hide the main range slider
                    xaxis2_title="Date",
                    yaxis_title="Price ($)",
                    yaxis2_title="ATR ($)",
                    height=700,
                    template='plotly_white'
                )
                
                # Finalize axis visibility and ranges
                fig_combined.update_xaxes(showgrid=True, row=1, col=1)
                fig_combined.update_yaxes(showgrid=True, row=1, col=1)
                fig_combined.update_yaxes(showgrid=True, row=2, col=1)
                
                st.plotly_chart(fig_combined, use_container_width=True)


                st.write("### What is ATR?")

                atr_explanation = (
                    "The average true range (ATR) is a technical analysis indicator that measures market volatility by "
                    "decomposing the entire range of an asset price for that period. The true range indicator is taken as "
                    "the greatest of the following: current high less the current low; the absolute value of the current high "
                    "less the previous close; and the absolute value of the current low less the previous close. The ATR is then "
                    "a moving average of the true ranges. While the ATR doesn't tell us in which direction the breakout will occur, "
                    "it can be added to the closing price, and the trader can buy whenever the next day's price trades above that value. "
                    "Trading signals occur relatively infrequently but usually indicate significant breakout points. The logic behind these "
                    "signals is that whenever a price closes more than an ATR above the most recent close, a change in volatility has occurred."
                )

                st.info(atr_explanation)
                st.markdown("---")


                # --- Daily Returns Calculation and Plot ---
                df['DailyReturns'] = calculate_daily_returns(df['Close'])
                
                st.subheader(f"Daily Returns for {ticker_symbol}")

                # Daily Returns Plot
                fig_daily_return = px.line(
                    x=df['Date'],
                    y=df['DailyReturns'],
                    title=f"Daily Returns for {ticker_symbol}",
                )

                # Rename X and Y Axis
                fig_daily_return.update_layout(
                    xaxis_title = "Date",
                    yaxis_title = "Daily Returns (%)"
                )

                st.plotly_chart(fig_daily_return, use_container_width=True)

            

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")


elif dashboard_selection == "ATR":
    st.markdown("---") 
    st.header("Average True Range (ATR)")
    st.subheader("ATR")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    if ticker_symbol:
        try:
            # --- NEW TIME RANGE SELECTION LOGIC ---
            TIME_RANGES = {
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30),      # Approximation
                "3M": timedelta(days=90),      # Approximation
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), # Approximation
            }

            selected_range_label = st.radio(
                "Select Time Range:",
                options=list(TIME_RANGES.keys()),
                index=3, # Default to 1Y
                horizontal=True
            )

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

                # Stock Data Handling
                data = {
                    'Date': stock_data.index,
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)
                df = preprocess(df)

                # REST OF THE CODE GOES HERE ----------------------------------------------------------------------------------------------

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")

#------------------------------------END OF KAI REI PART------------------------------------

#------------------------------------START OF WYNN PART-------------------------------------

elif dashboard_selection == "Max Profit Calculation":
    st.markdown("---")
    st.header("Max Profit Calculation Dashboard")
    
    # #Stock Ticker Selection
    # ticker = st.text_input("Enter Stock Ticker Symbol").upper()

    # if not ticker:
    #     st.warning("Please enter a valid stock ticker symbol")
    #     st.stop()

    # #Stock Time Period Selection
    # period_options = {
    #     "1 Week": 7,
    #     "1 Month": 30,
    #     "3 Months": 90,
    #     "6 Months": 180,
    #     "1 Year": 365,
    #     "2 Years": 365*2,
    #     "3 Years": 365*3
    # }

    # selected_period = st.selectbox(
    #     "Select Time Period",
    #     list(period_options.keys()),
    #     index=4  # Default is 1 Year
    # )

    # # Calculate Date Range
    # if period_options[selected_period]:
    #     end_date = datetime.now()
    #     start_date = end_date - timedelta(days=period_options[selected_period])

    # # Download Stock Data
    # with st.spinner(f"Fetching {ticker} data..."):
    #     try:
    #         df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
    #         if df.empty:
    #             st.error(f"No data available for {ticker} in selected period")
    #             st.stop()
                
    #         # Reset Index To Keep "Date" As A Column
    #         df.reset_index(inplace=True)
            
    #     except Exception as e:
    #         st.error(f"Error downloading data: {str(e)}")
    #         st.stop()

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    if ticker_symbol:
        try:
            # --- NEW TIME RANGE SELECTION LOGIC ---
            TIME_RANGES = {
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30),      # Approximation
                "3M": timedelta(days=90),      # Approximation
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), # Approximation
            }

            selected_range_label = st.radio(
                "Select Time Range:",
                options=list(TIME_RANGES.keys()),
                index=3, # Default to 1Y
                horizontal=True
            )

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

                # Stock Data Handling
                data = {
                    'Date': stock_data.index,
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)
                df = preprocess(df)

                # REST OF THE CODE GOES HERE ----------------------------------------------------------------------------------------------

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")

        # Sorting Transactions
        prices = df["Close"].values
        dates = pd.to_datetime(df["Date"]).values  

        total_profit, transactions = max_profit_with_days(prices, dates)
        rounded_total_profit = round(float(total_profit),2)


        transactions_df = pd.DataFrame(transactions)

        transactions_df["Buy Date"] = pd.to_datetime(transactions_df["Buy Date"])
        transactions_df["Profit"] = transactions_df["Profit"].astype(float)


        # Find Top 5 Profit Transactions
        top5_df = transactions_df.sort_values(by="Profit", ascending=False).head(5)
        top5_df["Buy Date"] = pd.to_datetime(top5_df["Buy Date"])
        top5_df["Profit"] = top5_df["Profit"].astype(float)
        
        #Plot Total Profit
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=transactions_df["Buy Date"],
            y=transactions_df["Profit"],
            mode="lines+markers",
            line=dict(color="lightpink"),
            name="Total Profits"
        ))

        # Highlight Top 5 Profits
        fig.add_trace(go.Scatter(
            x=top5_df["Buy Date"],
            y=top5_df["Profit"],
            mode="markers+text",
            marker=dict(color="lightblue", size=10),
            text=[f"{float(p):.2f}" for p in top5_df["Profit"].values],
            textposition="top center",
            name="Top 5 Profits"
        ))

        #Layout 
        fig.update_layout(
            title="Profit per Transaction",
            xaxis_title="Buy Date",
            yaxis_title="Profit",
            template="plotly_white",
            xaxis=dict(tickangle=45, showgrid=True, type='date'),
            yaxis=dict(showgrid=True),
            height=600,
            width=1000
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info('The maximum profit calculation is used to determine how much profit could be earned by buying and selling a stock whenever its price increases from one day to the next. It goes through each day in the stock’s price history and, if the price increased compared to the previous day, it assumes a “buy yesterday, sell today” transaction. Each profitable transaction is recorded with its buy and sell dates, prices, and the profit made. The function then sums up all the daily gains to give the total maximum profit.')

#------------------------------------END OF WYNN PART---------------------------------------

#------------------------------------START OF YUAN WEI PART---------------------------------

elif dashboard_selection == "Trends Analysis":
    st.markdown("---")
    st.header("Trends Analysis Dashboard")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    if ticker_symbol:
        try:
            # --- NEW TIME RANGE SELECTION LOGIC ---
            TIME_RANGES = {
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30),      # Approximation
                "3M": timedelta(days=90),      # Approximation
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), # Approximation
            }

            selected_range_label = st.radio(
                "Select Time Range:",
                options=list(TIME_RANGES.keys()),
                index=3, # Default to 1Y
                horizontal=True
            )

    def plot_bollinger_bands(data: pd.DataFrame, fig: go.Figure) -> go.Figure:
        '''
        Plots Bollinger Bands on plotly chart after
        running bollinger_bands from calculations.py

        Parameters:
        data: Stock's Dataframe consisting on Close values
        fig: Plotly.graph_objects Figure object

        Returns: Plotly.graph_objects Figure Object
        '''
        window = st.slider("Window", 1, 50, 5)
        k = st.slider("Standard Deviation", 1, 10, 2)


        bands = bollinger_bands(data=data, window=window, k=k)

        # Closing price line
        fig.add_trace(go.Candlestick(
            x=data.index, 
            open=data['Open'],
            close=data['Close'],
            high=data['High'],
            low=data['Low'],
            name="Closing Price"
        ))

        # SMA
        fig.add_trace(go.Scatter(
            x=data.index, y=bands['SMA'],
            name="SMA",
            line=dict(color="orange")
        ))

        # Upper Band
        fig.add_trace(go.Scatter(
            x=data.index, y=bands['UpperBand'],
            name="Upper Band",
            line=dict(color="green"),
            mode="lines"
        ))

        # Lower Band
        fig.add_trace(go.Scatter(
            x=data.index, y=bands['LowerBand'],
            name="Lower Band",
            line=dict(color="red"),
            mode="lines",
            fill='tonexty',         # fills to previous trace (Upper Band)
            fillcolor="rgba(128,128,128,0.3)"
        ))

        fig.update_layout(
            title="Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price($)",
            hovermode="x unified",
            xaxis_rangeslider_visible=False
        )
        return fig
    
    def plot_trends(data: pd.DataFrame, fig: go.Figure) -> go.Figure:
        '''
        Plots Upward trend and Downward trend
        based on previous and current Close value.

        Parameters:
        data: Stock's Dataframe consisting on Close values
        fig: Plotly.graph_objects Figure object

        Returns:
        fig: Plotly.graph_objects Figure object
        '''
        close = data["Close"].values

        for i in range(1, len(data)):
            prev, current = close[i-1], close[i]
            if current > prev:
                color = "green"
            elif current < prev:
                color = "red"
            else:
                color = "grey"

            # Add each small segment as its own trace
            fig.add_trace(go.Scatter(
                x=data.index[i-1:i+1],
                y=data["Close"].iloc[i-1:i+1],
                mode="lines",
                line=dict(color=color, width=1.8),
                hoverinfo='skip',
                showlegend=False   # hide legend spam
            ))
        
        # Dummy data to show legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="green", width=2),
            name="Uptrend"
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="red", width=2),
            name="Downtrend"
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="grey", width=2),
            name="Flat"
        ))

        fig.update_layout(
            title="Trend Line Chart",
            xaxis_title="Date",
            yaxis_title="Price($)",
            hovermode="x unified",
            legend=dict(
                orientation="v"
            )
        )
        return fig
    

    def plot_candles(data, fig):
        # Create subplots: 2 rows, shared X-axis, Price (row 1) is taller than Volume (row 2)
        
        new_fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
            )
        
        for trace in fig.data:
            new_fig.add_trace(trace, row=1, col=1)
        
        # 1. Candlestick Chart (Row 1)
        new_fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            showlegend=True
        ), row=1, col=1)
        
        # 2. Volume Bar Chart (Row 2)
        # Determine bar color based on daily movement (Close > Open = Green; else Red)
        volume_colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red'
                            for i in range(len(data))]

        new_fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=volume_colors,
            name='Volume'
            ), row=2, col=1)

        # Update layout for a cleaner financial look
        new_fig.update_layout(
            title_text=f"Historical Price and Volume Analysis for {ticker_symbol}",
            xaxis_rangeslider_visible=False, # Hide the main range slider
            xaxis2_title="Date",
            yaxis_title="Price ($)",
            yaxis2_title="Volume",
            height=700,
            template='plotly_white'
            )
                
        # Finalize axis visibility and ranges
        new_fig.update_xaxes(showgrid=True, row=1, col=1)
        new_fig.update_yaxes(showgrid=True, row=1, col=1)
        new_fig.update_yaxes(showgrid=True, row=2, col=1)

        return new_fig 
    

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

                # Stock Data Handling
                data = {
                    'Date': stock_data.index,
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                data = preprocess(pd.DataFrame(data))
                
                indicators_plot = {
                "Bollinger Bands": {"function": plot_bollinger_bands,
                                    "info": "are a technical analysis tool that consists of three lines: "
                                    " a middle line which is a simple moving average (SMA), "
                                    "and two outer bands set a standard deviation above and below the SMA. "
                                    "These bands dynamically adjust to market volatility, with wider bands "
                                    "indicating high volatility and narrower bands signaling low volatility. "
                                    "Traders use Bollinger Bands to identify if a security is overbought or oversold, "
                                    "gauge price volatility, and potentially identify support and resistance levels or breakout signals"},

                "Price Trends": {"function": plot_trends,
                           "info": "is the general direction that an asset's price moves over a specific period. "
                           "There are three main types of price trends: an uptrend (also called a bull market), "
                           "where prices are rising; a downtrend (or bear market), where prices are falling; and a sideways trend, "
                           "where prices fluctuate within a narrow, range-bound market without a clear upward or downward direction"},

                "Candles": {"function": plot_candles,
                            "info": "is a visual tool showing an asset's price movements for a specific time period, "
                            "displaying the open, close, high, and low prices, with the \"body\" representing the open-to-close range "
                            "and \"wicks\" (or shadows) indicating the high and low prices"}
            }

            selected_options = st.multiselect(
                "Select indicators to display:",
                list(indicators_plot.keys())
            )

            if "Candles" not in selected_options and "Price Trends" not in selected_options:
                fig = px.line(data, x=data.index, y="Close", title=f"Closing Price of {ticker_symbol}", height=600)
                fig.update_xaxes(
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikecolor="grey",
                    spikethickness=1
                )
                fig.update_yaxes(
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikecolor="grey",
                    spikethickness=1
                )

            else:
                fig = go.Figure()

            valid_selection = []
            for option in selected_options:
                if option in indicators_plot:
                    fig = indicators_plot[option]["function"](data, fig)
                    valid_selection.append(option)

            st.plotly_chart(fig, use_container_width=True)

            for option in valid_selection:
                st.info(f"**{option}**: {indicators_plot[option]['info']}")

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")
        
#------------------------------------END OF YUAN WEI PART-----------------------------------

