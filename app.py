import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from datetime import datetime, timedelta
# import pandas_ta as ta
from calculations import *
from apihandler import *
from helper import *
import csv
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Display different dashboards based on selection For different Team Members---

#------------------------------------START OF THAW ZIN PART-------------------------------------------

# I am ThawZinHtun , a 1st year Applied Artificial Intelligence student at Singapore Institute of Technology
# I tried my best to develop and implement in a way that resembles a real world project
# I encouraged my team to use best practices such as modular code, error handling, comments, and documentation
# I have added comments to explain each part of the code for better understanding

# Session state to hold chat history
# Initialize chat history if not present , hardcoded initial message from FinSight 

if "chat_messages" not in st.session_state:
    initial_message = """
    Hello! I'm **FinSight** 🧠, your AI financial research and analysis assistant. I can help you understand stock data, technical indicators, and market trends.

    Ask me about **Moving averages**, **Support/resistance**, **Relative Strength Index**or anything else related to technical analysis!
    """
    # Initialize with a welcome message from the assistant
    st.session_state["chat_messages"] = [{"role": "assistant", "content": initial_message}]

# Set up the page configuration
st.set_page_config(
    page_title="Financial Trend Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Dashboard UI (Frontend) ---
st.title("Financial Trend Analysis Dashboard")

# --- Introduction Section ---
st.markdown("""
Welcome to the P1-3's live & historical financial data analysis dashboard! This tool allows you
to analyze stock tickers and get AI-powered insights.
""")

# --- Sidebar Menu for Navigation ---
st.sidebar.title("FinSight Dashboards")
st.sidebar.markdown("Navigate between different analysis and AI tools:")

# --- Dashboard Menu Selection ---
dashboard_selection = st.sidebar.radio(
    "Choose view:",
    (
        "📈 Trends Analysis",
        "📊 Daily Returns",
        "📉 SMA & EMA",
        "🖌️ RSI Visualization & Explanation",
        "📐 ATR",
        "💰 Max Profit Calculation",
    )
)

# --- Dashboard Content Change Based on Selection ---

if dashboard_selection == "🖌️ RSI Visualization & Explanation":
    st.markdown("---")
    st.subheader("Relative Strength Index (RSI) Visualization & Explanation")
    
    # RSI Visualization & Explanation UI 
    # why I put this in app.py instead of calculations.py
    # because this part is more of UI part instead of calculation part
    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    if ticker_symbol:
        try:
            # Time Range Selection
            TIME_RANGES = {
                "1M": timedelta(days=30), 
                "3M": timedelta(days=90), 
                "6M": timedelta(days=180),
                "1Y": timedelta(days=365),
                "3Y": timedelta(days=365 * 3), 
            }

            # Radio buttons for time range selection
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
            
            # Download data from yfinance
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {selected_range_label}..."):
                full_stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            # If wrong/invalid ticker symbol entered, return error msg
            if full_stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(full_stock_data.columns, pd.MultiIndex):
                    full_stock_data.columns = full_stock_data.columns.get_level_values(0)
            

                # Stock Data Handling
                data = {
                    'Date': full_stock_data.index,
                    'Open': full_stock_data['Open'],
                    'High': full_stock_data['High'],
                    'Low': full_stock_data['Low'],
                    'Close': full_stock_data['Close'],
                    'Volume': full_stock_data['Volume'],
                }

                # Convert to DataFrame . thanks to the team for this preprocess function
                df = pd.DataFrame(data)
                df = preprocess(df)


                # --- TAB DEFINITION ---
                tab1, tab2, tab3 = st.tabs(["📊 RSI Indicator", "🤖 FinSight AI Assistant", "📰 Stock News Feed"])
                
                # Just hardcoded explanation of RSI
                # I could use LLM to generate it 
                # I was able to do it using huggingface api
                # But I want to save the free tier api call for the chat part 
                # So I hardcoded it here
                with tab1:
                    fixed_rsi_period = 14  # Fixed RSI period

                    # Calculate RSI on the filtered data using the fixed 14-day period
                    # why it is common for 14 days
                    df['RSI'] = calculate_rsi(df, periods=fixed_rsi_period)

                    # Plot RSI 
                    # I did alot of commenting here to explain each line for better understanding
                    # When I first learned plotly it was very hard to understand
                    # During my previous internship, I had to learn plotly by myself
                    # I got grilled by my mentor when I couldn't explain details parameters 
                    fig_rsi = px.line(
                        df.dropna(),  # DataFrame to plot. dropna() removes any rows with NaN values to avoid plotting errors.
                        x=df.dropna().index,  # x-axis values: using the DataFrame's index (usually datetime or period numbers)
                        y='RSI',  # y-axis values: the 'RSI' column from the DataFrame
                        title=f"RSI Trend (Period {fixed_rsi_period} Days) over Last {selected_range_label}",  
                        # title of the chart. Includes the RSI calculation period and selected date range for context
                        labels={'RSI': 'RSI Value'}  
                        # labels parameter: maps DataFrame column names to more user-friendly axis labels
                    )

                    # Add a horizontal line at RSI=70, typically considered the "overbought" threshold
                    fig_rsi.add_hline(
                        y=70,  # y-value for the horizontal line
                        line_dash="dash",  # dashed line style
                        line_color="red",  # line color indicating warning/overbought condition
                        annotation_text="Overbought"  # text label displayed on the line
                    )

                    # Add a horizontal line at RSI=30, typically considered the "oversold" threshold
                    fig_rsi.add_hline(
                        y=30,  # y-value for the horizontal line
                        line_dash="dash",  # dashed line style
                        line_color="green",  # line color indicating buy signal/oversold condition
                        annotation_text="Oversold"  # text label displayed on the line
                    )

                    # Customize y-axis
                    fig_rsi.update_yaxes(
                        range=[0, 100],  # restrict y-axis from 0 to 100, standard for RSI
                        title="RSI Value"  # label for y-axis
                    )

                    # Customize x-axis
                    fig_rsi.update_xaxes(
                        title="Periods"  # label for x-axis, usually dates or periods depending on your index
                    )

                    # Render the Plotly figure in Streamlit
                    st.plotly_chart(
                        fig_rsi, 
                        use_container_width=True  # automatically adjusts chart width to the Streamlit container
                    )
                    st.subheader("What is the Relative Strength Index (RSI)?")
                    st.info(
                        """
                The **Relative Strength Index (RSI)** is a **momentum oscillator** in technical analysis that measures the 
                **speed and magnitude of price movements** of a financial instrument.

                **Key Points:**
                - **Range:** RSI values range from **0 to 100**.
                - **Overbought:** RSI above **70** may indicate an asset is overbought, potentially leading to a **price pullback**.
                - **Oversold:** RSI below **30** may indicate an asset is oversold, potentially leading to a **price rebound**.
                - **Typical Period:** RSI is commonly calculated over **14 periods** (e.g., 14 days for daily charts).
                """
                    )


                    st.info(
                        """
                **How RSI is Calculated (Simplified):**
                1. Calculate the **average gain** and **average loss** over the chosen period.
                2. Compute the **Relative Strength (RS)** = Average Gain ÷ Average Loss.
                3. RSI = 100 - (100 ÷ (1 + RS))  

                **Interpretation Tips:**
                - RSI **rising** → strengthening momentum.
                - RSI **falling** → weakening momentum.
                - Divergences between RSI and price can signal trend reversals:
                - Price makes a **new high**, but RSI does **not** → possible bearish reversal.
                - Price makes a **new low**, but RSI does **not** → possible bullish reversal.

                **Use Case:**  
                Traders often use RSI to **identify entry/exit points**, confirm trends, or combine with other indicators.

                Check the chart above for the RSI trend of the selected stock over the chosen time range.
                """
                    )

                # Tab 2: explanation
                with tab2:
                    st.header("FinSight 💬")

                    # 1. DISPLAY ALL MESSAGES FROM HISTORY
                    # This loop renders all messages from previous runs.
                    for message in st.session_state.chat_messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # 2. Handle new user input
                    if prompt := st.chat_input("Ask about any technical analysis...", key="ticker_input_chat"):
                        
                        # A. Append user message to history immediately
                        st.session_state.chat_messages.append({"role": "user", "content": prompt})
                        
                        # B. EXPLICITLY DISPLAY THE NEW USER MESSAGE in the current run
                        # This makes it appear instantly before the API call starts.
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        # 3. Get response from LLM
                        with st.chat_message("assistant"):
                            with st.spinner("Assistant is thinking..."):
                                try:
                                    # Prepare the full conversation context for the LLM (Universalized prompt)
                                    full_prompt = "You are an expert in financial and technical analyst. Answer the user's question concisely. User: " + prompt
                                    
                                    # Blocking API call happens here
                                    llm_response = call_huggingface_api(full_prompt)
                                    
                                    # Check Display and append the response
                                    # Why I hardcode here , cause whenever free tier huggingface api limit reached, it returns that hardcodeded message
                                    # That error message is not user friendly, so I replace it with my own message
                                    # Why it is not exception , because the api call itself is successful, just the response is not what we want
                                    if llm_response.startswith("Error occurred while generating the response."):
                                        llm_response = """The language Model Implemented is called from HuggingFace Inference API.
                                        The free tier has rate limits and usage limits. If you see this message, 
                                        it likely means the limit has been reached. Sorry for the inconvenience. 
                                        Please try again later. I even tried to do local LLM hosting but my laptop is not powerful enough :<"""
                                    st.markdown(llm_response)
                                    st.session_state.chat_messages.append({"role": "assistant", "content": llm_response})
                                    
                                except Exception as e:
                                    error_message = "Sorry, I can't reach the technical analysis server right now. Please try again later."
                                    st.error(error_message)
                                    # Append the error message to the chat history
                                    st.session_state.chat_messages.append({"role": "assistant", "content": error_message})
                            
                            # The st.rerun() is no longer strictly necessary if the user message is displayed immediately,
                            # but we keep it to ensure the latest state (especially after an error) is clean.
                            st.rerun() 

                # Tab 3: news
                with tab3:
                    st.header(f"Recent News for {ticker_symbol}")
                    with st.spinner(f"Fetching recent news for {ticker_symbol}..."):
                        stock_news = fetch_latest_news(ticker_symbol, limit=8)

                    if stock_news:
                        # Display news in a structured format
                        st.markdown("Click on a headline to open the full article.")
                        st.markdown("---")
                        # Use a two-column layout for index and source
                        for i, news_item in enumerate(stock_news):
                            col_index, col_source = st.columns([0.1, 0.9])
                            col_index.markdown(f"**{i+1}.**")
                            col_source.markdown(f"*{news_item.get('Source', 'Unknown')}*")
                            # Expandable section for the headline and link
                            # Tried my best to sort of news latest first using lambda function in apihandler.py
                            with st.expander(f"**{news_item.get('Title','No title')}**"):
                                st.markdown(f"Read full article: [**{news_item.get('URL','')}**]({news_item.get('URL','')})")
                            st.markdown("---")
                    else:
                        st.warning("No recent news articles found for this ticker.")

        except Exception as e:
            # Show the exception trace in the app for debugging
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data.")
            st.exception(e)


#------------------------------------END OF THAW ZIN PART---------------------------------------------
            
            
#------------------------------------START OF SIYONA PART------------------------------------

elif dashboard_selection == "📉 SMA & EMA":
    st.markdown("---") 
    st.header("Simple Moving Average (SMA) & Exponential Moving Average (EMA)")
    
    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()
    st.session_state["previous_ticker"] = ticker_symbol

   
    if ticker_symbol:
        try:
           

            # end date = today's date
            end_date = datetime.now()

            # start date = 3 years ago from today
            start_date = end_date - timedelta(days=3*365)

            
            range_label = "Last 3 years"
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {range_label}..."):
                stock_data = yf.download(ticker_symbol, start_date, end_date)
            
            if stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                # Stock Data Handling
                data = {
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)

                # Preprocess data using preprocess function in helper.py
                try:
                    df = preprocess(df)
                except Exception as e:
                    # Code to handle the error
                    st.error(f"Error with data for ticker: {ticker_symbol}. {e}. Please try another ticker.")

                # REST OF THE CODE GOES HERE ----------------------------------------------------------------------------------------------

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")

    # ---- Load dataset manually ----
    
# --- DATA RETRIEVAL ---
        full_stock_data = df
        close_prices2 = full_stock_data["Close"]

        # ---- Calculate SMAs using imported function ----
        sma_50 = simple_moving_average(close_prices2, 50)
        sma_100 = simple_moving_average(close_prices2, 100)
        sma_200 = simple_moving_average(close_prices2, 200)

        # ---- EMA periods ----
        periods = [12, 26, 50, 200]
        ema_dict = {f"EMA_{p}": exponential_moving_average(close_prices2, p) for p in periods}

        # --- Create separate figures ---
        fig_sma = go.Figure()
        fig_ema = go.Figure()

        # Add Closing Price as Candlestick to SMA chart
        fig_sma.add_trace(go.Candlestick(
            x=full_stock_data.index,
            open=full_stock_data["Open"],
            high=full_stock_data["High"],
            low=full_stock_data["Low"],
            close=full_stock_data["Close"],
            name="Close Price"
        ))


        # Add SMAs
        fig_sma.add_trace(go.Scatter(
            x=full_stock_data.index,
            y=sma_50,
            mode="lines",
            name="SMA 50",
            line=dict(color="blue", width=1.5)
        ))
        fig_sma.add_trace(go.Scatter(
            x=full_stock_data.index,
            y=sma_100,
            mode="lines",
            name="SMA 100",
            line=dict(color="orange", width=1.5)
        ))
        fig_sma.add_trace(go.Scatter(
            x=full_stock_data.index,
            y=sma_200,
            mode="lines",
            name="SMA 200",
            line=dict(color="red", width=1.5)
        ))

        # Add Closing Price as Candlestick to EMA chart
        fig_ema.add_trace(go.Candlestick(
            x=full_stock_data.index,
            open=full_stock_data["Open"],
            high=full_stock_data["High"],
            low=full_stock_data["Low"],
            close=full_stock_data["Close"],
            name="Close Price"
        ))


        # Add EMAs
        for period, ema_series in ema_dict.items():
            fig_ema.add_trace(go.Scatter(
                x=full_stock_data.index,

                y=ema_series,
                mode="lines",
                name=period,
                line=dict(width=1.5, dash="dot")  # dashed style for EMAs
            ))

        # Layout settings for SMA
        fig_sma.update_layout(
            title=f"{ticker_symbol} Stock Price with SMAs (50, 100, 200)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
            height=500
        )

        # Layout settings for EMA
        fig_ema.update_layout(
            title=f"{ticker_symbol} Stock Price with EMAs (12, 26, 50, 200)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
            height=500
        )

        # --- Streamlit render ---
        st.plotly_chart(fig_sma, use_container_width=True)
        st.success("Please drag the **above needle** to view at a granular level")
        st.info("The Simple Moving Average (SMA) is a widely used technical indicator that calculates the average of a stock’s prices over a specific number of periods. By smoothing out daily price fluctuations, the SMA helps traders and analysts identify the overall trend of a stock, making it easier to distinguish short-term noise from meaningful movements. Commonly used SMA periods include 50-day, which reflects short-term trends, 100-day for medium-term trends, and 200-day for long-term trends, often serving as key levels of support or resistance in market analysis")
        st.plotly_chart(fig_ema, use_container_width=True)
        st.success("Please drag the **above needle** to view at a granular level")
        st.info("The Exponential Moving Average (EMA), on the other hand, places greater weight on recent prices, allowing it to respond more quickly to changes in market direction. This responsiveness makes the EMA particularly useful for detecting trends and reversals sooner than the SMA. Typical EMA periods include 12-day and 26-day for short-term trends, which are often used in combination to generate trading signals, as well as 50-day and 200-day EMAs that help identify intermediate and long-term market trends. By choosing the appropriate EMA periods, traders can effectively balance sensitivity to recent price movements with the overall trend of the market.")

#------------------------------------END OF SIYONA PART------------------------------------

#------------------------------------START OF KAI REI PART----------------------------------

elif dashboard_selection == "📊 Daily Returns":
    st.markdown("---")
    st.header("Daily Returns")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()
    
    if ticker_symbol:
        try:
            # Daily Returns explaination
            st.markdown("---")
            st.write("### What is Daily Returns?")

            # Daily Returns formula to show
            daily_returns_formula = (r"r_t = \frac{p_t - p_{t-1}}{p_{t-1}} \times 100")
            st.latex(daily_returns_formula)

            st.write("**where:**")
            st.write("- $r_t$ = percentage daily return")
            st.write("- $p_t$ and $p_{t-1}$ = closing prices of day t and day t-1, respectively")

            # Left align formula
            st.markdown('''
            <style>
            .katex-html {
                text-align: left;
            }
            </style>
            ''', unsafe_allow_html=True)

            st.markdown("---")

            # Time Range Selection
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
            
            # Download data from yfinance
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {selected_range_label}..."):
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            # If wrong ticker symbol entered, return error msg
            if stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                # Stock Data Handling
                data = {
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)

                # Preprocess data using preprocess function in helper.py
                try:
                    df = preprocess(df)
                except Exception as e:
                    # Code to handle the error
                    st.error(f"Error with data for ticker: {ticker_symbol}. {e}. Please try another ticker.")

                # Calculate Daily Returns
                df['DailyReturns'] = calculate_daily_returns(df['Close'].tolist())
                
                st.subheader(f"Daily Returns for {ticker_symbol}")

                # Daily Returns Plot
                fig_daily_return = px.line(
                    x=df.index,
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


elif dashboard_selection == "📐 ATR":
    st.markdown("---") 
    st.header("Average True Range (ATR)")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

    if ticker_symbol:
        try:
            # ATR explaination
            st.markdown("---")
            st.write("### What is ATR?")

            atr_explanation = (
                "The average true range (ATR) is a technical analysis indicator that measures market volatility by "
                "decomposing the entire range of an asset price for that period. While the ATR doesn't tell us in which direction the breakout will occur, "
                "it can be added to the closing price, and the trader can buy whenever the next day's price trades above that value. "
                "Trading signals occur relatively infrequently but usually indicate significant breakout points. The logic behind these "
                "signals is that whenever a price closes more than an ATR above the most recent close, a change in volatility has occurred."
            )

            st.info(atr_explanation)

            # TR and ATR formulas to show
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("True Range (TR)")
                true_range_formula = (r"TR = \max[(H - L), |H - C_p|, |L - C_p|]")
                st.latex(true_range_formula)
                
                st.write("**where:**")
                st.write("- $H$ = Today's high")
                st.write("- $L$ = Today's low") 
                st.write("- $C_p$ = Yesterday's closing price")
                st.write("- $\max$ = Highest value of the three terms")
                
                st.write("**so that:**")
                st.write("- $(H - L)$ = Today's high minus low")
                st.write("- $|H - C_p|$ = |Today's high - yesterday's close|")
                st.write("- $|L - C_p|$ = |Today's low - yesterday's close|")

            with col2:
                st.subheader("Average True Range (ATR)")
                atr_formula = (r"ATR = \frac{1}{n}\sum_{i=1}^{n} TR_i")
                st.latex(atr_formula)
                
                st.write("**where:**")
                st.write("- $ATR$ = Average True Range")
                st.write("- $n$ = Number of periods")
                st.write("- $TR_i$ = True Range for period $i$")
                st.write("- $\sum$ = Summation of all TR values")
                
                st.write("**so that:**")
                st.write("ATR is the simple moving average")
                st.write("of True Range over $n$ periods")

            # Left align formula
            st.markdown('''
            <style>
            .katex-html {
                text-align: left;
            }
            </style>
            ''', unsafe_allow_html=True)

            st.markdown("---")

            # Time Range Selection
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
            
            # Download data from yfinance
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {selected_range_label}..."):
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            # If wrong ticker symbol entered, return error msg
            if stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                # Stock Data Handling
                data = {
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)

                # Preprocess data using preprocess function in helper.py
                try:
                    df = preprocess(df)
                except Exception as e:
                    # Code to handle the error
                    st.error(f"Error with data for ticker: {ticker_symbol}. {e}. Please try another ticker.")

                st.subheader(f"Average True Range (ATR) Analysis for {ticker_symbol}")
                
                # Create subplots: 2 rows, shared x-axis
                fig_combined = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3]
                )
                
                # Close Line Chart (0.5/2)
                close_line = go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close'
                )
                fig_combined.add_trace(close_line, row=1, col=1)

                # Calculate TR, ATR and Close+ATR values
                df['TR'] = calculate_true_range(df['High'].tolist(), df['Low'].tolist(), df['Close'].tolist())
                df['ATR'] = calculate_average_true_range(df['TR'].tolist())
                df['Close+ATR'] = [close + atr for close, atr in zip(df['Close'].tolist(), df['ATR'].tolist())]

                # Close+ATR Line Chart (1/2)
                close_atr_line = go.Scatter(
                    x=df.index,
                    y=df['Close+ATR'],
                    mode='lines',
                    name='Close+ATR'
                )
                fig_combined.add_trace(close_atr_line, row=1, col=1)
            

                # ATR Line Chart (2/2)
                atr_line = go.Scatter(
                    x=df.index,
                    y=df['ATR'],
                    mode='lines',
                    name='ATR'
                )
                fig_combined.add_trace(atr_line, row=2, col=1)

                # Update layout
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

        except Exception as e:
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")

#------------------------------------END OF KAI REI PART------------------------------------

#------------------------------------START OF WYNN PART-------------------------------------

elif dashboard_selection == "💰 Max Profit Calculation":
    st.markdown("---")
    st.header("Max Profit Calculation Dashboard")
    

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

                # Preprocess data using preprocess function in helper.py
                try:
                    df = preprocess(df)
                except Exception as e:
                    # Code to handle the error
                    st.error(f"Error with data for ticker: {ticker_symbol}. {e}. Please try another ticker.")

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

elif dashboard_selection == "📈 Trends Analysis":
    st.markdown("---")
    st.header("Trends Analysis Dashboard")

    ticker_symbol = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOG)", ).upper()

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
        validation_bands = data.ta.bbands(length=window, std=k)

        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.5, 0.5]
            )

        # Closing price line
        fig.add_trace(go.Candlestick(
            x=data.index, 
            open=data['Open'],
            close=data['Close'],
            high=data['High'],
            low=data['Low'],
            name="Closing Price"
        ), row=1, col=1)

        # SMA
        fig.add_trace(go.Scatter(
            x=data.index, y=bands['SMA'],
            name="SMA",
            line=dict(color="orange")
        ), row=1, col=1)

        # Upper Band
        fig.add_trace(go.Scatter(
            x=data.index, y=bands['UpperBand'],
            name="Upper Band",
            line=dict(color="green"),
            mode="lines"
        ), row=1, col=1)

        # Lower Band
        fig.add_trace(go.Scatter(
            x=data.index, y=bands['LowerBand'],
            name="Lower Band",
            line=dict(color="red"),
            mode="lines",
            fill='tonexty',         # fills to previous trace (Upper Band)
            fillcolor="rgba(128,128,128,0.3)"
        ), row=1, col=1)


        fig.add_trace(go.Scatter(
            x=data.index,
            y=validation_bands[f"BBU_{window}_2.0_2.0"],
            mode="lines",
            name="Upper Band",
            line=dict(color="green"),
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=validation_bands[f"BBL_{window}_2.0_2.0"],
            mode="lines",
            name="Lower Band",
            line=dict(color="red"),
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=validation_bands[f"BBM_{window}_2.0_2.0"],
            mode="lines",
            name="Middle Band (SMA)",
            line=dict(color="blue"),
            showlegend=False
        ), row=2, col=1)

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
        up_streak = 0
        down_streak = 0
        longest_up = {"length": 0, "end_date": None}
        longest_down = {"length": 0, "end_date": None}

        for i in range(1, len(data)):
            prev, current = close[i-1], close[i]
            if current > prev: # Uptrend
                color = "green"
                up_streak += 1
                down_streak = 0
            elif current < prev: # Downtrend
                color = "red"
                down_streak += 1
                up_streak = 0
            else: # Flat
                color = "grey"
                up_streak = down_streak = 0

            if up_streak > longest_up["length"]:
                longest_up.update({"length": up_streak, "end_date": data.index[i]})
            if down_streak > longest_down["length"]:
                longest_down.update({"length": down_streak, "end_date": data.index[i]})

            # Add each small segment as its own trace
            fig.add_trace(go.Scatter(
                x=data.index[i-1:i+1],
                y=data["Close"].iloc[i-1:i+1],
                mode="lines",
                line=dict(color=color, width=1.8),
                hoverinfo='skip',
                showlegend=False   # hide legend spam
            ))

        # Computing start dates for longest streak
        if longest_up["end_date"] is not None:
            end_idx = data.index.get_loc(longest_up["end_date"])
            longest_up["start_date"] = data.index[end_idx - longest_up['length']]

        if longest_down["end_date"] is not None:
            end_idx = data.index.get_loc(longest_down["end_date"])
            longest_down["start_date"] = data.index[end_idx - longest_down['length']]

        streaks = {
            "longest_uptrend": longest_up,
            "longest_downtrend": longest_down
        }
        
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


        shapes = []
        annotations = []

        if longest_up["end_date"]:
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=longest_up["start_date"], x1=longest_up["end_date"],
                y0=0, y1=1,
                fillcolor="rgba(0,255,0,0.15)",
                line=dict(width=0),
                layer="below"
            ))
            annotations.append(dict(
                x=longest_up["end_date"],
                y=max(close),
                text=f"Longest Uptrend ({longest_up['length']} days)",
                showarrow=False,
                font=dict(color="green", size=12)
            ))

        if longest_down["end_date"]:
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=longest_down["start_date"], x1=longest_down["end_date"],
                y0=0, y1=1,
                fillcolor="rgba(255,0,0,0.15)",
                line=dict(width=0),
                layer="below"
            ))
            annotations.append(dict(
                x=longest_down["end_date"],
                y=min(close),
                text=f"Longest Downtrend ({longest_down['length']} days)",
                showarrow=False,
                font=dict(color="red", size=12)
            ))

        fig.update_layout(
        title="Trend Line Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(orientation="v", x=1, xanchor="right"),
        shapes=shapes,
        annotations=annotations
        )

        st.info(
            f"This longest Uptrend is from {streaks['longest_uptrend']['start_date'].strftime('%b %d, %Y')} "
            f"to {streaks['longest_uptrend']['end_date'].strftime('%b %d, %Y')} "
            f"and lasted for {streaks['longest_uptrend']['length']} days"
        )

        st.info(
            f"This longest Downtrend is from {streaks['longest_downtrend']['start_date'].strftime('%b %d, %Y')} "
            f"to {streaks['longest_downtrend']['end_date'].strftime('%b %d, %Y')} "
            f"and lasted for {streaks['longest_downtrend']['length']} days"
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
            # Time Range Selection
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
            
            # Download data from yfinance
            with st.spinner(f"Fetching data for {ticker_symbol} over the last {selected_range_label}..."):
                stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
            
            # If wrong ticker symbol entered, return error msg
            if stock_data.empty:
                st.error(f"Could not find data for ticker: {ticker_symbol}. Please check the symbol and try again.")
            else:
                # Flatten columns if multi-level
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                # Stock Data Handling
                data = {
                    'Open': stock_data['Open'],
                    'High': stock_data['High'],
                    'Low': stock_data['Low'],
                    'Close': stock_data['Close'],
                    'Volume': stock_data['Volume'],
                }

                df = pd.DataFrame(data)

                # Preprocess data using preprocess function in helper.py
                try:
                    data = preprocess(df)
                except Exception as e:
                    # Code to handle the error
                    st.error(f"Error with data for ticker: {ticker_symbol}. {e}. Please try another ticker.")
                
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
            st.error(f"An error occurred: {e}. The ticker may be invalid or there was an issue fetching data. Please try again.")# 


            
        
#------------------------------------END OF YUAN WEI PART-----------------------------------

