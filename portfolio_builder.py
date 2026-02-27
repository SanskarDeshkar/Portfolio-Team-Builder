import streamlit as st
import yfinance as yf
import pandas as pd

# setting the title and description of the app
st.title("Quant Portfolio Team Builder")
st.write("Draft the assets and optimize the risk.")

# input for the user to enter the ticker symbols
st.subheader("Enter the ticker symbols of the assets you want to include in your portfolio (separated by commas):")
tickers_input = st.text_input("Ticker Symbols", "AAPL, MSFT, JNJ, XOM, GLD")

# cleaning up the users text into a list of tickers
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

# button to trigger the optimization process
if st.button("Optimize the tickers"):
    st.write(f"Fetching data for: {', '.join(tickers)}")
    data = yf.download(tickers, period="5y")["Close"] # downloading the historical closing prices for the tickers
    data = data.dropna() # dropping any rows with missing values
    st.subheader("Asset Performance Data") # displaying the historical closing prices of the assets
    st.write(data)
    st.success("Data loaded. Ready to start optimization math.")

