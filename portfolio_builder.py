import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# streamlit run portfolio_builder.py

# setting the title and description of the app
st.title("Quant Portfolio Team Builder")
st.write("Draft the assets and optimize the risk.")

# input for the user to enter the ticker symbols
st.subheader("Enter the ticker symbols of the assets you want to include in your portfolio (separated by commas):")
tickers_input = st.text_input("Ticker Symbols", "AAPL, MSFT, JNJ, XOM, GLD")

# cleaning up the users text into a list of tickers
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

# input value for the user to enter how much they want to invest in the portfolio
investment_amount = st.number_input("Enter the total amount you want to invest in the portfolio:", min_value=100.0, value=1000.0, step=100.0)

# button to trigger the optimization process
if st.button("Optimize the tickers"):
    st.write(f"Fetching data for: {', '.join(tickers)}")

    data = yf.download(tickers, period="5y")["Close"] # downloading the historical closing prices for the tickers
    data = data.dropna() # dropping any rows with missing values

    st.subheader("Asset Performance Data") # displaying the historical closing prices of the assets
    
    # line chart of the asset performance
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1116")
    ax.set_facecolor("#0f1116")
    data.plot(ax=ax)
    ax.set_title("Historical Closing Prices", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    legend = ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(data.index.min(), data.index.max())
    plt.setp(legend.get_texts(), color="white")
    st.pyplot(fig)

    st.write(data)
    st.success("Data loaded. Ready to start optimization math.")

    # calculating the risk and return of the portfolio
    st.subheader("Portfolio Optimization")

    # calculating daily returns
    returns = data.pct_change().dropna()

    # calculating mean returns and covariance matrix (risk)
    mean_returns = returns.mean() * 252 # 252 trading days in a year
    cov_matrix = returns.cov() * 252

    def get_portfolio_performance(weights):
        weights = np.array(weights)
        p_return = np.sum(mean_returns * weights)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = p_return / p_volatility # higher is better
        return p_return, p_volatility, sharpe_ratio

    # finding the optimal weights using the Sharpe ratio as the objective function
    def min_func_sharpe(weights):
        return -get_portfolio_performance(weights)[2] # we want to maximize the Sharpe ratio, so we minimize its negative

    # constraints: weights must sum to 100% (1.0)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for x in range(len(tickers))) # weights must be between 0 and 1
    initial_guess = len(tickers) * [1. / len(tickers)] # start with equal weights
    optimized = minimize(min_func_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # displaying the optimized portfolio weights and the cash to invest in each asset
    st.write("### Optimized Portfolio Weights:")
    results_df = pd.DataFrame({
        'Asset': tickers, 
        'Weight (%)': (optimized.x * 100).round(2), # convert weights to percentages and round to 2 decimal places
        'Cash to Invest ($)': (optimized.x * investment_amount).round(2) # calculate the cash to invest in each asset
    }) 
    df_display = results_df.sort_values(by='Weight (%)', ascending=False).reset_index(drop=True) # sort by weight and reset index
    df_display["Weight (%)"] = df_display["Weight (%)"].map("{:.2f}%".format) # format weight as percentage string
    df_display["Cash to Invest ($)"] = df_display["Cash to Invest ($)"].map("${:,.2f}".format) # format cash to invest as currency string
    st.table(df_display) 
    st.success("Optimization complete.")
    
    # downloading S&P 500 data for comparison
    spy_data = yf.download("SPY", period="5y")["Close"].dropna()

    # calculating the cumulative returns for the assets and S&P 500 (starting both at $1.00)
    portfolio_cumulative = (1 + returns.dot(optimized.x)).cumprod() # cumulative returns of the optimized portfolio
    spy_cumulative = (1 + spy_data.pct_change().dropna()).cumprod() # cumulative returns of the S&P 500

    # plotting the cumulative returns of the optimized portfolio vs S&P 500
    st.subheader("Optimized Portfolio vs S&P 500")
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0f1116")
    ax2.set_facecolor("#0f1116")
    ax2.plot(portfolio_cumulative, label="Optimized Portfolio", color="#1f77b4", linewidth=2)
    ax2.plot(spy_cumulative, label="S&P 500 (SPY)", color="#ff7f0e", linewidth=2)
    ax2.set_title("Portfolio Performance vs S&P 500", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.set_ylabel("Cumulative Return", color="white")
    ax2.legend(facecolor="#0f1116", frameon=False, loc="upper left", labelcolor="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")
    st.pyplot(fig2)

    if portfolio_cumulative.iloc[-1].item() > spy_cumulative.iloc[-1].item():
        st.success("Your optimized portfolio outperformed the S&P 500 over the last 5 years.")
    else:
        st.warning("Your optimized portfolio underperformed the S&P 500 over the last 5 years.")

    # getting the final results for the optimized portfolio
    p_return, p_volatility, sharpe_ratio = get_portfolio_performance(optimized.x)

    # displaying the portfolio performance metrics
    st.write(f"### Portfolio Performance:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Annual Return", f"{p_return * 100:.2f}%")
    with col2:
        st.metric("Annual Volatility (Risk)", f"{p_volatility * 100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # brief explanation of the performance metrics
    st.info(f"""**Interpretation:** - This portfolio expects a return of **{p_return * 100:.2f}%** per year based on historical data.
        - The volatility of **{p_volatility * 100:.2f}%** represents the 'swing'—higher numbers mean a bumpier ride.
        - A Sharpe Ratio of **{sharpe_ratio:.2f}** means you are getting a solid amount of return for the risk you're taking.
        """)