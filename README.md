# Quant Portfolio Optimizer (Team Builder)

A Python-based utility for financial portfolio construction and risk management. This application utilizes historical market data to determine asset weightings that maximize the risk-adjusted return (Sharpe Ratio) of a given portfolio.

## Technical Specifications
* **Optimization Logic:** Implements the Sequential Least Squares Programming (SLSQP) algorithm via `scipy.optimize` to find the Maximum Sharpe Ratio.
* **Data Sourcing:** Integration with `yfinance` for five-year historical closing price retrieval.
* **Risk Assessment:** Calculates annualized volatility and expected returns based on the covariance matrix of daily returns.
* **Benchmarking:** Provides direct comparison against the S&P 500 (SPY) using normalized cumulative returns.
* **Allocation Engine:** Converts optimized weights into actionable capital requirements based on user-defined investment parameters.

## Tech Stack
* **Language:** Python 3.12
* **Numerical Libraries:** NumPy, Pandas, SciPy
* **Web Framework:** Streamlit
* **Visualization:** Matplotlib

## Implementation
1. Clone the repository:
   ```bash
   git clone [https://github.com/SanskarDeshkar/Portfolio-Team-Builder.git](https://github.com/SanskarDeshkar/Portfolio-Team-Builder.git)