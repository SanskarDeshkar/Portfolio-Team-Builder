# Quant Portfolio Optimizer

A Python-based utility for financial portfolio construction and risk management. This application utilizes historical market data to determine asset weightings that maximize the risk-adjusted return (Sharpe Ratio) of a given portfolio.

## Technical Specifications
* **Optimization Logic**: Implements the Sequential Least Squares Programming (SLSQP) algorithm via scipy.optimize to find the Maximum Sharpe Ratio.
* **Data Sourcing**: Integration with yfinance for five-year historical closing price retrieval.
* **Risk Assessment**: Calculates annualized volatility and expected returns based on the covariance matrix of daily returns.
* **Benchmarking**: Provides direct comparison against the S&P 500 (SPY) using normalized cumulative returns.
* **Allocation Engine**: Converts optimized weights into actionable capital requirements based on user-defined investment parameters.

## Tech Stack
* **Language**: Python 3.12
* **Numerical Libraries**: NumPy, Pandas, SciPy
* **Web Framework**: Streamlit
* **Visualization**: Matplotlib

## Implementation
1. Clone the repository:
```bash
git clone https://github.com/SanskarDeshkar/Portfolio-Team-Builder.git
```

2. Install required packages:
```bash
pip install streamlit yfinance pandas numpy scipy matplotlib
```

3. Execute the application:
```bash
streamlit run portfolio_builder.py
```

## Mathematical Framework
The objective function maximizes the Sharpe Ratio:

$$Sharpe Ratio = \frac{R_p - R_f}{\sigma_p}$$

Where R_p represents the expected portfolio return and \sigma_p represents the portfolio standard deviation.