# Quant Portfolio Optimizer (Team Builder)

A Python-based utility for financial portfolio construction and risk management. This application utilizes historical market data to determine asset weightings that maximize the risk-adjusted return (Sharpe Ratio) of a given portfolio while respecting user-defined volatility constraints.

## Technical Specifications
* **Optimization Logic**: Implements the Sequential Least Squares Programming (SLSQP) algorithm via scipy.optimize to find the Maximum Sharpe Ratio.
* **Risk Constraint Engine**: Employs a custom inequality constraint ($$Target \sigma - Portfolio \sigma \geq 0$$) to ensure the final portfolio does not exceed the user's selected volatility threshold.
* **Dual-Pass Optimization**: Executes a baseline 'Natural' optimization and a second 'Constrained' optimization to quantify capital migration.
* **Data Sourcing**: Integration with yfinance for five-year historical closing price retrieval.
* **Benchmarking**: Provides direct comparison against the S&P 500 (SPY) using normalized cumulative returns.

## Key Features
* **Interactive Risk Slider**: Allows users to toggle between Low, Medium, and High risk tolerances, dynamically injecting new constraints into the mathematical model.
* **Rebalancing Summary**: Generates a 'Trade List' that quantifies the dollar shift (Delta) required to move from a baseline portfolio to a risk-adjusted one.
* **Automated Interpretation**: Provides a real-time natural language explanation (via st.info) of the optimizer's actions based on the resulting capital shifts.

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

Where $\R_p$ represents the expected portfolio return and $\sigma_p$ represents the portfolio standard deviation.

### Risk Constraint
When a risk limit is active, the optimizer satisfies:
$$\sigma_{portfolio} \leq \sigma_{target}$$

## Portfolio Interpretation
The application provides a comparative analysis between the Unconstrained Max Sharpe Ratio (Baseline) and the Constrained Risk-Adjusted Portfolio.

* **Capital Allocation**: Values represent the exact USD amount to be invested in each ticker based on the provided principal.
* **Rebalancing Delta**: The 'Shift' column quantifies the capital migration required to satisfy user volatility constraints. A non-zero shift indicates the baseline portfolio exceeded the target risk threshold and was recalibrated for stability.