# Quant Portfolio Optimizer (Team Builder)

An AI-powered Streamlit app for portfolio construction, risk-constrained optimization, and rebalancing analysis. The app pulls historical market data, computes annualized return and volatility statistics, and uses SLSQP optimization to build a maximum-Sharpe portfolio that can also respect a user-selected volatility ceiling.

## Current State

This project now includes:

- A runnable Streamlit app in `portfolio_builder.py`
- A built-in chat interface using Streamlit's chat components
- Python project metadata in `pyproject.toml`
- Dependency install support in `requirements.txt`
- A basic repo setup for local virtual environment workflows

This repository was also updated through a Codex editing pass. That pass injected a cleanup and hardening layer into the project, especially inside `portfolio_builder.py`, without changing the app's core purpose.

## What The App Does

- Accepts a custom basket of ticker symbols
- Downloads approximately five years of adjusted price history with `yfinance`
- Builds an unconstrained maximum-Sharpe baseline portfolio
- Builds a risk-adjusted portfolio using the selected volatility target
- Computes dollar allocations from the user's investment amount
- Estimates trading costs for the rebalance
- Compares portfolio performance against `SPY`
- Breaks down each asset's contribution to total portfolio risk
- Lets users chat with a portfolio assistant directly on the page

## Tech Stack

- **Language**: Python 3.11+
- **App Framework**: Streamlit
- **Data Source**: yfinance
- **Numerical Libraries**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib
- **Chat**: Streamlit chat UI with optional OpenAI integration

## Setup

1. Clone the repository:

```bash
git clone https://github.com/SanskarDeshkar/Portfolio-Team-Builder.git
cd Portfolio-Team-Builder
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run portfolio_builder.py
```

5. Optional: enable OpenAI-backed chatbot responses:

```bash
export OPENAI_API_KEY=your_api_key_here
```

If no API key is configured, the app still exposes the chatbox and falls back to a built-in portfolio assistant.

## Optimization Model

The app maximizes the Sharpe Ratio:

$$
Sharpe Ratio = \frac{R_p}{\sigma_p}
$$

Where:

- $R_p$ is the annualized expected portfolio return
- $\sigma_p$ is the annualized portfolio volatility

The constrained portfolio additionally enforces:

$$
\sigma_{portfolio} \leq \sigma_{target}
$$

## Outputs

When the optimization succeeds, the app renders:

- Historical price charts for the selected assets
- Optimized portfolio weights
- Dollar allocation per asset
- Baseline vs. risk-adjusted rebalance summary
- Estimated trading cost impact
- Portfolio vs. `SPY` cumulative performance comparison
- Annual return, volatility, and Sharpe metrics
- Individual asset risk contribution chart

## Notes

- The optimization is long-only, with weights bounded between `0` and `1`
- All portfolio statistics are based on historical price data and should not be treated as forward-looking guarantees
- The benchmark comparison depends on `SPY` data being available and alignable with the selected portfolio history
