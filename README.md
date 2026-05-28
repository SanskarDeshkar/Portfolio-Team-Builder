# Quant Portfolio Optimizer (Team Builder)

An AI-powered Streamlit app for portfolio construction, risk-constrained optimization, and rebalancing analysis. The app pulls historical market data, computes annualized return and volatility statistics, and uses SLSQP optimization to build a maximum-Sharpe portfolio that can also respect a user-selected volatility ceiling.

## Current State

This project now includes:

- A runnable Streamlit app in `portfolio_builder.py`
- A built-in chat interface using Streamlit's chat components
- Local Ollama-backed chatbot responses with a built-in fallback assistant
- Sidebar trading-fee controls
- Dark themed charts, centered result tables, and responsive layout styling
- Python project metadata in `pyproject.toml`
- Dependency install support in `requirements.txt`
- A basic repo setup for local virtual environment workflows

The app has also been hardened around invalid tickers, missing benchmark data, infeasible volatility targets, and optimizer failures so that common data or market-data issues produce user-facing messages instead of uncaught exceptions.

## What The App Does

- Accepts a custom basket of ticker symbols
- Downloads approximately five years of adjusted price history with `yfinance`
- Builds an unconstrained maximum-Sharpe baseline portfolio
- Builds a risk-adjusted portfolio using the selected volatility target
- Computes dollar allocations from the user's investment amount
- Estimates trading costs for the rebalance
- Compares portfolio performance against `SPY`
- Breaks down each asset's contribution to total portfolio risk
- Lets users chat with a portfolio assistant directly in the sidebar
- Uses local Ollama AI by default when available
- Falls back to a local rule-based assistant if Ollama is unavailable or the OpenAI client package is not installed

## Tech Stack

- **Language**: Python 3.11+
- **App Framework**: Streamlit
- **Data Source**: yfinance
- **Numerical Libraries**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib
- **Chat**: Streamlit sidebar chat UI with local Ollama support through an OpenAI-compatible client
- **Configuration**: python-dotenv, Streamlit secrets, and environment variables

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

5. Optional: enable local Ollama chatbot responses:

Install and start Ollama, then pull the local model used by the app:

```bash
ollama pull llama3.2
```

Keep the Ollama app or server running in the background. The Streamlit chatbot connects to Ollama at:

```text
http://localhost:11434/v1/
```

When no personal OpenAI API key is configured, the app uses Ollama with `llama3.2` for AI chatbot responses. If Ollama is not running, the app falls back to the built-in rule-based portfolio assistant.

6. Optional: use your own OpenAI API key instead of local Ollama:

```bash
export OPENAI_API_KEY=your_api_key_here
```

You can also place the key in a local `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

Or configure it through Streamlit secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

The app also includes a sidebar button for entering a personal API key at runtime. If a key is configured, the chatbot uses OpenAI instead of local Ollama.

7. Optional: choose a different OpenAI chat model for the personal-key mode:

```bash
export OPENAI_CHAT_MODEL=gpt-5-mini
```

If `OPENAI_CHAT_MODEL` is not set, the app defaults to `gpt-5-mini`.

## Local Checks

Run a quick syntax check before committing changes:

```bash
python3 -m py_compile portfolio_builder.py
```

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
- Filterable annual price-history summaries
- Optimized portfolio weights
- Dollar allocation per asset
- Baseline vs. risk-adjusted rebalance summary
- Estimated trading cost impact
- Portfolio vs. `SPY` relative performance comparison
- Alpha, beta, benchmark-relative Sharpe, and tracking-risk metrics
- Annual return, volatility, and Sharpe metrics
- Individual asset risk contribution chart
- Plain-English interpretation of return, volatility, Sharpe ratio, risk concentration, and capital shift

## Notes

- The optimization is long-only, with weights bounded between `0` and `1`
- All portfolio statistics are based on historical price data and should not be treated as forward-looking guarantees
- The benchmark comparison depends on `SPY` data being available and alignable with the selected portfolio history
- The constrained optimization will stop with a clear message if the selected volatility target is lower than the basket's minimum achievable volatility
- The default AI chatbot path expects Ollama to be running locally with the `llama3.2` model pulled
