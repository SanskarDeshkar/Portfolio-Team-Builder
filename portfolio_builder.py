import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import OptimizeResult, minimize

TRADING_DAYS = 252
DEFAULT_TICKERS = "AAPL, MSFT, JNJ, XOM, GLD"
RISK_MAPPING = {
    "Low (Stable)": 0.10,
    "Medium (Balanced)": 0.18,
    "High (Aggressive)": 0.30,
}
PLOT_BACKGROUND = "#0f1116"
PORTFOLIO_COLOR = "#1f77b4"
BENCHMARK_COLOR = "#ff7f0e"


def parse_tickers(raw_input: str) -> list[str]:
    seen = set()
    tickers = []
    for value in raw_input.split(","):
        ticker = value.strip().upper()
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers


def ensure_dataframe(data: pd.Series | pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        column_name = columns[0] if columns else data.name or "Asset"
        return data.to_frame(name=column_name)

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]

    return data.copy()


@st.cache_data(show_spinner=False)
def download_prices(tickers: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    downloaded = yf.download(
        list(tickers),
        period=period,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    prices = ensure_dataframe(downloaded, list(tickers))
    prices = prices.reindex(columns=list(tickers))
    return prices.dropna(axis=1, how="all")


def prepare_price_history(tickers: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    prices = download_prices(tuple(tickers))
    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]

    if prices.empty:
        return None, tickers

    cleaned = prices.dropna(axis=0, how="any")
    if cleaned.empty:
        return None, missing_tickers

    return cleaned, missing_tickers


def annualized_statistics(price_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    returns = price_history.pct_change().dropna()
    mean_returns = returns.mean() * TRADING_DAYS
    covariance = returns.cov() * TRADING_DAYS
    return returns, mean_returns, covariance


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[float, float, float]:
    portfolio_return = float(np.dot(mean_returns.values, weights))
    variance = float(weights.T @ covariance.values @ weights)
    portfolio_volatility = float(np.sqrt(max(variance, 0.0)))
    if portfolio_volatility <= 1e-10:
        sharpe_ratio = -np.inf
    else:
        sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    total = clipped.sum()
    if total <= 0:
        raise ValueError("Optimizer returned non-positive total weight.")
    return clipped / total


def optimize_weights(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    target_volatility: float | None = None,
) -> tuple[np.ndarray, OptimizeResult]:
    asset_count = len(mean_returns)
    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    initial_guess = np.full(asset_count, 1.0 / asset_count)

    def objective(weights: np.ndarray) -> float:
        return -portfolio_performance(weights, mean_returns, covariance)[2]

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    if target_volatility is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: target_volatility - portfolio_performance(x, mean_returns, covariance)[1],
            }
        )

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500},
    )
    if not result.success:
        raise ValueError(result.message)

    return normalize_weights(result.x), result


def minimize_volatility(covariance: pd.DataFrame) -> tuple[np.ndarray, float]:
    asset_count = len(covariance)
    bounds = tuple((0.0, 1.0) for _ in range(asset_count))
    initial_guess = np.full(asset_count, 1.0 / asset_count)

    def objective(weights: np.ndarray) -> float:
        variance = float(weights.T @ covariance.values @ weights)
        return float(np.sqrt(max(variance, 0.0)))

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}],
        options={"maxiter": 500},
    )
    if not result.success:
        raise ValueError(result.message)

    weights = normalize_weights(result.x)
    min_volatility = objective(weights)
    return weights, min_volatility


def build_allocation_table(tickers: list[str], weights: np.ndarray, investment_amount: float) -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "Asset": tickers,
            "Weight (%)": weights * 100,
            "Cash to Invest ($)": weights * investment_amount,
        }
    ).sort_values(by="Weight (%)", ascending=False, ignore_index=True)
    return results


def build_rebalance_table(
    tickers: list[str],
    baseline_weights: np.ndarray,
    optimized_weights: np.ndarray,
    investment_amount: float,
    fee_percent: float,
) -> pd.DataFrame:
    rebalance_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Baseline ($)": baseline_weights * investment_amount,
            "Risk-Adjusted ($)": optimized_weights * investment_amount,
        }
    )
    rebalance_df["Shift ($)"] = rebalance_df["Risk-Adjusted ($)"] - rebalance_df["Baseline ($)"]
    rebalance_df["Trading Cost ($)"] = rebalance_df["Shift ($)"].abs() * fee_percent
    rebalance_df["Net Shift ($)"] = rebalance_df["Shift ($)"] - rebalance_df["Trading Cost ($)"]
    return rebalance_df


def build_risk_contribution_table(
    tickers: list[str],
    weights: np.ndarray,
    covariance: pd.DataFrame,
    portfolio_volatility: float,
) -> pd.DataFrame:
    if portfolio_volatility <= 1e-10:
        contributions = np.zeros(len(tickers))
    else:
        marginal_risk = covariance.values @ weights / portfolio_volatility
        component_risk = weights * marginal_risk
        contributions = component_risk / portfolio_volatility * 100

    return pd.DataFrame(
        {
            "Asset": tickers,
            "Risk Contribution (%)": np.round(contributions, 2),
        }
    ).sort_values(by="Risk Contribution (%)", ascending=False, ignore_index=True)


def create_dark_chart(figsize: tuple[int, int] = (10, 5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=PLOT_BACKGROUND)
    ax.set_facecolor(PLOT_BACKGROUND)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    return fig, ax


def plot_price_history(price_history: pd.DataFrame) -> None:
    fig, ax = create_dark_chart()
    price_history.plot(ax=ax, linewidth=2)
    ax.set_title("Historical Closing Prices", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    legend = ax.legend(loc="upper left", frameon=False)
    if legend is not None:
        plt.setp(legend.get_texts(), color="white")
    st.pyplot(fig)
    plt.close(fig)


def plot_benchmark_comparison(portfolio_cumulative: pd.Series, spy_cumulative: pd.Series) -> None:
    fig, ax = create_dark_chart()
    ax.plot(portfolio_cumulative.index, portfolio_cumulative.values, label="Optimized Portfolio", color=PORTFOLIO_COLOR, linewidth=2)
    ax.plot(spy_cumulative.index, spy_cumulative.values, label="S&P 500 (SPY)", color=BENCHMARK_COLOR, linewidth=2)
    ax.set_title("Portfolio Performance vs S&P 500", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Cumulative Return", color="white")
    ax.legend(facecolor=PLOT_BACKGROUND, frameon=False, loc="upper left", labelcolor="white")
    st.pyplot(fig)
    plt.close(fig)


def plot_risk_contribution(risk_df: pd.DataFrame) -> None:
    fig, ax = create_dark_chart()
    ax.bar(risk_df["Asset"], risk_df["Risk Contribution (%)"], color=PORTFOLIO_COLOR)
    ax.set_title("Individual Asset Risk Contribution", color="white")
    ax.set_ylabel("Percentage of Portfolio Risk", color="white")
    st.pyplot(fig)
    plt.close(fig)


def load_benchmark_returns() -> pd.Series | None:
    spy_prices = download_prices(("SPY",))
    if spy_prices.empty or "SPY" not in spy_prices.columns:
        return None
    return spy_prices["SPY"].pct_change().dropna()


def render_app() -> None:
    st.title("Quant Portfolio Team Builder")
    st.write("Draft the assets and optimize the risk.")

    st.subheader("Enter the ticker symbols of the assets you want to include in your portfolio (separated by commas):")
    tickers_input = st.text_input("Ticker Symbols", DEFAULT_TICKERS)
    tickers = parse_tickers(tickers_input)

    investment_amount = st.number_input(
        "Enter the total amount you want to invest in the portfolio:",
        min_value=100.0,
        value=1000.0,
        step=100.0,
    )

    st.subheader("Select your risk tolerance level:")
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=list(RISK_MAPPING.keys()),
        value="Medium (Balanced)",
    )
    target_volatility = RISK_MAPPING[risk_tolerance]

    st.sidebar.subheader("Trading Parameters")
    fee_percent = (
        st.sidebar.number_input(
            "Trading Fee (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.05,
        )
        / 100
    )

    if not st.button("Optimize the tickers"):
        return

    if len(tickers) < 2:
        st.error("Enter at least two unique ticker symbols to build a portfolio.")
        return

    st.write(f"Fetching data for: {', '.join(tickers)}")

    try:
        price_history, missing_tickers = prepare_price_history(tickers)
    except Exception as exc:
        st.error(f"Unable to download price history: {exc}")
        return

    if missing_tickers:
        st.warning(f"No usable history was found for: {', '.join(missing_tickers)}")

    if price_history is None or price_history.shape[1] < 2:
        st.error("Not enough valid ticker history was available to optimize the portfolio.")
        return

    active_tickers = price_history.columns.tolist()
    if len(active_tickers) != len(tickers):
        st.info(f"Continuing with the valid assets only: {', '.join(active_tickers)}")

    st.subheader("Asset Performance Data")
    plot_price_history(price_history)
    st.dataframe(price_history)
    st.success("Data loaded. Ready to start optimization math.")

    st.subheader("Portfolio Optimization")
    returns, mean_returns, covariance = annualized_statistics(price_history)
    if returns.empty:
        st.error("There was not enough return history to run the optimizer.")
        return

    try:
        _, min_volatility = minimize_volatility(covariance)
    except ValueError as exc:
        st.error(f"Unable to solve the minimum-volatility portfolio: {exc}")
        return

    if target_volatility + 1e-6 < min_volatility:
        st.error(
            f"The selected risk target is infeasible for this basket. "
            f"Minimum achievable annual volatility is {min_volatility * 100:.2f}%."
        )
        return

    try:
        baseline_weights, _ = optimize_weights(mean_returns, covariance)
        optimized_weights, _ = optimize_weights(mean_returns, covariance, target_volatility)
    except ValueError as exc:
        st.error(f"Optimization failed: {exc}")
        return

    allocation_df = build_allocation_table(active_tickers, optimized_weights, investment_amount)
    st.write("### Optimized Portfolio Weights:")
    st.table(
        allocation_df.style.format(
            {
                "Weight (%)": "{:.2f}%",
                "Cash to Invest ($)": "${:,.2f}",
            }
        )
    )
    st.success("Optimization complete.")

    rebalance_df = build_rebalance_table(
        active_tickers,
        baseline_weights,
        optimized_weights,
        investment_amount,
        fee_percent,
    )
    st.subheader("Rebalance Summary")
    st.write(
        "This table shows how much you would invest in each asset based on the baseline "
        "(max Sharpe ratio) versus the risk-adjusted portfolio, along with the shift in dollars."
    )
    st.table(
        rebalance_df.style.format(
            {
                "Baseline ($)": "${:,.2f}",
                "Risk-Adjusted ($)": "${:,.2f}",
                "Shift ($)": "${:,.2f}",
                "Trading Cost ($)": "${:,.2f}",
                "Net Shift ($)": "${:,.2f}",
            }
        )
    )

    total_fees = float(rebalance_df["Trading Cost ($)"].sum())
    if total_fees > 0.01:
        st.warning(f"Total trading costs for this rebalance will be ${total_fees:,.2f}.")

    net_invested_capital = investment_amount - total_fees
    total_shift = float(rebalance_df["Shift ($)"].abs().sum())

    st.divider()
    st.metric(
        "Total Rebalancing Cost",
        f"${total_fees:,.2f}",
        delta=f"{fee_percent * 100:,.2f}% Fee",
        delta_color="inverse",
    )
    st.write(f"**Net Invested Capital after Rebalancing Costs:** ${net_invested_capital:,.2f}")

    if total_shift < 0.01:
        st.info(
            "The optimized portfolio is very similar to the baseline max Sharpe ratio portfolio, "
            "which suggests the selected risk target was already satisfied."
        )
    else:
        st.info(
            f"The optimized portfolio shifts ${total_shift:,.2f} versus the baseline max Sharpe "
            "ratio portfolio, indicating the risk target materially changed the allocations."
        )

    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(
        optimized_weights,
        mean_returns,
        covariance,
    )

    benchmark_returns = load_benchmark_returns()
    portfolio_cumulative = (1 + returns.dot(optimized_weights)).cumprod()
    if benchmark_returns is not None:
        comparison_df = pd.concat(
            [
                portfolio_cumulative.rename("Portfolio"),
                (1 + benchmark_returns).cumprod().rename("SPY"),
            ],
            axis=1,
            join="inner",
        ).dropna()
    else:
        comparison_df = pd.DataFrame()

    st.subheader("Optimized Portfolio vs S&P 500")
    if comparison_df.empty:
        st.info("Benchmark comparison is unavailable because SPY data could not be aligned with the portfolio history.")
    else:
        plot_benchmark_comparison(comparison_df["Portfolio"], comparison_df["SPY"])
        if comparison_df["Portfolio"].iloc[-1] > comparison_df["SPY"].iloc[-1]:
            st.success("Your optimized portfolio outperformed the S&P 500 over the shared comparison period.")
        else:
            st.warning("Your optimized portfolio underperformed the S&P 500 over the shared comparison period.")

    st.write("### Portfolio Performance:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Annual Return", f"{portfolio_return * 100:.2f}%")
    with col2:
        st.metric("Annual Volatility (Risk)", f"{portfolio_volatility * 100:.2f}%")
    with col3:
        sharpe_display = sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0
        st.metric("Sharpe Ratio", f"{sharpe_display:.2f}")

    st.subheader("Individual Risk Attribution")
    risk_df = build_risk_contribution_table(active_tickers, optimized_weights, covariance, portfolio_volatility)
    plot_risk_contribution(risk_df)

    max_risk_asset = risk_df.iloc[0]["Asset"]
    max_risk_value = risk_df.iloc[0]["Risk Contribution (%)"]
    st.info(
        f"""
        **Portfolio Interpretation:**
        - **Annual Return:** Expected **{portfolio_return * 100:.2f}%** based on 5-year historical trends.
        - **Volatility:** At **{portfolio_volatility * 100:.2f}%**, this represents the expected annualized price swing.
        - **Sharpe Ratio:** A score of **{sharpe_display:.2f}** indicates your return efficiency per unit of risk.
        - **Primary Risk Driver:** **{max_risk_asset}** contributes **{max_risk_value:.2f}%** of total portfolio risk.
        """
    )


render_app()
