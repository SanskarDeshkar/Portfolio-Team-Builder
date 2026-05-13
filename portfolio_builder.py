import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from scipy.optimize import OptimizeResult, minimize
from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# streamlit run portfolio_builder.py

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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
POSITIVE_SPREAD_COLOR = "#2ca02c"
NEGATIVE_SPREAD_COLOR = "#d62728"
DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_ASSISTANT_MESSAGE = (
    "Ask me about your portfolio, risk target, allocations, or rebalancing costs. "
    "If an OpenAI API key is configured, I will answer with an AI model. "
    "Otherwise I will use the built-in portfolio assistant."
)
SIDEBAR_WIDTH = "23rem"
ASSET_COLORS = plt.get_cmap("tab10").colors


def apply_sidebar_styles() -> None:
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"]:not([aria-expanded="false"]) {{
            width: {SIDEBAR_WIDTH} !important;
            min-width: {SIDEBAR_WIDTH} !important;
        }}

        section[data-testid="stSidebar"]:not([aria-expanded="false"]) > div:first-child {{
            width: {SIDEBAR_WIDTH} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def collapse_sidebar() -> None:
    components.html(
        """
        <script>
        function collapseStreamlitSidebar() {
            const doc = window.parent.document;
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar || sidebar.getAttribute('aria-expanded') === 'false') {
                return true;
            }

            const directSelectors = [
                'button[aria-label="Close sidebar"]',
                'button[aria-label="Collapse sidebar"]',
                'button[title="Close sidebar"]',
                'button[title="Collapse sidebar"]',
                '[data-testid="stSidebarCollapseButton"] button',
                'button[data-testid="stSidebarCollapseButton"]'
            ];

            for (const selector of directSelectors) {
                const button = doc.querySelector(selector);
                if (button) {
                    button.click();
                    return true;
                }
            }

            const sidebarRect = sidebar.getBoundingClientRect();
            const buttons = Array.from(doc.querySelectorAll('button'));
            const sidebarButton = buttons.find((button) => {
                const label = [
                    button.getAttribute('aria-label') || '',
                    button.getAttribute('title') || '',
                    button.textContent || ''
                ].join(' ');
                if (/close sidebar|collapse sidebar/i.test(label)) {
                    return true;
                }

                const rect = button.getBoundingClientRect();
                return rect.top < 90 && Math.abs(rect.right - sidebarRect.right) < 80;
            });

            if (sidebarButton) {
                sidebarButton.click();
                return true;
            }

            return false;
        }

        let attempts = 0;
        const timer = setInterval(() => {
            attempts += 1;
            if (collapseStreamlitSidebar() || attempts >= 12) {
                clearInterval(timer);
            }
        }, 150);
        </script>
        """,
        height=0,
        width=0,
    )


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
    investment_amount: float,
) -> pd.DataFrame:
    if portfolio_volatility <= 1e-10:
        contributions = np.zeros(len(tickers))
    else:
        marginal_risk = covariance.values @ weights / portfolio_volatility
        component_risk = weights * marginal_risk
        contributions = component_risk / portfolio_volatility * 100

    risk_df = pd.DataFrame(
        {
            "Asset": tickers,
            "Investment ($)": weights * investment_amount,
            "Allocation (%)": weights * 100,
            "Risk Contribution (%)": np.round(contributions, 2),
        }
    )
    risk_df["Risk vs Allocation (pp)"] = risk_df["Risk Contribution (%)"] - risk_df["Allocation (%)"]
    return risk_df.sort_values(by="Risk Contribution (%)", ascending=False, ignore_index=True)


def assess_expected_return(portfolio_return: float) -> str:
    if portfolio_return >= 0.08:
        return "This is a strong historical return estimate"
    if portfolio_return >= 0.04:
        return "This is a moderate historical return estimate"
    if portfolio_return >= 0:
        return "This is a low but positive historical return estimate"
    return "This is a weak estimate because the expected return is negative"


def assess_volatility(portfolio_volatility: float) -> str:
    if portfolio_volatility <= 0.12:
        return "This is relatively low risk for a stock portfolio"
    if portfolio_volatility <= 0.20:
        return "This is a moderate level of risk"
    if portfolio_volatility <= 0.30:
        return "This is a high level of risk"
    return "This is very high risk and may feel uncomfortable for many investors"


def assess_sharpe_ratio(sharpe_ratio: float) -> str:
    if sharpe_ratio >= 1:
        return "This is generally good because the portfolio is getting solid return for the risk taken"
    if sharpe_ratio >= 0.5:
        return "This is okay, but the risk-adjusted return is not especially strong"
    if sharpe_ratio >= 0:
        return "This is weak because the portfolio is not getting much return for the risk taken"
    return "This is poor because the expected return is negative relative to the risk"


def assess_primary_risk(max_risk_value: float) -> str:
    if max_risk_value <= 35:
        return "This is fairly balanced because no single asset dominates the risk"
    if max_risk_value <= 50:
        return "This is somewhat concentrated, so this asset deserves attention"
    return "This is highly concentrated, meaning one asset is driving a large share of the portfolio's movement"


def assess_relative_risk(relative_risk_value: float) -> str:
    if relative_risk_value <= 0:
        return "That is not a concern on a relative basis because it contributes less risk than its allocation share"
    if relative_risk_value <= 10:
        return "That is only slightly elevated relative to the amount invested"
    if relative_risk_value <= 25:
        return "That is meaningfully elevated relative to the amount invested"
    return "That is highly elevated relative to the amount invested"


def assess_capital_shift(total_shift: float, investment_amount: float) -> str:
    shift_percent = total_shift / investment_amount if investment_amount else 0
    if shift_percent <= 0.01:
        return "This is very small, so the risk-adjusted portfolio is close to the baseline"
    if shift_percent <= 0.10:
        return "This is a moderate change, meaning the optimizer adjusted the mix but did not overhaul it"
    return "This is a large change, meaning your risk target materially changed the portfolio"


def create_dark_chart(figsize: tuple[int, int] = (10, 5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=PLOT_BACKGROUND)
    ax.set_facecolor(PLOT_BACKGROUND)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    return fig, ax


def get_asset_color_map(tickers: list[str]) -> dict[str, tuple[float, float, float]]:
    return {ticker: ASSET_COLORS[index % len(ASSET_COLORS)] for index, ticker in enumerate(tickers)}


def plot_price_history(price_history: pd.DataFrame, color_map: dict[str, tuple[float, float, float]]) -> None:
    fig, ax = create_dark_chart()
    colors = [color_map.get(ticker, PORTFOLIO_COLOR) for ticker in price_history.columns]
    price_history.plot(ax=ax, linewidth=2, color=colors)
    ax.set_title("Historical Closing Prices", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    legend = ax.legend(loc="upper left", frameon=False)
    if legend is not None:
        plt.setp(legend.get_texts(), color="white")
    st.pyplot(fig)
    plt.close(fig)


def calculate_benchmark_analysis(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    aligned_returns = pd.concat(
        [
            portfolio_returns.rename("Portfolio"),
            benchmark_returns.rename("SPY"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned_returns.empty:
        return {}

    portfolio_daily = aligned_returns["Portfolio"]
    benchmark_daily = aligned_returns["SPY"]
    active_daily = portfolio_daily - benchmark_daily
    benchmark_variance = float(benchmark_daily.var())
    tracking_error = float(active_daily.std() * np.sqrt(TRADING_DAYS))

    beta = 0.0 if benchmark_variance <= 1e-12 else float(portfolio_daily.cov(benchmark_daily) / benchmark_variance)
    portfolio_annual_return = float(portfolio_daily.mean() * TRADING_DAYS)
    benchmark_annual_return = float(benchmark_daily.mean() * TRADING_DAYS)
    portfolio_volatility = float(portfolio_daily.std() * np.sqrt(TRADING_DAYS))
    benchmark_volatility = float(benchmark_daily.std() * np.sqrt(TRADING_DAYS))
    alpha = portfolio_annual_return - beta * benchmark_annual_return
    relative_sharpe = 0.0 if tracking_error <= 1e-12 else float((portfolio_annual_return - benchmark_annual_return) / tracking_error)

    return {
        "alpha": alpha,
        "beta": beta,
        "relative_sharpe": relative_sharpe,
        "portfolio_annual_return": portfolio_annual_return,
        "benchmark_annual_return": benchmark_annual_return,
        "portfolio_volatility": portfolio_volatility,
        "benchmark_volatility": benchmark_volatility,
        "tracking_error": tracking_error,
    }


def plot_benchmark_comparison(portfolio_cumulative: pd.Series, spy_cumulative: pd.Series) -> None:
    spread = (portfolio_cumulative - spy_cumulative) * 100
    fig, ax = create_dark_chart()
    ax.axhline(0, color="white", linewidth=1, alpha=0.6)
    ax.plot(spread.index, spread.values, label="Portfolio Return - S&P 500 Return", color="white", linewidth=2)
    ax.fill_between(
        spread.index,
        spread.values,
        0,
        where=spread.values >= 0,
        color=POSITIVE_SPREAD_COLOR,
        alpha=0.35,
        interpolate=True,
        label="Portfolio ahead",
    )
    ax.fill_between(
        spread.index,
        spread.values,
        0,
        where=spread.values < 0,
        color=NEGATIVE_SPREAD_COLOR,
        alpha=0.35,
        interpolate=True,
        label="S&P 500 ahead",
    )
    ax.set_title("Relative Performance vs S&P 500", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Return Spread (percentage points)", color="white")
    ax.legend(facecolor=PLOT_BACKGROUND, frameon=False, loc="upper left", labelcolor="white")
    st.pyplot(fig)
    plt.close(fig)


def plot_risk_contribution(risk_df: pd.DataFrame, color_map: dict[str, tuple[float, float, float]]) -> None:
    fig, ax = create_dark_chart()
    x_positions = np.arange(len(risk_df))
    width = 0.38
    colors = [color_map.get(asset, PORTFOLIO_COLOR) for asset in risk_df["Asset"]]
    ax.bar(
        x_positions - width / 2,
        risk_df["Allocation (%)"],
        width,
        label="Allocation (%)",
        color=colors,
        alpha=0.45,
    )
    ax.bar(
        x_positions + width / 2,
        risk_df["Risk Contribution (%)"],
        width,
        label="Risk Contribution (%)",
        color=colors,
    )
    ax.set_title("Investment Allocation vs Risk Contribution", color="white")
    ax.set_ylabel("Percentage", color="white")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(risk_df["Asset"], color="white")
    ax.legend(facecolor=PLOT_BACKGROUND, frameon=False, loc="upper right", labelcolor="white")
    st.pyplot(fig)
    plt.close(fig)


def load_benchmark_returns() -> pd.Series | None:
    spy_prices = download_prices(("SPY",))
    if spy_prices.empty or "SPY" not in spy_prices.columns:
        return None
    return spy_prices["SPY"].pct_change().dropna()


def initialize_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]


def resolve_chat_api_key() -> str:
    session_key = st.session_state.get("chat_api_key", "").strip()
    if session_key:
        return session_key
    try:
        secret_value = st.secrets["OPENAI_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        secret_value = ""
    if secret_value:
        return str(secret_value).strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_portfolio_snapshot() -> str:
    snapshot = st.session_state.get("portfolio_snapshot")
    if not snapshot:
        return (
            "No optimization has been run yet. Ask the user to enter at least two tickers and run the optimizer "
            "if they want portfolio-specific analysis."
        )

    top_allocations = ", ".join(
        f"{row['Asset']} {row['Weight (%)']:.2f}%"
        for row in snapshot["allocation_df"].head(3).to_dict("records")
    )
    top_risk = ", ".join(
        f"{row['Asset']} {row['Risk Contribution (%)']:.2f}%"
        for row in snapshot["risk_df"].head(3).to_dict("records")
    )

    return (
        f"Tickers: {', '.join(snapshot['tickers'])}. "
        f"Investment amount: ${snapshot['investment_amount']:,.2f}. "
        f"Risk tolerance: {snapshot['risk_tolerance']} with target volatility {snapshot['target_volatility'] * 100:.2f}%. "
        f"Expected annual return: {snapshot['portfolio_return'] * 100:.2f}%. "
        f"Portfolio volatility: {snapshot['portfolio_volatility'] * 100:.2f}%. "
        f"Sharpe ratio: {snapshot['sharpe_ratio']:.2f}. "
        f"Total rebalancing cost: ${snapshot['total_fees']:,.2f}. "
        f"Net invested capital: ${snapshot['net_invested_capital']:,.2f}. "
        f"Total capital shift versus baseline: ${snapshot['total_shift']:,.2f}. "
        f"Top allocations: {top_allocations}. "
        f"Top risk contributors: {top_risk}."
    )


def build_chat_messages() -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for message in st.session_state["chat_messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "input_text", "text": message["content"]}],
            }
        )
    return messages


def generate_fallback_reply(prompt: str) -> str:
    snapshot = st.session_state.get("portfolio_snapshot")
    user_prompt = prompt.lower()

    if snapshot is None:
        return (
            "I can answer general questions now, but I need an optimization run before I can comment on allocations, "
            "risk contributions, or rebalancing results."
        )

    if any(keyword in user_prompt for keyword in ["allocation", "weight", "holding", "invest"]):
        top_rows = snapshot["allocation_df"].head(3).to_dict("records")
        summary = ", ".join(f"{row['Asset']} at {row['Weight (%)']:.2f}%" for row in top_rows)
        return f"The current portfolio leans most heavily toward {summary}."

    if any(keyword in user_prompt for keyword in ["risk", "volatile", "volatility"]):
        top_risk = snapshot["risk_df"].iloc[0]
        return (
            f"The portfolio's annualized volatility is {snapshot['portfolio_volatility'] * 100:.2f}%. "
            f"The largest risk contributor is {top_risk['Asset']} at {top_risk['Risk Contribution (%)']:.2f}% of total portfolio risk."
        )

    if any(keyword in user_prompt for keyword in ["rebalance", "fee", "cost", "shift"]):
        return (
            f"The rebalance moves ${snapshot['total_shift']:,.2f} relative to the baseline allocation and "
            f"incurs about ${snapshot['total_fees']:,.2f} in trading costs."
        )

    if any(keyword in user_prompt for keyword in ["return", "sharpe", "performance"]):
        return (
            f"The optimized portfolio has an expected annual return of {snapshot['portfolio_return'] * 100:.2f}% "
            f"with a Sharpe ratio of {snapshot['sharpe_ratio']:.2f}."
        )

    return (
        "I can help interpret the optimizer output. Ask about allocations, risk contributors, expected return, "
        "Sharpe ratio, or rebalancing costs."
    )


def generate_ai_reply(model: str, api_key: str) -> str:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "You are a portfolio assistant embedded in a Streamlit app. "
            "Use the provided portfolio snapshot when relevant. "
            "Be concise, practical, and do not give absolute financial guarantees.\n\n"
            f"Portfolio snapshot: {build_portfolio_snapshot()}"
        ),
        input=build_chat_messages(),
    )
    return response.output_text.strip()


def render_chatbot() -> None:
    initialize_chat_state()
    st.sidebar.divider()
    st.sidebar.subheader("AI Portfolio Chatbot")

    api_key = resolve_chat_api_key()
    using_openai = bool(api_key and OpenAI is not None)
    if using_openai:
        st.sidebar.caption(f"Chat mode: OpenAI `{DEFAULT_CHAT_MODEL}`")
    elif OpenAI is None:
        st.sidebar.caption("Chat mode: built-in portfolio assistant (`openai` package not installed)")
    else:
        st.sidebar.caption("Use the key button below to enable AI responses.")

    chat_history = st.sidebar.container()
    for message in st.session_state["chat_messages"]:
        speaker = "You" if message["role"] == "user" else "Assistant"
        chat_history.markdown(f"**{speaker}:** {message['content']}")

    with st.sidebar.form("chatbot_form", clear_on_submit=True):
        prompt = st.text_area(
            "Chat Message",
            placeholder="Ask about allocations, risk, or performance...",
            height=140,
        )
        submitted = st.form_submit_button("Send")

    st.sidebar.divider()
    if "show_personal_key_input_v2" not in st.session_state:
        st.session_state["show_personal_key_input_v2"] = False

    if st.sidebar.button("Use your own OpenAI API key", use_container_width=True):
        st.session_state["show_personal_key_input_v2"] = not st.session_state["show_personal_key_input_v2"]

    if st.session_state["show_personal_key_input_v2"]:
        st.sidebar.caption("Enter your key below to use OpenAI responses for the chatbot.")
        st.sidebar.text_input(
            "Your OpenAI API key",
            key="chat_api_key",
            type="password",
            help="Enter your own OpenAI API key to enable AI responses. Without it, the app uses the built-in portfolio assistant.",
        )

    if not submitted or not prompt.strip():
        return

    prompt = prompt.strip()
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.sidebar:
        with st.spinner("Thinking..."):
            try:
                if using_openai:
                    reply = generate_ai_reply(DEFAULT_CHAT_MODEL, api_key)
                else:
                    reply = generate_fallback_reply(prompt)
            except Exception as exc:
                reply = f"Chatbot error: {exc}"

    st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
    st.rerun()


def run_optimizer(
    tickers: list[str],
    investment_amount: float,
    risk_tolerance: str,
    target_volatility: float,
    fee_percent: float,
) -> None:
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
    asset_color_map = get_asset_color_map(active_tickers)
    if len(active_tickers) != len(tickers):
        st.info(f"Continuing with the valid assets only: {', '.join(active_tickers)}")

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
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(
        optimized_weights,
        mean_returns,
        covariance,
    )
    sharpe_display = sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0

    portfolio_daily_returns = returns.dot(optimized_weights)
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
    benchmark_returns = load_benchmark_returns()
    if benchmark_returns is not None:
        benchmark_analysis = calculate_benchmark_analysis(portfolio_daily_returns, benchmark_returns)
        comparison_df = pd.concat(
            [
                portfolio_cumulative.rename("Portfolio"),
                (1 + benchmark_returns).cumprod().rename("SPY"),
            ],
            axis=1,
            join="inner",
        ).dropna()
    else:
        benchmark_analysis = {}
        comparison_df = pd.DataFrame()

    risk_df = build_risk_contribution_table(
        active_tickers,
        optimized_weights,
        covariance,
        portfolio_volatility,
        investment_amount,
    )
    rebalance_df = build_rebalance_table(
        active_tickers,
        baseline_weights,
        optimized_weights,
        investment_amount,
        fee_percent,
    )
    total_fees = float(rebalance_df["Trading Cost ($)"].sum())
    net_invested_capital = investment_amount - total_fees
    total_shift = float(rebalance_df["Shift ($)"].abs().sum())

    max_risk_asset = risk_df.iloc[0]["Asset"]
    max_risk_value = risk_df.iloc[0]["Risk Contribution (%)"]
    relative_risk_df = risk_df.sort_values(by="Risk vs Allocation (pp)", ascending=False, ignore_index=True)
    relative_risk_asset = relative_risk_df.iloc[0]["Asset"]
    relative_risk_value = relative_risk_df.iloc[0]["Risk vs Allocation (pp)"]
    relative_risk_contribution = relative_risk_df.iloc[0]["Risk Contribution (%)"]
    relative_risk_allocation = relative_risk_df.iloc[0]["Allocation (%)"]

    st.success("Optimization complete.")
    analysis_tabs = [
        "Overview",
        "Price History",
        "Allocation",
        "Benchmark",
        "Risk",
        "Rebalance",
    ]
    active_tab = st.segmented_control(
        "Analysis section",
        analysis_tabs,
        default="Overview",
        key="active_analysis_tab",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_tab is None:
        active_tab = "Overview"

    with st.container(height=900, border=False):
        if active_tab == "Overview":
            st.write("### Portfolio Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Annual Return", f"{portfolio_return * 100:.2f}%")
            with col2:
                st.metric("Annual Volatility (Risk)", f"{portfolio_volatility * 100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_display:.2f}")
    
            st.info(
                f"""
                **Portfolio Interpretation:**
                This is the quick health check for the portfolio the app built from your selected tickers and risk tolerance.
    
                - **Expected Annual Return:** The model estimates **{portfolio_return * 100:.2f}%** per year. **{assess_expected_return(portfolio_return)}.** This means the optimizer is using recent history to estimate yearly growth, but it is not a guarantee.
                - **Annual Volatility (Risk):** The portfolio's estimated volatility is **{portfolio_volatility * 100:.2f}%**. **{assess_volatility(portfolio_volatility)}.** This tells you how much the account value may swing up and down in a typical year; higher volatility means a bumpier ride.
                - **Sharpe Ratio:** The score is **{sharpe_display:.2f}**. **{assess_sharpe_ratio(sharpe_display)}.** This compares return against risk, so it helps answer whether the portfolio is being rewarded enough for the risk it takes.
                - **Primary Risk Driver:** On a raw basis, **{max_risk_asset}** contributes the most total risk at **{max_risk_value:.2f}%**. Adjusting for how much money is invested, **{relative_risk_asset}** stands out most: it is **{relative_risk_allocation:.2f}%** of the portfolio but contributes **{relative_risk_contribution:.2f}%** of total risk, a difference of **{relative_risk_value:.2f} percentage points**. **{assess_relative_risk(relative_risk_value)}.**
                - **Capital Shift:** The risk-adjusted portfolio moves **${total_shift:,.2f}** compared with the baseline max-Sharpe portfolio. **{assess_capital_shift(total_shift, investment_amount)}.** This shows how much money the optimizer would reallocate to better match your selected risk level.
                """
            )
    
        elif active_tab == "Price History":
            st.write("### Asset Performance Data")
            st.info(
                "This tab shows the historical price data used by the optimizer. The chart gives a visual comparison "
                "of the selected assets over time, and the table shows the cleaned price history behind the calculations. "
                "Use the year filter to inspect a specific period; the chart, raw price table, and summary statistics "
                "all update to that selected period."
            )
            available_years = sorted(price_history.index.year.unique(), reverse=True)
            selected_year = st.selectbox("Year", ["All years", *available_years], key="price_history_year")
            filtered_price_history = price_history
            if selected_year != "All years":
                filtered_price_history = price_history[price_history.index.year == selected_year]
    
            plot_price_history(filtered_price_history, asset_color_map)
            filtered_returns = filtered_price_history.pct_change().dropna()
            if filtered_returns.empty:
                st.info("There is not enough data in this period to calculate return statistics.")
            else:
                price_summary = pd.DataFrame(
                    {
                        "Asset": filtered_price_history.columns,
                        "Mean Price ($)": filtered_price_history.mean().values,
                        "Total Return (%)": (
                            (filtered_price_history.iloc[-1] / filtered_price_history.iloc[0] - 1) * 100
                        ).values,
                        "Mean Daily Return (%)": (filtered_returns.mean() * 100).values,
                        "Annualized Volatility (%)": (filtered_returns.std() * np.sqrt(TRADING_DAYS) * 100).values,
                    }
                )
                st.dataframe(
                    price_summary.style.format(
                        {
                            "Mean Price ($)": "${:,.2f}",
                            "Total Return (%)": "{:,.2f}%",
                            "Mean Daily Return (%)": "{:,.3f}%",
                            "Annualized Volatility (%)": "{:,.2f}%",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            st.dataframe(filtered_price_history, use_container_width=True)
    
        elif active_tab == "Allocation":
            st.write("### Optimized Portfolio Weights")
            st.info(
                "This tab shows the risk-adjusted portfolio allocation. Weight is the percent of your portfolio assigned "
                "to each ticker, and Cash to Invest converts that weight into dollars using your investment amount."
            )
            st.dataframe(
                allocation_df.style.format(
                    {
                        "Weight (%)": "{:.2f}%",
                        "Cash to Invest ($)": "${:,.2f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
    
        elif active_tab == "Benchmark":
            st.write("### Optimized Portfolio vs S&P 500")
            st.info(
                "This tab compares the optimized portfolio with the S&P 500 on a risk-adjusted basis. "
                "Alpha estimates return above or below what beta would predict, Beta shows sensitivity to S&P 500 "
                "movement, and Relative Sharpe compares extra return against tracking risk. The relative performance "
                "chart shows when your portfolio is ahead or behind."
            )
            if not benchmark_analysis or comparison_df.empty:
                st.info("Benchmark comparison is unavailable because SPY data could not be aligned with the portfolio history.")
            else:
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Alpha", f"{benchmark_analysis['alpha'] * 100:.2f}%")
                with metric_col2:
                    st.metric("Beta", f"{benchmark_analysis['beta']:.2f}")
                with metric_col3:
                    st.metric("Benchmark-Relative Sharpe", f"{benchmark_analysis['relative_sharpe']:.2f}")

                st.write("### Relative Performance Spread")
                plot_benchmark_comparison(comparison_df["Portfolio"], comparison_df["SPY"])
                if benchmark_analysis["portfolio_annual_return"] > benchmark_analysis["benchmark_annual_return"]:
                    st.success("The optimized portfolio has a higher annualized return than the S&P 500 over the shared period.")
                else:
                    st.warning("The optimized portfolio has a lower annualized return than the S&P 500 over the shared period.")
    
        elif active_tab == "Risk":
            st.write("### Individual Risk Attribution")
            st.info(
                "This tab compares how much money is invested in each asset with how much risk that asset adds. "
                "Risk vs Allocation shows whether a ticker contributes more or less risk than its portfolio weight "
                "would suggest."
            )
            plot_risk_contribution(risk_df, asset_color_map)
            st.dataframe(
                risk_df.style.format(
                    {
                        "Investment ($)": "${:,.2f}",
                        "Allocation (%)": "{:,.2f}%",
                        "Risk Contribution (%)": "{:,.2f}%",
                        "Risk vs Allocation (pp)": "{:,.2f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
    
        elif active_tab == "Rebalance":
            st.write("### Rebalance Summary")
            st.info(
                "**How to read this table:** Baseline is the portfolio the optimizer would choose if it only tried to "
                "maximize return per unit of risk. Risk-Adjusted is the version that also respects your selected risk "
                "tolerance. Shift shows how many dollars move into or out of each asset, Trading Cost estimates the fee "
                "on that move, and Net Shift shows the final dollar change after that cost."
            )
            st.dataframe(
                rebalance_df.style.format(
                    {
                        "Baseline ($)": "${:,.2f}",
                        "Risk-Adjusted ($)": "${:,.2f}",
                        "Shift ($)": "${:,.2f}",
                        "Trading Cost ($)": "${:,.2f}",
                        "Net Shift ($)": "${:,.2f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.metric(
                "Total Rebalancing Cost",
                f"${total_fees:,.2f}",
                delta=f"{fee_percent * 100:,.2f}% Fee",
                delta_color="inverse",
            )
            st.write(f"**Net Invested Capital after Rebalancing Costs:** ${net_invested_capital:,.2f}")
            if total_fees > 0.01:
                st.warning(f"Total trading costs for this rebalance will be ${total_fees:,.2f}.")
    
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
    
    st.session_state["portfolio_snapshot"] = {
        "tickers": active_tickers,
        "investment_amount": investment_amount,
        "risk_tolerance": risk_tolerance,
        "target_volatility": target_volatility,
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_display,
        "total_fees": total_fees,
        "net_invested_capital": net_invested_capital,
        "total_shift": total_shift,
        "allocation_df": allocation_df,
        "risk_df": risk_df,
    }


def render_app() -> None:
    apply_sidebar_styles()
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

    if st.button("Optimize the tickers"):
        st.session_state["optimizer_inputs"] = {
            "tickers": tickers,
            "investment_amount": investment_amount,
            "risk_tolerance": risk_tolerance,
            "target_volatility": target_volatility,
            "fee_percent": fee_percent,
        }
        st.session_state["collapse_sidebar_after_optimize"] = True

    if st.session_state.get("collapse_sidebar_after_optimize"):
        collapse_sidebar()
        st.session_state["collapse_sidebar_after_optimize"] = False

    optimizer_inputs = st.session_state.get("optimizer_inputs")
    if optimizer_inputs:
        run_optimizer(**optimizer_inputs)

    render_chatbot()


render_app()
