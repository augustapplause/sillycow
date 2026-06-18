import html
import concurrent.futures
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def money0(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def money2(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def signed_money0(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def signed_pct1(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:+.1f}%"


def normalize_ohlcv_columns(data, ticker=None):
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        required = {"Open", "High", "Low", "Close", "Volume"}
        level0 = set(map(str, data.columns.get_level_values(0)))
        levellast = set(map(str, data.columns.get_level_values(-1)))

        if required.issubset(level0):
            data.columns = data.columns.get_level_values(0)
        elif required.issubset(levellast):
            data.columns = data.columns.get_level_values(-1)
        elif ticker is not None:
            ticker_upper = str(ticker).upper()
            for level in range(data.columns.nlevels):
                labels = [str(x).upper() for x in data.columns.get_level_values(level)]
                if ticker_upper in labels:
                    key = data.columns.get_level_values(level)[labels.index(ticker_upper)]
                    data = data.xs(key=key, axis=1, level=level)
                    break

    data = data.loc[:, ~data.columns.duplicated()]
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data for {ticker or 'ticker'} is missing columns: {missing}")

    out = data[required_cols].dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlcv(ticker: str, period: str):
    ticker = str(ticker).strip().upper()
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_ohlcv_columns(raw, ticker=ticker)





def _run_with_timeout(func, timeout_seconds=4, fallback=None):
    """Run a potentially slow yfinance call with a short app-safe timeout."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except Exception:
        return fallback
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_live_quote(ticker: str):
    """Return a live-ish yfinance quote without letting startup hang.

    The quote is opportunistic: fast_info is attempted first, then a short
    intraday-history fallback. If Yahoo/yfinance is slow, the app continues
    and displays N/A rather than spinning indefinitely.
    """
    ticker = str(ticker).strip().upper()
    out = {"last_price": np.nan, "previous_close": np.nan}
    if not ticker:
        return out

    def _fast_info_quote():
        yticker = yf.Ticker(ticker)
        fast_info = yticker.fast_info

        def _fast_value(key):
            try:
                return fast_info.get(key) if hasattr(fast_info, "get") else fast_info[key]
            except Exception:
                return None

        result = {"last_price": np.nan, "previous_close": np.nan}
        for key in ["lastPrice", "last_price", "regularMarketPrice"]:
            value = _fast_value(key)
            if value is not None and np.isfinite(float(value)):
                result["last_price"] = float(value)
                break

        for key in ["previousClose", "previous_close", "regularMarketPreviousClose"]:
            value = _fast_value(key)
            if value is not None and np.isfinite(float(value)):
                result["previous_close"] = float(value)
                break
        return result

    fast_result = _run_with_timeout(_fast_info_quote, timeout_seconds=3, fallback=None)
    if isinstance(fast_result, dict):
        out.update(fast_result)

    if pd.isna(out["last_price"]):
        def _intraday_last_price():
            intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
            if intraday is not None and not intraday.empty and "Close" in intraday.columns:
                closes = intraday["Close"].dropna()
                if not closes.empty:
                    return float(closes.iloc[-1])
            return np.nan

        last_price = _run_with_timeout(_intraday_last_price, timeout_seconds=3, fallback=np.nan)
        if last_price is not None and pd.notna(last_price) and np.isfinite(float(last_price)):
            out["last_price"] = float(last_price)

    return out


def get_live_price(ticker: str):
    return get_live_quote(ticker).get("last_price", np.nan)

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_display_name(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if not ticker:
        return ""

    def _lookup_name():
        info = yf.Ticker(ticker).get_info()
        if not isinstance(info, dict):
            return ""
        for key in ["longName", "shortName", "displayName"]:
            value = info.get(key)
            if value and str(value).strip().upper() != ticker:
                return str(value).strip()
        return ""

    return _run_with_timeout(_lookup_name, timeout_seconds=3, fallback="") or ""


def clv(df):
    return ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"] + 1e-9
    )


def atr(df, period=20):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def _rank(x):
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1] * 100

    return series.rolling(window, min_periods=max(30, window // 3)).apply(_rank, raw=False)


def make_rs_col_name(benchmark: str) -> str:
    suffix = "".join(ch if ch.isalnum() else "_" for ch in str(benchmark).strip().upper()).strip("_")
    return f"RS_20_{suffix}" if suffix else "RS_20"


def build_rs_col_map(benchmarks: List[str]) -> dict:
    out = {}
    for benchmark in benchmarks:
        benchmark = str(benchmark).strip().upper()
        if not benchmark or benchmark in out:
            continue
        out[benchmark] = make_rs_col_name(benchmark)
    return out


def add_profile_columns(df: pd.DataFrame, benchmark_dfs=None) -> pd.DataFrame:
    df = df.copy()

    df["CLV"] = clv(df)
    df["CLV_5"] = df["CLV"].rolling(5).mean()
    df["CLV_20"] = df["CLV"].rolling(20).mean()
    df["CLV_Trend"] = df["CLV_5"] - df["CLV_20"]

    df["ATR10"] = atr(df, 10)
    df["ATR50"] = atr(df, 50)
    df["Compression_Ratio"] = df["ATR10"] / (df["ATR50"] + 1e-9)

    df["Volume_20_Mean"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_20_Mean"] + 1e-9)

    df["Future_Close_5D"] = df["Close"].shift(-5)
    df["Forward_Return_5D"] = (df["Future_Close_5D"] / df["Close"] - 1) * 100

    if isinstance(benchmark_dfs, pd.DataFrame):
        benchmark_dfs = {"Benchmark": benchmark_dfs}
    elif benchmark_dfs is None:
        benchmark_dfs = {}

    first_rs_col = None
    for benchmark, benchmark_df in benchmark_dfs.items():
        benchmark = str(benchmark).strip().upper()
        rs_col = make_rs_col_name(benchmark)
        df[rs_col] = np.nan

        if benchmark_df is not None and not benchmark_df.empty:
            aligned = df[["Close"]].rename(columns={"Close": "Stock_Close"}).join(
                benchmark_df[["Close"]].rename(columns={"Close": "Benchmark_Close"}),
                how="left",
            )
            aligned["Benchmark_Close"] = aligned["Benchmark_Close"].ffill()
            df[rs_col] = (
                aligned["Stock_Close"].pct_change(20)
                - aligned["Benchmark_Close"].pct_change(20)
            ) * 100

        if first_rs_col is None:
            first_rs_col = rs_col

    if first_rs_col is not None:
        df["RS_20"] = df[first_rs_col]
    else:
        df["RS_20"] = np.nan

    return df


STATE_ORDER = ["Bear", "Sideways", "Bull"]
STATE_COLORS = {"Bear": "red", "Sideways": "#111936", "Bull": "green"}


def _safe_softmax_normalize(arr, axis=None):
    arr = np.asarray(arr, dtype=float)
    total = arr.sum(axis=axis, keepdims=True) + 1e-300
    return arr / total


def _build_hmm_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create market-regime features from price only.

    The prior HMM used only 1-day returns. That can cause the highest-return
    state to become a rare one-day spike bucket instead of a persistent bull
    regime. These features are designed to capture trend, momentum, and risk.
    """
    close = df["Close"].dropna().astype(float)
    daily_return = close.pct_change() * 100
    features = pd.DataFrame(index=close.index)
    features["Return_1D"] = daily_return
    features["Return_5D"] = close.pct_change(5) * 100
    features["Return_20D"] = close.pct_change(20) * 100
    features["Volatility_20D"] = daily_return.rolling(20).std()
    features["Price_vs_SMA50"] = (close / close.rolling(50).mean() - 1) * 100
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def _standardize_features(features: pd.DataFrame):
    means = features.mean()
    stds = features.std(ddof=0).replace(0, np.nan).fillna(1.0)
    z = (features - means) / stds
    return z.astype(float), means, stds


def _log_gaussian_diag_pdf(x, means, vars_):
    """Log density for a diagonal-covariance Gaussian emission model."""
    vars_ = np.maximum(np.asarray(vars_, dtype=float), 1e-6)
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    n_features = x.shape[1]
    out = np.zeros((x.shape[0], means.shape[0]))
    constant = n_features * np.log(2 * np.pi)
    for j in range(means.shape[0]):
        diff = x - means[j]
        out[:, j] = -0.5 * (constant + np.log(vars_[j]).sum() + ((diff * diff) / vars_[j]).sum(axis=1))
    return out


def _forward_backward_diag(x, pi, trans, means, vars_):
    n = x.shape[0]
    k = len(pi)
    log_emissions = _log_gaussian_diag_pdf(x, means, vars_)
    log_emissions = np.clip(log_emissions, -745, 700)
    emissions = np.exp(log_emissions) + 1e-300

    alpha = np.zeros((n, k))
    scales = np.ones(n)
    alpha[0] = pi * emissions[0]
    scales[0] = alpha[0].sum() + 1e-300
    alpha[0] /= scales[0]

    for t in range(1, n):
        alpha[t] = emissions[t] * (alpha[t - 1] @ trans)
        scales[t] = alpha[t].sum() + 1e-300
        alpha[t] /= scales[t]

    beta = np.zeros((n, k))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = trans @ (emissions[t + 1] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma = _safe_softmax_normalize(gamma, axis=1)

    xi_sum = np.zeros((k, k))
    for t in range(n - 1):
        xi = alpha[t][:, None] * trans * emissions[t + 1][None, :] * beta[t + 1][None, :]
        xi_sum += xi / (xi.sum() + 1e-300)

    log_likelihood = float(np.log(scales + 1e-300).sum())
    return gamma, xi_sum, log_likelihood


def fit_gaussian_hmm_diag(x, n_states=3, max_iter=120, tol=1e-5, random_seed=7):
    """Fit a small diagonal-covariance Gaussian HMM without external packages."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 120:
        raise ValueError("At least 120 daily observations are needed for multifeature HMM state analysis.")

    rng = np.random.default_rng(random_seed)
    trend_proxy = x[:, 2] + 0.5 * x[:, 1] + 0.35 * x[:, 4] - 0.20 * x[:, 3]
    qs = np.linspace(0.18, 0.82, n_states)
    centers = np.quantile(trend_proxy, qs)
    means = np.zeros((n_states, x.shape[1]))
    for j, center in enumerate(centers):
        idx = np.argsort(np.abs(trend_proxy - center))[: max(20, min(80, len(x) // 8))]
        means[j] = x[idx].mean(axis=0)
    means += rng.normal(0, 0.02, size=means.shape)

    base_var = np.var(x, axis=0) + 1e-3
    vars_ = np.tile(base_var, (n_states, 1))
    pi = np.full(n_states, 1.0 / n_states)
    trans = np.full((n_states, n_states), 0.06 / max(n_states - 1, 1))
    np.fill_diagonal(trans, 0.94)

    prev_ll = -np.inf
    for _ in range(max_iter):
        gamma, xi_sum, ll = _forward_backward_diag(x, pi, trans, means, vars_)
        weights = gamma.sum(axis=0) + 1e-9
        pi = gamma[0] + 1e-6
        pi /= pi.sum()
        trans = xi_sum + 1e-5
        trans /= trans.sum(axis=1, keepdims=True)
        means = (gamma.T @ x) / weights[:, None]
        for j in range(n_states):
            diff = x - means[j]
            vars_[j] = (gamma[:, j][:, None] * diff * diff).sum(axis=0) / weights[j]
        vars_ = np.maximum(vars_, 1e-5)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    gamma, _, ll = _forward_backward_diag(x, pi, trans, means, vars_)
    return pi, trans, means, vars_, gamma, ll


def _state_run_lengths(state_labels: pd.Series, include_current=True):
    run_lengths = {state: [] for state in STATE_ORDER}
    if state_labels.empty:
        return run_lengths
    previous_state = None
    run_length = 0
    labels = list(state_labels)
    for idx, state in enumerate(labels):
        if state == previous_state:
            run_length += 1
        else:
            if previous_state is not None:
                run_lengths[previous_state].append(run_length)
            previous_state = state
            run_length = 1
    if previous_state is not None and include_current:
        run_lengths[previous_state].append(run_length)
    return run_lengths


def _empirical_transition_matrix(state_labels: pd.Series):
    counts = pd.DataFrame(0, index=STATE_ORDER, columns=STATE_ORDER, dtype=float)
    labels = list(state_labels)
    for current_state, next_state in zip(labels[:-1], labels[1:]):
        if current_state in STATE_ORDER and next_state in STATE_ORDER:
            counts.loc[current_state, next_state] += 1
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100
    return counts, probs


def _age_conditioned_transition_matrix(state_labels: pd.Series, current_state: str, current_age_percentile: float):
    """Empirical next-day transitions for the current state at similarly old ages.

    Buckets are selected by run-age percentile within completed historical runs
    for the same state. This answers: when this state was this mature before,
    what happened the next day?
    """
    if current_state not in STATE_ORDER or pd.isna(current_age_percentile):
        return pd.DataFrame(), ""

    if current_age_percentile >= 75:
        bucket_low, bucket_high, label = 75, 100, "75th-100th age percentile"
    elif current_age_percentile >= 25:
        bucket_low, bucket_high, label = 25, 75, "25th-75th age percentile"
    else:
        bucket_low, bucket_high, label = 0, 25, "0-25th age percentile"

    completed_runs = _state_run_lengths(state_labels, include_current=False).get(current_state, [])
    completed_runs = np.asarray(completed_runs, dtype=float)
    if completed_runs.size < 5:
        return pd.DataFrame(), label

    rows = []
    labels_list = list(state_labels)
    run_state = None
    run_age = 0
    for i in range(len(labels_list) - 1):
        state = labels_list[i]
        if state == run_state:
            run_age += 1
        else:
            run_state = state
            run_age = 1
        if state != current_state:
            continue
        age_pct = (completed_runs <= run_age).mean() * 100
        if bucket_low <= age_pct <= bucket_high:
            rows.append(labels_list[i + 1])

    if not rows:
        return pd.DataFrame(), label

    probs = pd.Series(rows).value_counts(normalize=True).reindex(STATE_ORDER).fillna(0.0) * 100
    return pd.DataFrame({"To State": STATE_ORDER, "Age-Conditioned Probability": [probs[s] for s in STATE_ORDER]}), label


def compute_hmm_state_analysis(df: pd.DataFrame) -> dict:
    features = _build_hmm_feature_frame(df)
    if len(features) < 120:
        return {
            "transition_table": pd.DataFrame(),
            "model_transition_table": pd.DataFrame(),
            "age_conditioned_table": pd.DataFrame(),
            "age_conditioned_label": "",
            "latest_state": "N/A",
            "latest_date": None,
            "current_state_age": np.nan,
            "current_state_age_percentile": np.nan,
            "state_history": pd.DataFrame(),
            "state_summary": pd.DataFrame(),
        }

    z, feature_means, feature_stds = _standardize_features(features)
    pi, trans, means_z, vars_z, gamma, _ = fit_gaussian_hmm_diag(z.values, n_states=3)

    # Label states by economic regime score, not daily return alone.
    means_original = pd.DataFrame(means_z, columns=features.columns) * feature_stds + feature_means
    regime_score = (
        0.20 * means_original["Return_1D"]
        + 0.35 * means_original["Return_5D"]
        + 0.70 * means_original["Return_20D"]
        + 0.45 * means_original["Price_vs_SMA50"]
        - 0.20 * means_original["Volatility_20D"]
    )
    order = np.argsort(regime_score.values)
    label_for_model_state = {int(order[0]): "Bear", int(order[1]): "Sideways", int(order[2]): "Bull"}
    ordered_model_states = [int(order[0]), int(order[1]), int(order[2])]
    ordered_model_trans = trans[np.ix_(ordered_model_states, ordered_model_states)] * 100

    state_ids = gamma.argmax(axis=1)
    state_labels = pd.Series([label_for_model_state[int(i)] for i in state_ids], index=features.index, name="State")
    state_history = features.copy()
    state_history["State"] = state_labels

    latest_state = str(state_labels.iloc[-1])
    latest_date = state_labels.index[-1]
    current_state_age = 1
    for prev_state in reversed(state_labels.iloc[:-1].tolist()):
        if prev_state == latest_state:
            current_state_age += 1
        else:
            break

    # Percentile is based on completed historical runs only, so the current
    # unfinished run does not rank itself.
    completed_run_lengths = _state_run_lengths(state_labels, include_current=False)
    same_state_completed_runs = np.asarray(completed_run_lengths.get(latest_state, []), dtype=float)
    if same_state_completed_runs.size:
        current_state_age_percentile = float((same_state_completed_runs <= current_state_age).mean() * 100)
    else:
        current_state_age_percentile = np.nan

    empirical_counts, empirical_probs = _empirical_transition_matrix(state_labels)
    transition_table = empirical_probs.reset_index().rename(columns={"index": "From State"}).melt(
        id_vars="From State", var_name="To State", value_name="Daily Transition Probability"
    )

    model_transition_table = pd.DataFrame(ordered_model_trans, index=STATE_ORDER, columns=STATE_ORDER).reset_index().rename(columns={"index": "From State"}).melt(
        id_vars="From State", var_name="To State", value_name="Model-Implied Probability"
    )

    age_conditioned_table, age_conditioned_label = _age_conditioned_transition_matrix(
        state_labels, latest_state, current_state_age_percentile
    )

    state_summary = pd.DataFrame({"State": STATE_ORDER})
    summary_rows = []
    for state in STATE_ORDER:
        mask = state_labels == state
        state_features = features.loc[mask]
        completed = np.asarray(completed_run_lengths.get(state, []), dtype=float)
        summary_rows.append(
            {
                "State": state,
                "Avg 1D Return %": state_features["Return_1D"].mean() if not state_features.empty else np.nan,
                "Avg 20D Return %": state_features["Return_20D"].mean() if not state_features.empty else np.nan,
                "Avg 20D Volatility %": state_features["Volatility_20D"].mean() if not state_features.empty else np.nan,
                "Avg Price vs SMA50 %": state_features["Price_vs_SMA50"].mean() if not state_features.empty else np.nan,
                "Days Classified": int(mask.sum()),
                "Avg Completed Run": completed.mean() if completed.size else np.nan,
            }
        )
    state_summary = pd.DataFrame(summary_rows)

    return {
        "transition_table": transition_table,
        "model_transition_table": model_transition_table,
        "age_conditioned_table": age_conditioned_table,
        "age_conditioned_label": age_conditioned_label,
        "latest_state": latest_state,
        "latest_date": latest_date,
        "current_state_age": int(current_state_age),
        "current_state_age_percentile": current_state_age_percentile,
        "state_history": state_history,
        "state_summary": state_summary,
    }




def info_tooltip(label: str, tooltip: str) -> str:
    """Return an inline help control that works with desktop hover and iPad tap.

    Streamlit/iPad Safari can be unreliable with pure CSS hover tooltips inside
    dataframe-heavy layouts. A native details/summary control gives us a visible
    info icon, a hover title, and a tap-to-open explanation without JavaScript.
    """
    safe_label = html.escape(str(label))
    safe_title = html.escape(" ".join(str(tooltip).split()))
    paragraphs = [part.strip() for part in str(tooltip).split("\n\n") if part.strip()]
    safe_paragraphs = "".join(
        f"<p>{html.escape(paragraph).replace(chr(10), '<br>')}</p>"
        for paragraph in paragraphs
    )
    return (
        f'<details class="info-tooltip-details">'
        f'<summary title="{safe_title}"><span class="info-tooltip-label">{safe_label}</span> '
        f'<span class="info-tooltip-icon" aria-label="More information">ⓘ</span></summary>'
        f'<div class="info-tooltip-panel">{safe_paragraphs}</div>'
        f'</details>'
    )


AGE_CONDITIONED_TOOLTIP = """Shows what happened next historically when the current regime reached a similar age.

The app measures how old the current regime is relative to completed historical runs of the same state and places it into an age bucket: 0-25th percentile = young regime, 25-75th percentile = mid-life regime, and 75-100th percentile = mature regime.

It then examines all historical occurrences of the same regime at similar ages and calculates the probability of transitioning to Bear, Sideways, or Bull on the next trading day.

Example: if the current Bull regime is in the 75th-100th age percentile, the table answers: when Bull regimes were this mature in the past, what state did they enter the next day?

This is an empirical historical calculation, not a forecast."""


REGIME_CHARACTERISTICS_TOOLTIP = """These statistics describe the historical behavior of each HMM regime after classification.

Avg 1D Return: average daily return while in the regime.
Avg 20D Return: average rolling 20-trading-day return during the regime.
Avg 20D Volatility: average 20-day realized volatility.
Avg Price vs SMA50: average distance from the 50-day moving average.
Days Classified: number of historical trading days assigned to the regime.
Avg Completed Run: average duration, in trading days, before the regime transitioned to a different state.

The highlighted row is the current regime."""

def render_hmm_state_section(profile, key_prefix="main"):
    st.markdown('<div class="section-title">Hidden Markov Model Regime Probabilities ⓘ</div>', unsafe_allow_html=True)
    analysis = compute_hmm_state_analysis(profile["df"])
    transition_table = analysis.get("transition_table", pd.DataFrame())

    latest_state = analysis.get("latest_state", "N/A")
    latest_date = analysis.get("latest_date")
    age = analysis.get("current_state_age", np.nan)
    percentile = analysis.get("current_state_age_percentile", np.nan)
    state_color = STATE_COLORS.get(latest_state, "#111936")
    latest_date_label = latest_date.strftime("%Y-%m-%d") if latest_date is not None else "N/A"
    age_label = "N/A" if pd.isna(age) else f"{int(age):,} trading days"
    percentile_label = "N/A" if pd.isna(percentile) else f"{float(percentile):.1f}%"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="summary-card"><div class="summary-label">Most Recent Trading Day</div><div class="summary-value">{latest_date_label}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="summary-card"><div class="summary-label">Current State</div><div class="summary-value" style="color:{state_color};">{latest_state}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="summary-card"><div class="summary-label">Current State Age</div><div class="summary-value">{age_label}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="summary-card"><div class="summary-label">Age Percentile</div><div class="summary-value">{percentile_label}</div></div>', unsafe_allow_html=True)

    if transition_table.empty:
        st.info("Not enough historical data for HMM regime probabilities.")
        return

    pivot = transition_table.pivot(index="From State", columns="To State", values="Daily Transition Probability").reindex(index=STATE_ORDER, columns=STATE_ORDER)
    styled = pivot.style.format("{:.2f}%").map(
        lambda v: "font-weight: 850;" if isinstance(v, (int, float, np.number)) else ""
    )
    st.markdown("**Empirical next-day transitions from inferred HMM regimes**")
    st.dataframe(styled, use_container_width=True, height=180)
    st.caption(
        "Rows are the inferred regime on day T and columns are the inferred regime on day T+1. "
        "Regimes are inferred with a 3-state Gaussian HMM using 1D return, 5D return, 20D return, 20D volatility, and price versus 50D moving average. "
        "States are labeled by a trend/momentum/volatility regime score, not by one-day return alone."
    )

    age_conditioned = analysis.get("age_conditioned_table", pd.DataFrame())
    age_conditioned_label = analysis.get("age_conditioned_label", "")
    if not age_conditioned.empty:
        ac_pivot = age_conditioned.set_index("To State").T.reindex(columns=STATE_ORDER)
        ac_styled = ac_pivot.style.format("{:.2f}%").map(
            lambda v: "font-weight: 850;" if isinstance(v, (int, float, np.number)) else ""
        )
        st.markdown(f'<div class="section-title">{info_tooltip(f"Transition probabilities - current {latest_state} regime ({age_conditioned_label})", AGE_CONDITIONED_TOOLTIP)}</div>', unsafe_allow_html=True)
        st.dataframe(ac_styled, use_container_width=True, height=90)
        st.caption("This is an empirical historical calculation, not a forecast. It reflects how similar-aged regimes behaved in the past.")

    model_transition_table = analysis.get("model_transition_table", pd.DataFrame())
    if not model_transition_table.empty:
        with st.expander("Show model-implied HMM transition matrix"):
            model_pivot = model_transition_table.pivot(index="From State", columns="To State", values="Model-Implied Probability").reindex(index=STATE_ORDER, columns=STATE_ORDER)
            st.dataframe(model_pivot.style.format("{:.2f}%"), use_container_width=True, height=180)
            st.caption("This is the fitted HMM transition matrix after state relabeling. The empirical table above is often easier to interpret because it counts transitions from the final inferred historical state path.")

    state_summary = analysis.get("state_summary", pd.DataFrame())
    if not state_summary.empty:
        st.markdown(
            f'<div class="section-title">{info_tooltip("HMM Regime Characteristics (historical averages)", REGIME_CHARACTERISTICS_TOOLTIP)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hmm-current-state-note">Highlighted row = current inferred regime: <strong>{html.escape(str(latest_state))}</strong>.</div>',
            unsafe_allow_html=True,
        )

        def _highlight_current_state(row):
            if str(row.get("State", "")) == str(latest_state):
                return [
                    "background-color: #fff0b8; color: #111936; font-weight: 850; border-top: 2px solid #d8a500; border-bottom: 2px solid #d8a500;"
                    for _ in row
                ]
            return ["" for _ in row]

        summary_styled = state_summary.style.apply(_highlight_current_state, axis=1).format(
            {
                "Avg 1D Return %": "{:+.3f}%",
                "Avg 20D Return %": "{:+.2f}%",
                "Avg 20D Volatility %": "{:.3f}%",
                "Avg Price vs SMA50 %": "{:+.2f}%",
                "Days Classified": "{:,.0f}",
                "Avg Completed Run": "{:.1f}",
            }
        )
        st.dataframe(summary_styled, use_container_width=True, hide_index=True, height=175)

def scan_similar_setups(
    df: pd.DataFrame,
    compression_tolerance_ratio: float,
    selected_filters: List[str],
    clv_tolerance: float,
    volume_tolerance_pct: float,
    rs_tolerance_pp: float,
) -> pd.DataFrame:
    """Find historical analogs by direct +/- tolerance matching.

    The current setup is always the latest available trading day in the profile.
    Historical candidates must have a known 5-trading-day forward outcome, but
    the current setup itself does not need a future outcome. This keeps the
    summary table's Current Setup and the analog-matching anchor aligned.
    """
    current_required = ["Compression_Ratio", "CLV_Trend", "Volume_Ratio"]
    if "Relative strength" in selected_filters:
        current_required.append("RS_20")

    current_pool = df.dropna(subset=current_required).copy()
    if current_pool.empty:
        return pd.DataFrame()

    current = current_pool.iloc[-1]

    candidate_required = [
        "Compression_Ratio",
        "CLV_Trend",
        "Volume_Ratio",
        "Future_Close_5D",
        "Forward_Return_5D",
    ]
    if "Relative strength" in selected_filters:
        candidate_required.append("RS_20")

    candidates = df[df.index < current.name].dropna(subset=candidate_required).copy()
    if candidates.empty:
        return pd.DataFrame()

    # Compression matching is a simple direct tolerance band around the latest
    # setup: current compression +/- the sidebar slider value.
    compression_low = current["Compression_Ratio"] - compression_tolerance_ratio
    compression_high = current["Compression_Ratio"] + compression_tolerance_ratio
    mask = candidates["Compression_Ratio"].between(compression_low, compression_high)

    if "CLV trend" in selected_filters:
        clv_low = current["CLV_Trend"] - clv_tolerance
        clv_high = current["CLV_Trend"] + clv_tolerance
        mask &= candidates["CLV_Trend"].between(clv_low, clv_high)

    if "Volume support" in selected_filters:
        vol_tol = volume_tolerance_pct / 100
        volume_low = current["Volume_Ratio"] * (1 - vol_tol)
        volume_high = current["Volume_Ratio"] * (1 + vol_tol)
        mask &= candidates["Volume_Ratio"].between(volume_low, volume_high)

    if "Relative strength" in selected_filters:
        rs_low = current["RS_20"] - rs_tolerance_pp
        rs_high = current["RS_20"] + rs_tolerance_pp
        mask &= candidates["RS_20"].between(rs_low, rs_high)

    out = candidates.loc[mask].copy()
    if out.empty:
        return pd.DataFrame()

    out = out.sort_index()
    out = out.reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    out["Dollar_Change_5D"] = out["Future_Close_5D"] - out["Close"]

    out["Close"] = out["Close"].round(2)
    out["Future_Close_5D"] = out["Future_Close_5D"].round(2)
    out["Dollar_Change_5D"] = out["Dollar_Change_5D"].round(2)
    out["Forward_Return_5D"] = out["Forward_Return_5D"].round(2)
    out["Compression_Ratio"] = out["Compression_Ratio"].round(2)
    out["CLV_Trend"] = out["CLV_Trend"].round(2)
    out["Volume_Ratio"] = out["Volume_Ratio"].round(2)
    out["RS_20"] = out["RS_20"].round(1)

    extra_rs_cols = [c for c in out.columns if c.startswith("RS_20_") and c != "RS_20"]
    for col in extra_rs_cols:
        out[col] = out[col].round(1)

    base_cols = [
        "Date",
        "Close",
        "Future_Close_5D",
        "Dollar_Change_5D",
        "Forward_Return_5D",
        "Compression_Ratio",
        "CLV_Trend",
        "RS_20",
    ]
    return out[base_cols + extra_rs_cols + ["Volume_Ratio"]]


def build_profile(
    ticker: str,
    benchmark_dfs: dict,
    period: str,
    compression_tolerance_ratio: float,
    selected_filters: List[str],
    clv_tolerance: float,
    volume_tolerance_pct: float,
    rs_tolerance_pp: float,
    rs_col_map: dict,
):
    raw = download_ohlcv(ticker, period)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if len(raw) < 320:
        raise ValueError(f"{ticker} needs at least ~320 daily bars.")

    df = add_profile_columns(raw, benchmark_dfs)
    df = df.dropna(subset=["Compression_Ratio", "CLV_Trend", "Volume_Ratio"])

    ticker_upper = ticker.upper()

    live_quote = get_live_quote(ticker_upper)
    if not isinstance(live_quote, dict):
        live_quote = {"last_price": np.nan, "previous_close": np.nan}
    if pd.isna(live_quote.get("previous_close", np.nan)) and not raw.empty:
        live_quote["previous_close"] = float(raw.iloc[-1]["Close"])

    return {
        "ticker": ticker_upper,
        "stock_name": get_ticker_display_name(ticker_upper),
        "live_quote": live_quote,
        "df": df,
        "rs_col_map": rs_col_map,
        "analogs": scan_similar_setups(
            df,
            compression_tolerance_ratio,
            selected_filters,
            clv_tolerance,
            volume_tolerance_pct,
            rs_tolerance_pp,
        ),
    }

def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --app-offwhite: #fffaf0;
            --app-offwhite-soft: #fffdf7;
            --app-offwhite-head: #f6efe1;
            --app-border: #d8d0bf;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            background-color: var(--app-offwhite-soft) !important;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2.35rem !important;
            padding-bottom: 2rem;
            overflow: visible !important;
        }
        h1 {
            color: #111936;
            font-size: 2.45rem !important;
            line-height: 1.0 !important;
            margin-bottom: 0.2rem !important;
            font-weight: 850 !important;
        }
        div[data-testid="stCaptionContainer"] {
            color: #35405f;
            font-size: 0.96rem;
        }
        div[data-testid="stTabs"] button {
            color: red !important;
            font-weight: 800;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 2px solid red;
        }
        .red {
            color: red !important;
        }
        .green {
            color: green !important;
        }
        .outcome-card {
            border: 1px solid #d8d0bf;
            border-radius: 10px;
            padding: 18px 14px;
            text-align: center;
            min-height: 150px;
        }
        .outcome-title {
            color: #0f1730;
            font-size: 0.98rem;
            margin-bottom: 8px;
            font-weight: 800;
        }
        .outcome-number {
            font-size: 1.35rem;
            font-weight: 900;
            margin-bottom: 16px;
        }
        .divider {
            border-top: 1px solid #d5dbe7;
            margin: 4px 0 12px;
        }
        .section-title {
            color: #111936;
            font-size: 1.35rem;
            font-weight: 850;
            margin: 0 0 8px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #dfe5ef;
            border-radius: 8px;
            overflow: hidden;
        }
        .summary-card {
            border: 1px solid #dfe5ef;
            border-radius: 10px;
            background: var(--app-offwhite-soft);
            padding: 9px 8px 12px;
            min-height: 92px;
            overflow: hidden;
            text-align: left;
        }
        .summary-label {
            color: #33405f;
            font-size: clamp(0.78rem, 0.95vw, 0.92rem);
            line-height: 1.05;
            font-weight: 850;
            white-space: normal;
            overflow-wrap: normal;
            word-break: normal;
            min-height: 2.1rem;
            max-height: 2.25rem;
            margin-bottom: 8px;
        }
        .summary-value {
            color: #111936;
            font-size: clamp(1.22rem, 1.7vw, 1.72rem);
            line-height: 1.05;
            font-weight: 950;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .summary-range-value {
            font-size: clamp(1.02rem, 1.35vw, 1.42rem);
            line-height: 1.08;
            font-weight: 950;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .tab-control-card {
            border: 1px solid #dfe5ef;
            border-radius: 10px;
            padding: 12px 12px 4px;
            margin-bottom: 10px;
            background: var(--app-offwhite-soft);
        }

        /* iPad/Safari header/sidebar visibility fixes */
        header[data-testid="stHeader"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            height: 3.25rem !important;
            min-height: 3.25rem !important;
            z-index: 999999 !important;
            background: var(--app-offwhite-soft) !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[kind="header"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 1000000 !important;
            color: #111936 !important;
            background: rgba(255, 253, 247, 0.96) !important;
            border-radius: 8px !important;
        }
        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapsedControl"] svg,
        button[kind="header"] svg {
            fill: #111936 !important;
            stroke: #111936 !important;
        }
        .app-title-wrap {
            position: relative;
            z-index: 5;
            margin: 0.35rem 0 1.15rem 0;
            padding-top: 0.85rem;
            overflow: visible !important;
        }
        .app-title {
            color: #111936;
            font-size: clamp(2.05rem, 4.8vw, 2.45rem);
            line-height: 1.16;
            margin: 0 0 0.25rem 0;
            padding: 0.08rem 0;
            font-weight: 850;
            overflow: visible !important;
        }
        .app-subtitle {
            color: #35405f;
            font-size: 0.96rem;
            line-height: 1.25;
            margin: 0;
            font-weight: 500;
        }
        .info-tooltip-details {
            display: inline-block;
            position: relative;
            max-width: 100%;
            vertical-align: baseline;
        }
        .info-tooltip-details summary {
            list-style: none;
            cursor: help;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            color: #111936;
            outline: none;
        }
        .info-tooltip-details summary::-webkit-details-marker {
            display: none;
        }
        .info-tooltip-label {
            border-bottom: 1px dotted #35405f;
        }
        .info-tooltip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.05em;
            height: 1.05em;
            border: 1.5px solid #111936;
            border-radius: 999px;
            font-size: 0.86em;
            line-height: 1;
            font-weight: 850;
            flex: 0 0 auto;
        }
        .info-tooltip-panel {
            position: absolute;
            left: 0;
            top: calc(100% + 8px);
            width: min(560px, 82vw);
            max-width: 560px;
            z-index: 1000001;
            background: #111936;
            color: #fffdf7;
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 0.88rem;
            line-height: 1.35;
            font-weight: 500;
            box-shadow: 0 8px 24px rgba(17, 25, 54, 0.24);
            white-space: normal;
        }
        .info-tooltip-panel p {
            margin: 0 0 0.55rem 0;
        }
        .info-tooltip-panel p:last-child {
            margin-bottom: 0;
        }
        .info-tooltip-details:not([open]):hover .info-tooltip-panel {
            display: block;
        }
        .info-tooltip-details:not([open]) .info-tooltip-panel {
            display: none;
        }
        .hmm-current-state-note {
            color: #35405f;
            font-size: 0.92rem;
            margin: -2px 0 8px 0;
        }
        @media (hover: none) and (pointer: coarse) {
            .block-container {
                padding-top: 2.75rem !important;
            }
            .app-title-wrap {
                padding-top: 1.1rem;
                margin-top: 0.35rem;
            }
            .app-title {
                line-height: 1.2;
            }
            .info-tooltip-panel {
                position: fixed;
                left: 16px;
                right: 16px;
                top: 74px;
                width: auto;
                max-width: none;
            }

            /* Mobile sidebar handle: force visible contrast and tap target. */
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapsedControl"],
            button[kind="header"] {
                position: fixed !important;
                top: 0.75rem !important;
                left: 0.75rem !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 44px !important;
                height: 44px !important;
                min-width: 44px !important;
                min-height: 44px !important;
                visibility: visible !important;
                opacity: 1 !important;
                background: #111936 !important;
                color: #fffdf7 !important;
                border: 2px solid #fffdf7 !important;
                border-radius: 999px !important;
                box-shadow: 0 4px 14px rgba(17, 25, 54, 0.35) !important;
                z-index: 2147483647 !important;
            }

            [data-testid="collapsedControl"] svg,
            [data-testid="stSidebarCollapsedControl"] svg,
            button[kind="header"] svg {
                fill: #fffdf7 !important;
                stroke: #fffdf7 !important;
                color: #fffdf7 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def build_analog_chart(profile):
    analogs = profile["analogs"]
    ticker = profile["ticker"]
    stock_name = str(profile.get("stock_name") or "").strip()
    title_name = f"{ticker} ({stock_name})" if stock_name and stock_name.upper() != ticker.upper() else ticker
    live_quote = profile.get("live_quote", {}) or {}
    live_price = live_quote.get("last_price", np.nan)
    previous_close = live_quote.get("previous_close", np.nan)
    live_price_label = money2(live_price) if live_price is not None and pd.notna(live_price) else "N/A"
    live_change = live_price - previous_close if pd.notna(live_price) and pd.notna(previous_close) else np.nan
    live_change_label = money2(live_change) if pd.notna(live_change) else "N/A"
    live_change_color = "green" if pd.notna(live_change) and live_change > 0 else "red" if pd.notna(live_change) and live_change < 0 else "#111936"
    latest_close = float(profile["df"].iloc[-1]["Close"])

    fig = plt.figure(figsize=(18.0, 5.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.25, 6.75], wspace=0.22)
    legend_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    legend_ax.set_axis_off()

    if analogs.empty:
        ax.text(
            0.5,
            0.5,
            "No historical analogs found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
        )
        ax.set_axis_off()
        return fig

    chart_df = analogs.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    chart_df = chart_df.sort_values("Date")

    ax.plot(chart_df["Date"], chart_df["Future_Close_5D"], linewidth=1.4, alpha=0.65)
    ax.scatter(
        chart_df["Date"],
        chart_df["Future_Close_5D"],
        s=78,
        label="Close 5 trading days later",
        zorder=3,
    )

    best = chart_df.loc[chart_df["Dollar_Change_5D"].idxmax()]
    worst = chart_df.loc[chart_df["Dollar_Change_5D"].idxmin()]

    ax.scatter(
        [pd.to_datetime(best["Date"])],
        [best["Future_Close_5D"]],
        s=175,
        color="green",
        label="Biggest advance",
        zorder=4,
    )
    ax.scatter(
        [pd.to_datetime(worst["Date"])],
        [worst["Future_Close_5D"]],
        s=175,
        color="red",
        label="Biggest decline",
        zorder=4,
    )

    min_y = min(chart_df["Future_Close_5D"].min(), latest_close)
    max_y = max(chart_df["Future_Close_5D"].max(), latest_close)
    price_span = max(max_y - min_y, 1)

    min_date = chart_df["Date"].min()
    max_date = chart_df["Date"].max()
    date_span_days = max((max_date - min_date).days, 30)
    x_padding_days = max(int(date_span_days * 0.035), 10)
    label_x = max_date + pd.Timedelta(days=x_padding_days)

    ax.axhline(
        latest_close,
        linestyle="--",
        linewidth=1.6,
        color="#2166ff",
        label=f"Current close {money0(latest_close)}",
    )

    label_items = [
        {
            "y": latest_close,
            "text": f"Current close {money0(latest_close)}",
            "color": "#2166ff",
            "ha": "left",
            "x": label_x,
            "anchor_x": max_date,
        }
    ]

    # Prevent labels from sitting on top of one another.
    # Labels are assigned display y-positions at least one label-height apart;
    # when moved, leader lines point back to the exact horizontal level.
    sorted_items = sorted(label_items, key=lambda item: item["y"])
    min_gap = price_span * 0.065
    for idx in range(1, len(sorted_items)):
        prev_y = sorted_items[idx - 1].get("display_y", sorted_items[idx - 1]["y"])
        this_y = sorted_items[idx]["y"]
        if this_y - prev_y < min_gap:
            sorted_items[idx]["display_y"] = prev_y + min_gap
    for idx in range(len(sorted_items) - 2, -1, -1):
        next_y = sorted_items[idx + 1].get("display_y", sorted_items[idx + 1]["y"])
        this_y = sorted_items[idx].get("display_y", sorted_items[idx]["y"])
        if next_y - this_y < min_gap:
            sorted_items[idx]["display_y"] = next_y - min_gap

    for item in label_items:
        display_y = item.get("display_y", item["y"])
        moved = abs(display_y - item["y"]) > price_span * 0.01
        arrowprops = None
        if moved:
            arrowprops = {
                "arrowstyle": "-",
                "color": item["color"],
                "lw": 1.1,
                "alpha": 0.85,
                "shrinkA": 0,
                "shrinkB": 0,
            }
        ax.annotate(
            item["text"],
            xy=(item.get("anchor_x", item["x"]), item["y"]),
            xytext=(item["x"], display_y),
            textcoords="data",
            color=item["color"],
            ha=item["ha"],
            va="center",
            fontsize=18,
            fontweight="bold",
            arrowprops=arrowprops,
            zorder=5,
            clip_on=False,
        )

    label_y_values = [item.get("display_y", item["y"]) for item in label_items]
    adjusted_min_y = min(min_y, min(label_y_values))
    adjusted_max_y = max(max_y, max(label_y_values))
    adjusted_span = max(adjusted_max_y - adjusted_min_y, 1)
    ax.set_ylim(adjusted_min_y - adjusted_span * 0.08, adjusted_max_y + adjusted_span * 0.08)
    ax.set_title(
        "Historical Similar ATR10/ATR50 Compression Setups — Price 5 Days Later",
        fontsize=20,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Analog Date", fontsize=21, fontweight="bold")
    ax.set_ylabel("Share Price", fontsize=21, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(min_date - pd.Timedelta(days=x_padding_days), label_x + pd.Timedelta(days=x_padding_days))
    ax.grid(True, alpha=0.24)
    handles, labels = ax.get_legend_handles_labels()
    title_fontsize = max(14, min(24, 28 - max(len(title_name) - 12, 0) * 0.35))
    legend_ax.text(
        0.0,
        0.86,
        title_name,
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        color="#111936",
        wrap=True,
        clip_on=False,
    )
    legend_ax.text(
        0.0,
        0.70,
        f"Current price {live_price_label}",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#111936",
        clip_on=False,
    )
    legend_ax.text(
        0.72,
        0.70,
        f"({live_change_label})",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=live_change_color,
        clip_on=False,
    )
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.50),
        frameon=True,
        fontsize=18,
        ncol=1,
        borderaxespad=0.0,
        labelspacing=1.15,
        handlelength=2.2,
        handletextpad=0.75,
    )
    fig.subplots_adjust(left=0.02, right=0.985, top=0.89, bottom=0.17, wspace=0.22)
    return fig

def color_class_for_number(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return ""


def compute_similarity_pct(current_values: pd.Series, target_values: pd.Series, reference_values: pd.DataFrame):
    if target_values is None or target_values.empty or reference_values.empty:
        return np.nan
    std = reference_values.std(ddof=0).replace(0, np.nan)
    std = std.fillna(reference_values.abs().mean()).replace(0, 1.0).fillna(1.0)
    aligned_current = current_values[target_values.index].astype(float)
    aligned_target = target_values.astype(float)
    z_distance = ((aligned_current - aligned_target) / std[target_values.index]).pow(2).mean() ** 0.5
    return float(np.exp(-z_distance) * 100)



def compute_major_events_40d(profile, threshold_pct=10, lookahead_days=40):
    """Identify positive and negative major events independently.

    For each possible event start date, the app looks forward 40 trading days
    and measures both the maximum upside excursion and the maximum downside
    excursion. A positive event is recorded when the max gain is at least the
    threshold. A negative event is recorded when the max loss is at or below the
    negative threshold. Positive and negative event windows are de-duplicated
    separately, so a strong rally no longer suppresses an overlapping selloff.

    Setup metrics are taken from the trading day immediately before the event
    start. This keeps the setup date separate from the future event window.
    """
    df = profile["df"].copy()
    rs_col_map = profile.get("rs_col_map", {}) or {}
    rs_cols = [col for col in rs_col_map.values() if col in df.columns]
    required_cols = ["Compression_Ratio", "CLV_Trend", "RS_20", "Volume_Ratio", "Close", "High", "Low"] + rs_cols
    required_cols = list(dict.fromkeys(required_cols))
    df = df.dropna(subset=required_cols).copy()

    if len(df) <= lookahead_days + 1:
        return pd.DataFrame()

    positive_candidates = []
    negative_candidates = []
    dates = list(df.index)

    def _base_event_row(setup_row, setup_date, event_start, start_close, max_gain_pct, max_loss_pct):
        return {
            "Setup Date": setup_date.date(),
            "Event Start": event_start.date(),
            "Start Price": start_close,
            "Max Gain %": max_gain_pct,
            "Max Loss %": max_loss_pct,
            "Setup_Compression_Ratio": setup_row.get("Compression_Ratio", np.nan),
            "Setup_CLV_Trend": setup_row.get("CLV_Trend", np.nan),
            "Setup_RS_20": setup_row.get("RS_20", np.nan),
            "Setup_Volume_Ratio": setup_row.get("Volume_Ratio", np.nan),
            **{f"Setup_{col}": setup_row.get(col, np.nan) for col in rs_cols},
        }

    for start_pos in range(1, len(df) - lookahead_days):
        setup_pos = start_pos - 1
        event_start = dates[start_pos]
        setup_date = dates[setup_pos]
        start_close = float(df.iloc[start_pos]["Close"])

        if not np.isfinite(start_close) or start_close <= 0:
            continue

        forward_window = df.iloc[start_pos + 1 : start_pos + lookahead_days + 1]
        if len(forward_window) < lookahead_days:
            continue

        max_high = float(forward_window["High"].max())
        min_low = float(forward_window["Low"].min())
        max_high_date = forward_window["High"].idxmax()
        min_low_date = forward_window["Low"].idxmin()

        max_gain_pct = (max_high / start_close - 1) * 100
        max_loss_pct = (min_low / start_close - 1) * 100
        setup_row = df.iloc[setup_pos]
        base = _base_event_row(setup_row, setup_date, event_start, start_close, max_gain_pct, max_loss_pct)

        if max_gain_pct >= threshold_pct:
            positive_candidates.append(
                {
                    **base,
                    "Event End": max_high_date.date(),
                    "Event Type": "Positive",
                    "Days to Event": int(df.index.get_loc(max_high_date) - start_pos),
                    "Event Price": max_high,
                    "Dollar_Change_Event_Window": max_high - start_close,
                    "Percent_Change_Event_Window": max_gain_pct,
                    "window_start_pos": start_pos + 1,
                    "window_end_pos": start_pos + lookahead_days,
                    "abs_event_pct": abs(max_gain_pct),
                }
            )

        if max_loss_pct <= -threshold_pct:
            negative_candidates.append(
                {
                    **base,
                    "Event End": min_low_date.date(),
                    "Event Type": "Negative",
                    "Days to Event": int(df.index.get_loc(min_low_date) - start_pos),
                    "Event Price": min_low,
                    "Dollar_Change_Event_Window": min_low - start_close,
                    "Percent_Change_Event_Window": max_loss_pct,
                    "window_start_pos": start_pos + 1,
                    "window_end_pos": start_pos + lookahead_days,
                    "abs_event_pct": abs(max_loss_pct),
                }
            )

    def _dedupe_overlapping(candidates):
        selected = []
        for candidate in sorted(candidates, key=lambda row: row["abs_event_pct"], reverse=True):
            overlaps = False
            for chosen in selected:
                if not (
                    candidate["window_end_pos"] < chosen["window_start_pos"]
                    or candidate["window_start_pos"] > chosen["window_end_pos"]
                ):
                    overlaps = True
                    break
            if not overlaps:
                selected.append(candidate)
        return selected

    selected = _dedupe_overlapping(positive_candidates) + _dedupe_overlapping(negative_candidates)
    if not selected:
        return pd.DataFrame()

    events = pd.DataFrame(selected).drop(columns=["window_start_pos", "window_end_pos", "abs_event_pct"])
    events = events.sort_values(["Event Start", "Event Type"], ascending=[False, True]).reset_index(drop=True)
    return events

def get_major_event_setup_stats(profile, threshold_pct=10):
    events = compute_major_events_40d(profile, threshold_pct=threshold_pct, lookahead_days=40)
    rs_col_map = profile.get("rs_col_map", {}) or {}

    metric_col_map = {"Compression": "Setup_Compression_Ratio"}
    for benchmark, col in rs_col_map.items():
        metric_col_map[f"RS vs {benchmark} (20D)"] = f"Setup_{col}"
    metric_col_map.update({"CLV": "Setup_CLV_Trend", "Volume Support": "Setup_Volume_Ratio"})

    df = profile["df"].copy()
    source_metric_col_map = {name: col.replace("Setup_", "", 1) for name, col in metric_col_map.items()}
    latest_subset = [c for c in source_metric_col_map.values() if c in df.columns]
    latest_pool = df.dropna(subset=latest_subset) if latest_subset else df
    latest = latest_pool.iloc[-1] if not latest_pool.empty else df.iloc[-1]
    current_vector = pd.Series({name: latest.get(col, np.nan) for name, col in source_metric_col_map.items()})

    event_metric_cols = [col for col in metric_col_map.values() if col in events.columns]
    event_metrics = pd.DataFrame()
    if not events.empty and event_metric_cols:
        reverse_map = {v: k for k, v in metric_col_map.items()}
        event_metrics = events[event_metric_cols].rename(columns=reverse_map)

    def _event_subset_stats(subset):
        if subset.empty:
            return {
                "count": np.nan,
                "metrics": {name: np.nan for name in metric_col_map},
                "avg_change": np.nan,
                "similarity": np.nan,
            }
        valid_cols = [col for col in metric_col_map.values() if col in subset.columns]
        reverse_map = {v: k for k, v in metric_col_map.items()}
        avg = subset[valid_cols].rename(columns=reverse_map).mean()
        return {
            "count": len(subset),
            "metrics": {name: avg.get(name, np.nan) for name in metric_col_map},
            "avg_change": subset["Dollar_Change_Event_Window"].mean(),
            "similarity": compute_similarity_pct(current_vector, avg, event_metrics),
        }

    positive_events = events[events["Event Type"] == "Positive"] if not events.empty else events
    negative_events = events[events["Event Type"] == "Negative"] if not events.empty else events

    return {
        "events": events,
        "event_metric_map": metric_col_map,
        "current_vector": current_vector,
        "event_metrics": event_metrics,
        "positive_stats": _event_subset_stats(positive_events),
        "negative_stats": _event_subset_stats(negative_events),
    }

def render_summary_metrics(profile, key_prefix="main"):
    analogs = profile["analogs"].copy()
    df = profile["df"]
    rs_col_map = profile.get("rs_col_map", {}) or {}

    metric_col_map = {"Compression": "Compression_Ratio"}
    for benchmark, col in rs_col_map.items():
        metric_col_map[f"RS vs {benchmark} (20D)"] = col
    metric_col_map.update({"CLV": "CLV_Trend", "Volume Support": "Volume_Ratio"})

    latest_subset = [c for c in metric_col_map.values() if c in df.columns]
    latest_pool = df.dropna(subset=latest_subset) if latest_subset else df
    latest = latest_pool.iloc[-1] if not latest_pool.empty else df.iloc[-1]

    summary_title_col, summary_slider_col = st.columns([2.65, 1.15])
    with summary_title_col:
        st.markdown('<div class="section-title">Summary of Analogs</div>', unsafe_allow_html=True)
    with summary_slider_col:
        threshold_pct = st.slider(
            "Major event threshold (+/- %)",
            1,
            75,
            10,
            1,
            key=f"{key_prefix}_major_event_threshold",
            help=(
                "Major events use a 40-trading-day look-forward window. Positive and negative events are detected independently: upside events qualify when max gain is at least the threshold, downside events qualify when max loss is at or below the negative threshold. Overlapping positive windows are de-duplicated separately from overlapping negative windows. Setup metrics come from the trading day immediately before the event start."
            ),
        )

    major_event_stats = get_major_event_setup_stats(profile, threshold_pct=threshold_pct)

    columns = [
        "Metric",
        "Current Setup",
        "Avg Positive Setup",
        "Avg Negative Setup",
        "Overall Avg Setup",
        "Major Event Avg Positive Setup",
        "Major Event Avg Negative Setup",
    ]

    header_labels = {
        "Metric": "Metric",
        "Current Setup": "Current<br>Setup",
        "Avg Positive Setup": "Avg<br>Positive<br>Setup",
        "Avg Negative Setup": "Avg<br>Negative<br>Setup",
        "Overall Avg Setup": "Overall<br>Avg<br>Setup",
        "Major Event Avg Positive Setup": "Major Event<br>Avg Positive<br>Setup",
        "Major Event Avg Negative Setup": "Major Event<br>Avg Negative<br>Setup",
    }

    current_vector = pd.Series({name: latest.get(col, np.nan) for name, col in metric_col_map.items()})

    def _blank_if_na(value):
        if value == "" or value is None or pd.isna(value):
            return ""
        return value

    def _fmt_number(value, decimals=2):
        value = _blank_if_na(value)
        if value == "":
            return ""
        return f"{float(value):.{decimals}f}"

    def _fmt_count(value):
        value = _blank_if_na(value)
        if value == "":
            return ""
        return f"{int(float(value)):,}"

    def _fmt_money(value):
        value = _blank_if_na(value)
        if value == "":
            return ""
        return signed_money0(value)

    def _fmt_pct(value):
        value = _blank_if_na(value)
        if value == "":
            return ""
        return f"{float(value):.1f}%"

    def _fmt_range(series):
        if series is None or len(series) == 0:
            return ""
        return f"{signed_money0(series.min())} to {signed_money0(series.max())}"

    def _subset_stats(subset):
        if subset.empty:
            return {
                "count": np.nan,
                "metrics": {name: np.nan for name in metric_col_map},
                "avg_change": np.nan,
                "win_pct": np.nan,
                "range": "",
                "similarity": np.nan,
            }
        available_map = {name: col for name, col in metric_col_map.items() if col in subset.columns}
        avg_vector = pd.Series({name: subset[col].mean() for name, col in available_map.items()})
        reference_values = subset[list(available_map.values())].rename(
            columns={col: name for name, col in available_map.items()}
        )
        return {
            "count": len(subset),
            "metrics": {name: avg_vector.get(name, np.nan) for name in metric_col_map},
            "avg_change": subset["Dollar_Change_5D"].mean(),
            "win_pct": (subset["Dollar_Change_5D"] > 0).mean() * 100,
            "range": _fmt_range(subset["Dollar_Change_5D"]),
            "similarity": compute_similarity_pct(current_vector, avg_vector, reference_values),
        }

    positives = analogs[analogs["Dollar_Change_5D"] > 0] if not analogs.empty else analogs
    negatives = analogs[analogs["Dollar_Change_5D"] <= 0] if not analogs.empty else analogs

    positive_stats = _subset_stats(positives)
    negative_stats = _subset_stats(negatives)
    overall_stats = _subset_stats(analogs)
    major_pos = major_event_stats["positive_stats"]
    major_neg = major_event_stats["negative_stats"]

    raw_rows = [
        {
            "Metric": "Number of Analogs",
            "Current Setup": "",
            "Avg Positive Setup": positive_stats["count"],
            "Avg Negative Setup": negative_stats["count"],
            "Overall Avg Setup": overall_stats["count"],
            "Major Event Avg Positive Setup": major_pos["count"],
            "Major Event Avg Negative Setup": major_neg["count"],
        }
    ]

    for metric_name in metric_col_map:
        raw_rows.append(
            {
                "Metric": metric_name,
                "Current Setup": current_vector.get(metric_name, np.nan),
                "Avg Positive Setup": positive_stats["metrics"].get(metric_name, np.nan),
                "Avg Negative Setup": negative_stats["metrics"].get(metric_name, np.nan),
                "Overall Avg Setup": overall_stats["metrics"].get(metric_name, np.nan),
                "Major Event Avg Positive Setup": major_pos["metrics"].get(metric_name, np.nan),
                "Major Event Avg Negative Setup": major_neg["metrics"].get(metric_name, np.nan),
            }
        )

    raw_rows.extend(
        [
            {
                "Metric": "Avg $ Change",
                "Current Setup": "",
                "Avg Positive Setup": positive_stats["avg_change"],
                "Avg Negative Setup": negative_stats["avg_change"],
                "Overall Avg Setup": overall_stats["avg_change"],
                "Major Event Avg Positive Setup": major_pos["avg_change"],
                "Major Event Avg Negative Setup": major_neg["avg_change"],
            },
            {
                "Metric": "% Win",
                "Current Setup": "",
                "Avg Positive Setup": "",
                "Avg Negative Setup": "",
                "Overall Avg Setup": overall_stats["win_pct"],
                "Major Event Avg Positive Setup": "",
                "Major Event Avg Negative Setup": "",
            },
            {
                "Metric": "$ Change Range",
                "Current Setup": "",
                "Avg Positive Setup": positive_stats["range"],
                "Avg Negative Setup": negative_stats["range"],
                "Overall Avg Setup": overall_stats["range"],
                "Major Event Avg Positive Setup": "",
                "Major Event Avg Negative Setup": "",
            },
            {
                "Metric": "Similarity",
                "Current Setup": "",
                "Avg Positive Setup": positive_stats["similarity"],
                "Avg Negative Setup": negative_stats["similarity"],
                "Overall Avg Setup": overall_stats["similarity"],
                "Major Event Avg Positive Setup": major_pos["similarity"],
                "Major Event Avg Negative Setup": major_neg["similarity"],
            },
        ]
    )

    def _format_cell(metric, value):
        if metric == "Number of Analogs":
            return _fmt_count(value)
        if metric.startswith("RS vs"):
            return _fmt_number(value, 1)
        if metric == "Volume Support":
            formatted = _fmt_number(value, 2)
            return f"{formatted}x" if formatted else ""
        if metric in ["Compression", "CLV"]:
            return _fmt_number(value, 2)
        if metric == "Avg $ Change":
            return _fmt_money(value)
        if metric in ["% Win", "Similarity"]:
            return _fmt_pct(value)
        if metric == "$ Change Range":
            return value if value else ""
        return "" if value is None else str(value)

    positive_cols = {"Avg Positive Setup", "Major Event Avg Positive Setup"}
    negative_cols = {"Avg Negative Setup", "Major Event Avg Negative Setup"}

    def _cell_class(metric, col, raw_value):
        classes = ["analog-summary-cell"]
        if col == "Metric":
            classes.append("analog-summary-metric")
            return " ".join(classes)
        if col in positive_cols:
            classes.append("analog-summary-positive")
        elif col in negative_cols:
            classes.append("analog-summary-negative")
        elif metric == "Avg $ Change" and raw_value not in ["", None] and not pd.isna(raw_value):
            classes.append("analog-summary-positive" if float(raw_value) > 0 else "analog-summary-negative" if float(raw_value) < 0 else "")
        return " ".join([c for c in classes if c])

    header_html = "".join(f'<th>{header_labels[col]}</th>' for col in columns)
    body_rows = []
    for row in raw_rows:
        metric = row["Metric"]
        cells = []
        for col in columns:
            raw_value = row[col]
            value = html.escape(str(metric if col == "Metric" else _format_cell(metric, raw_value)))
            cls = _cell_class(metric, col, raw_value)
            cells.append(f'<td class="{cls}">{value}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <style>
    .analog-summary-wrap {{
        width: 100%;
        margin: 0 0 12px 0;
        overflow: visible;
    }}
    .analog-summary-table {{
        width: 100%;
        table-layout: fixed;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid #dfe5ef;
        border-radius: 10px;
        overflow: hidden;
        font-family: inherit;
        background: #fffaf0;
    }}
    .analog-summary-table th,
    .analog-summary-table td {{
        width: {100 / len(columns):.6f}%;
        border-right: 1px solid #d8d0bf;
        border-bottom: 1px solid #d8d0bf;
        padding: clamp(6px, 0.48vw, 9px) clamp(7px, 0.58vw, 11px);
        text-align: left;
        vertical-align: middle;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
        line-height: 1.04;
    }}
    .analog-summary-table th:last-child,
    .analog-summary-table td:last-child {{
        border-right: none;
    }}
    .analog-summary-table tr:last-child td {{
        border-bottom: none;
    }}
    .analog-summary-table th {{
        background: #f6efe1;
        color: #111936;
        font-weight: 700;
        font-size: clamp(0.90rem, 1.25vw, 1.30rem);
        line-height: 1.03;
        height: auto;
    }}
    .analog-summary-cell {{
        color: #2a2f3a;
        font-weight: 500;
        font-size: clamp(0.95rem, 1.35vw, 1.40rem);
        line-height: 1.05;
    }}
    .analog-summary-metric {{
        color: #2a2f3a;
    }}
    .analog-summary-positive {{
        color: green;
    }}
    .analog-summary-negative {{
        color: red;
    }}
    @media (max-width: 900px) {{
        .analog-summary-table th,
        .analog-summary-table td {{
            padding: 5px 4px;
        }}
        .analog-summary-table th {{
            font-size: 0.68rem;
        }}
        .analog-summary-cell {{
            font-size: 0.70rem;
        }}
    }}
    </style>
    <div class="analog-summary-wrap">
        <table class="analog-summary-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

def build_distribution_chart(profile, bins_count):
    analogs = profile["analogs"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    if analogs.empty:
        ax.text(
            0.5,
            0.5,
            "No distribution available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )
        ax.set_axis_off()
        return fig

    s = analogs["Dollar_Change_5D"].astype(float)
    counts, edges = np.histogram(s, bins=bins_count)

    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        labels.append(f"{lo:,.0f} to {hi:,.0f}")

    y = np.arange(len(labels))
    max_count = int(counts.max()) if len(counts) else 0

    # Scale text relative to bar thickness. Fewer bins allow larger labels;
    # more bins automatically reduce labels enough to avoid overlap.
    bar_height_inches = 5.0 / max(bins_count, 1)
    scaled_size = int(bar_height_inches * 34)
    bar_label_size = max(8, min(15, scaled_size))
    axis_tick_size = max(8, min(14, scaled_size - 1))
    axis_title_size = max(12, min(16, scaled_size + 1))

    ax.barh(y, counts, color="#0d5bd6")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=axis_tick_size)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=axis_title_size, fontweight="bold")
    ax.set_ylabel("Dollar Change (5D)", fontsize=axis_title_size, fontweight="bold")
    ax.tick_params(axis="x", labelsize=axis_tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.grid(axis="x", alpha=0.22)

    label_pad = max(max_count * 0.035, 0.18)
    x_limit = max(max_count + label_pad + max(max_count * 0.12, 1.0), 5)
    ax.set_xlim(0, x_limit)

    for i, count in enumerate(counts):
        label_x = min(int(count) + label_pad, x_limit * 0.96)
        ax.text(
            label_x,
            i,
            f"{int(count)}",
            va="center",
            ha="left",
            fontsize=bar_label_size,
            fontweight="bold",
            clip_on=True,
        )

    fig.tight_layout()
    return fig

def render_distribution(profile, key_prefix="main"):
    analogs = profile["analogs"]
    positives = int((analogs["Dollar_Change_5D"] > 0).sum()) if not analogs.empty else 0
    negatives = int((analogs["Dollar_Change_5D"] <= 0).sum()) if not analogs.empty else 0
    total = max(len(analogs), 1)

    st.markdown('<div class="section-title">Distribution of 5-Day Dollar Change ⓘ</div>', unsafe_allow_html=True)

    top_left, top_right = st.columns([3, 2])
    with top_right:
        bins_count = st.slider("Number of bins", 6, 24, 12, 1, key=f"{key_prefix}_distribution_bins")

    c1, c2 = st.columns([3.4, 1.25])
    with c1:
        fig = build_distribution_chart(profile, bins_count)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        st.markdown(
            f"""
            <div class="outcome-card">
                <div class="outcome-title">Positive outcomes</div>
                <div class="outcome-number green">{positives} ({positives / total * 100:.1f}%)</div>
                <div class="divider"></div>
                <div class="outcome-title">Negative outcomes</div>
                <div class="outcome-number red">{negatives} ({negatives / total * 100:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analogs_table(profile):
    analogs = profile["analogs"].copy()
    if analogs.empty:
        st.info("No similar historical setup was found with the current filters.")
        return

    rs_col_map = profile.get("rs_col_map", {}) or {}
    rs_rename = {col: f"RS vs {benchmark} (20D)" for benchmark, col in rs_col_map.items() if col in analogs.columns}
    analogs = analogs.rename(
        columns={
            "Compression_Ratio": "Compression",
            "RS_20": "Relative Strength (20D)",
            "CLV_Trend": "CLV",
            "Volume_Ratio": "Volume Support",
            **rs_rename,
        }
    )

    format_map = {
        "Close": "{:.2f}",
        "Future_Close_5D": "{:.2f}",
        "Dollar_Change_5D": "{:.2f}",
        "Forward_Return_5D": "{:.2f}%",
        "Compression": "{:.2f}",
        "CLV": "{:.2f}",
        "Relative Strength (20D)": "{:.1f}",
        "Volume Support": "{:.2f}",
    }
    for label in rs_rename.values():
        format_map[label] = "{:.1f}"

    styled = analogs.style.format(format_map).map(
        lambda v: "color: green; font-weight: 800;" if float(v) > 0 else "color: red; font-weight: 800;",
        subset=["Dollar_Change_5D", "Forward_Return_5D"],
    )

    visible_rows = min(len(analogs), 10)
    height = 38 * (visible_rows + 1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=height)
    st.caption(f"Showing all {len(analogs)} entries. Scroll inside the table to view rows beyond the first 10.")


def render_major_events_section(profile, key_prefix="main"):
    st.markdown('<div class="section-title">Major Events ⓘ</div>', unsafe_allow_html=True)

    threshold_pct = st.session_state.get(f"{key_prefix}_major_event_threshold", 10)
    events = compute_major_events_40d(profile, threshold_pct=threshold_pct, lookahead_days=40)
    rs_col_map = profile.get("rs_col_map", {}) or {}
    detail_rs_columns = {f"Setup_{col}": f"RS vs {benchmark} (20D)" for benchmark, col in rs_col_map.items()}

    def _fmt_money(value):
        if value == "" or pd.isna(value):
            return ""
        return signed_money0(value)

    def _fmt_pct(value):
        if value == "" or pd.isna(value):
            return ""
        return signed_pct1(value)

    def _fmt_plain_pct(value):
        if value == "" or pd.isna(value):
            return ""
        return f"{float(value):+.1f}%"

    def _fmt_number(value, decimals=1):
        if value == "" or pd.isna(value):
            return ""
        return f"{float(value):.{decimals}f}"

    def _fmt_volume_support(value):
        if value == "" or pd.isna(value):
            return ""
        return f"{float(value):.2f}x"

    if events.empty:
        display_table = pd.DataFrame(
            columns=[
                "Setup Date",
                "Event Start",
                "Event End",
                "Event Type",
                "Days to Event",
                "$ Change",
                "% Change",
                "Max Gain %",
                "Max Loss %",
                "Compression",
                *detail_rs_columns.values(),
                "CLV",
                "Volume Support",
            ]
        )
    else:
        rename_columns = {
            "Dollar_Change_Event_Window": "$ Change",
            "Percent_Change_Event_Window": "% Change",
            "Setup_Compression_Ratio": "Compression",
            "Setup_CLV_Trend": "CLV",
            "Setup_Volume_Ratio": "Volume Support",
        }
        rename_columns.update(detail_rs_columns)
        display_cols = [
            "Setup Date",
            "Event Start",
            "Event End",
            "Event Type",
            "Days to Event",
            "$ Change",
            "% Change",
            "Max Gain %",
            "Max Loss %",
            "Compression",
            *detail_rs_columns.values(),
            "CLV",
            "Volume Support",
        ]
        display_table = events.rename(columns=rename_columns)
        display_table = display_table[[c for c in display_cols if c in display_table.columns]]

    detailed_styled = display_table.style.format(
        {
            "Days to Event": lambda v: "" if v == "" or pd.isna(v) else f"{int(float(v)):,}",
            "$ Change": _fmt_money,
            "% Change": _fmt_pct,
            "Max Gain %": _fmt_plain_pct,
            "Max Loss %": _fmt_plain_pct,
            "Compression": lambda v: _fmt_number(v, 2),
            **{label: (lambda v: _fmt_number(v, 1)) for label in detail_rs_columns.values()},
            "CLV": lambda v: _fmt_number(v, 2),
            "Volume Support": _fmt_volume_support,
        }
    ).map(
        lambda v: "color: green; font-weight: 800;" if isinstance(v, (int, float, np.number)) and float(v) > 0 else (
            "color: red; font-weight: 800;" if isinstance(v, (int, float, np.number)) and float(v) < 0 else ""
        ),
        subset=["$ Change", "% Change", "Max Gain %", "Max Loss %"],
    ).map(
        lambda v: "color: green; font-weight: 800;" if str(v) == "Positive" else (
            "color: red; font-weight: 800;" if str(v) == "Negative" else ""
        ),
        subset=["Event Type"],
    )

    visible_rows = min(max(len(display_table), 1), 10)
    height = 38 * (visible_rows + 1)
    st.markdown(
        """
        <div class="section-title" title="Major events use a 40-trading-day look-forward window. Positive events qualify when the max gain reaches the slider threshold; negative events qualify when the max loss reaches the negative slider threshold. Positive and negative overlaps are de-duplicated separately, so a rally does not suppress an overlapping selloff. Setup metrics are from the trading day immediately before the event start.">
            Major Events 40 Day Look Forward ⓘ
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(detailed_styled, use_container_width=True, hide_index=True, height=height)
    event_count = len(display_table)
    st.caption(
        f"{event_count} major events where the independent 40-trading-day look-forward upside or downside move reached +/- {threshold_pct}% or greater. "
        "Setup metrics are from the trading day immediately before the event start. Scroll inside the table to view additional events."
    )

def render_profile(profile, key_prefix="main"):
    fig = build_analog_chart(profile)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    render_hmm_state_section(profile, key_prefix=key_prefix)
    render_summary_metrics(profile, key_prefix=key_prefix)
    render_distribution(profile, key_prefix=key_prefix)
    render_analogs_table(profile)
    render_major_events_section(profile, key_prefix=key_prefix)

def main():
    st.set_page_config(page_title="Stock Setup Profiler", layout="wide")
    inject_custom_css()

    st.markdown(
        """
        <div class="app-title-wrap">
            <div class="app-title">Stock Setup Profiler</div>
            <div class="app-subtitle">Find historical matches and outcomes for current market conditions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inputs (v12.26)")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        comparison_ticker = st.text_input("Second ticker for comparison", value="MSFT").strip().upper()
        benchmark = st.text_input("Benchmark 1", value="SPY").strip().upper()
        benchmark_2 = st.text_input("Benchmark 2", value="ITA").strip().upper()
        benchmark_3 = st.text_input("Benchmark 3", value="TLT").strip().upper()
        benchmark_4 = st.text_input("Benchmark 4", value="GLD").strip().upper()
        period = st.selectbox("History", ["2y", "5y", "10y", "max"], index=1)

        st.header("Similarity matching")
        compression_tolerance_ratio = st.slider("Compression ratio tolerance (+/-)", 0.05, 2.00, 0.20, 0.05)
        st.caption("Compression ratio = ATR10 / ATR50. Lower values mean compression; higher values mean expansion.")

        st.markdown("**Additional matching filters**")
        use_clv = st.toggle("Match CLV trend", value=False)
        clv_tolerance = st.slider("CLV trend tolerance", 0.01, 0.50, 0.10, 0.01, disabled=not use_clv)

        use_volume = st.toggle("Match volume support", value=False)
        volume_tolerance_pct = st.slider("Volume ratio tolerance (+/- %)", 5, 100, 25, 5, disabled=not use_volume)

        use_rs = st.toggle("Match relative strength (20D)", value=False)
        rs_tolerance_pp = st.slider("Relative strength (20D) tolerance (+/- percentage points)", 1, 30, 5, 1, disabled=not use_rs)

        run_button = st.button("Run profile", type="primary")

    selected_filters = []
    if use_clv:
        selected_filters.append("CLV trend")
    if use_volume:
        selected_filters.append("Volume support")
    if use_rs:
        selected_filters.append("Relative strength")

    benchmark_inputs = []
    for entered_benchmark in [benchmark, benchmark_2, benchmark_3, benchmark_4]:
        entered_benchmark = str(entered_benchmark).strip().upper()
        if entered_benchmark and entered_benchmark not in benchmark_inputs:
            benchmark_inputs.append(entered_benchmark)
    if not benchmark_inputs:
        st.error("Enter at least one benchmark.")
        return
    rs_col_map = build_rs_col_map(benchmark_inputs)

    requested_tickers = []
    for entered_ticker in [ticker, comparison_ticker]:
        if entered_ticker and entered_ticker not in requested_tickers:
            requested_tickers.append(entered_ticker)

    if not requested_tickers:
        st.error("Enter at least one ticker.")
        return

    if not run_button and "profiles" not in st.session_state:
        run_button = True

    current_request_key = (
        tuple(requested_tickers),
        tuple(benchmark_inputs),
        period,
        compression_tolerance_ratio,
        tuple(selected_filters),
        clv_tolerance,
        volume_tolerance_pct,
        rs_tolerance_pp,
    )

    if run_button or st.session_state.get("profile_request_key") != current_request_key:
        try:
            with st.spinner("Building profiles..."):
                benchmark_dfs = {bm: download_ohlcv(bm, period) for bm in benchmark_inputs}
                profiles = []
                for requested_ticker in requested_tickers:
                    profiles.append(
                        build_profile(
                            ticker=requested_ticker,
                            benchmark_dfs=benchmark_dfs,
                            period=period,
                            compression_tolerance_ratio=compression_tolerance_ratio,
                            selected_filters=selected_filters,
                            clv_tolerance=clv_tolerance,
                            volume_tolerance_pct=volume_tolerance_pct,
                            rs_tolerance_pp=rs_tolerance_pp,
                            rs_col_map=rs_col_map,
                        )
                    )
                st.session_state["profiles"] = profiles
                st.session_state["profile_request_key"] = current_request_key
        except Exception as e:
            st.error(str(e))
            return

    profiles = st.session_state.get("profiles", [])
    if not profiles:
        st.info("Run a profile to view results.")
        return

    tabs = st.tabs([f'{profile["ticker"]} outcomes' for profile in profiles])
    for tab, profile in zip(tabs, profiles):
        with tab:
            render_profile(profile, key_prefix=profile["ticker"].lower().replace(".", "_"))


if __name__ == "__main__":
    main()
