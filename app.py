import html
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def money0(x):
    if x is None or pd.isna(x):
        return "Not available"
    x = float(x)
    if x < 0:
        return f"-${abs(x):,.0f}"
    return f"${x:,.0f}"


def money2(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    if x < 0:
        return f"-${abs(x):,.2f}"
    return f"${x:,.2f}"


def signed_money0(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    if x < 0:
        return f"-${abs(x):,.0f}"
    return f"${x:,.0f}"


def pct1(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.1f}%"


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


def add_profile_columns(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    df["CLV"] = clv(df)
    df["CLV_5"] = df["CLV"].rolling(5).mean()
    df["CLV_20"] = df["CLV"].rolling(20).mean()
    df["CLV_Trend"] = df["CLV_5"] - df["CLV_20"]

    df["ATR20"] = atr(df, 20)
    df["ATR20_PctPrice"] = df["ATR20"] / (df["Close"] + 1e-9)
    df["Volatility_Percentile"] = rolling_percentile_rank(df["ATR20_PctPrice"], 252)
    df["Compression_Percentile"] = 100 - df["Volatility_Percentile"]

    df["Volume_20_Mean"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_20_Mean"] + 1e-9)

    df["Future_Close_5D"] = df["Close"].shift(-5)
    df["Forward_Return_5D"] = (df["Future_Close_5D"] / df["Close"] - 1) * 100

    if benchmark_df is not None and not benchmark_df.empty:
        aligned = df[["Close"]].rename(columns={"Close": "Stock_Close"}).join(
            benchmark_df[["Close"]].rename(columns={"Close": "Benchmark_Close"}),
            how="left",
        )
        df["RS_60"] = (aligned["Stock_Close"].pct_change(60) - aligned["Benchmark_Close"].pct_change(60)) * 100
        df["RS_20"] = (aligned["Stock_Close"].pct_change(20) - aligned["Benchmark_Close"].pct_change(20)) * 100
    else:
        df["RS_60"] = np.nan
        df["RS_20"] = np.nan

    return df


def compute_hvns(df, bins=140, top_nodes=10, decay_days=180):
    if df.empty:
        return []

    min_price = df["Low"].min()
    max_price = df["High"].max()
    if not np.isfinite(min_price) or not np.isfinite(max_price) or max_price <= min_price:
        return []

    bin_edges = np.linspace(min_price, max_price, bins + 1)
    volume_profile = np.zeros(bins)
    touch_counts = np.zeros(bins)
    latest_date = df.index[-1]

    for date, row in df.iterrows():
        low = float(row["Low"])
        high = float(row["High"])
        volume = float(row["Volume"])

        if high <= low or volume <= 0:
            continue

        age_days = max((latest_date - date).days, 0)
        weighted_volume = volume * np.exp(-age_days / decay_days)
        touched_bins = np.where((bin_edges[:-1] <= high) & (bin_edges[1:] >= low))[0]

        if len(touched_bins) == 0:
            continue

        volume_profile[touched_bins] += weighted_volume / len(touched_bins)
        touch_counts[touched_bins] += 1

    peaks = []
    for i in range(1, len(volume_profile) - 1):
        if volume_profile[i] > volume_profile[i - 1] and volume_profile[i] > volume_profile[i + 1]:
            peaks.append(i)

    if not peaks:
        peaks = list(np.argsort(volume_profile)[::-1][:top_nodes])

    ranked_idx = sorted(peaks, key=lambda i: volume_profile[i], reverse=True)[:top_nodes]
    total_volume = volume_profile.sum() + 1e-9

    out = []
    for rank, i in enumerate(ranked_idx, start=1):
        strength = "Very Strong" if rank <= 2 else "Strong" if rank <= 5 else "Moderate"
        out.append(
            {
                "rank": rank,
                "price": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "weighted_volume": float(volume_profile[i]),
                "percent_total": float(volume_profile[i] / total_volume * 100),
                "touches": int(touch_counts[i]),
                "strength": strength,
            }
        )

    return sorted(out, key=lambda x: x["price"], reverse=True)


def scan_similar_setups(
    df: pd.DataFrame,
    compression_tolerance_pp: float,
    selected_filters: List[str],
    clv_tolerance: float,
    volume_tolerance_pct: float,
    rs_tolerance_pp: float,
) -> pd.DataFrame:
    clean = df.dropna(
        subset=[
            "Compression_Percentile",
            "CLV_Trend",
            "Volume_Ratio",
            "RS_60",
            "Future_Close_5D",
            "Forward_Return_5D",
        ]
    ).copy()

    if clean.empty:
        return pd.DataFrame()

    current = clean.iloc[-1]
    candidates = clean[clean.index < clean.index[-1]].copy()

    mask = (candidates["Compression_Percentile"] - current["Compression_Percentile"]).abs() <= compression_tolerance_pp

    if "CLV trend" in selected_filters:
        mask &= (candidates["CLV_Trend"] - current["CLV_Trend"]).abs() <= clv_tolerance

    if "Volume support" in selected_filters:
        vol_tol = volume_tolerance_pct / 100
        mask &= candidates["Volume_Ratio"].between(
            current["Volume_Ratio"] * (1 - vol_tol),
            current["Volume_Ratio"] * (1 + vol_tol),
        )

    if "Relative strength" in selected_filters:
        mask &= (candidates["RS_60"] - current["RS_60"]).abs() <= rs_tolerance_pp

    out = candidates.loc[mask].copy()
    if out.empty:
        return pd.DataFrame()

    out = out.reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    out["Dollar_Change_5D"] = out["Future_Close_5D"] - out["Close"]

    out["Close"] = out["Close"].round(2)
    out["Future_Close_5D"] = out["Future_Close_5D"].round(2)
    out["Dollar_Change_5D"] = out["Dollar_Change_5D"].round(2)
    out["Forward_Return_5D"] = out["Forward_Return_5D"].round(2)
    out["Compression_Percentile"] = out["Compression_Percentile"].round(1)
    out["CLV_Trend"] = out["CLV_Trend"].round(2)
    out["RS_20"] = out["RS_20"].round(1)

    return out[
        [
            "Date",
            "Close",
            "Future_Close_5D",
            "Dollar_Change_5D",
            "Forward_Return_5D",
            "Compression_Percentile",
            "CLV_Trend",
            "RS_20",
        ]
    ]


def build_profile(
    ticker: str,
    benchmark_df: pd.DataFrame,
    period: str,
    compression_tolerance_pp: float,
    selected_filters: List[str],
    clv_tolerance: float,
    volume_tolerance_pct: float,
    rs_tolerance_pp: float,
    hvn_count: int,
    hvn_decay_days: int,
):
    raw = download_ohlcv(ticker, period)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if len(raw) < 320:
        raise ValueError(f"{ticker} needs at least ~320 daily bars.")

    df = add_profile_columns(raw, benchmark_df)
    df = df.dropna(subset=["Compression_Percentile", "CLV_Trend", "Volume_Ratio"])

    return {
        "ticker": ticker.upper(),
        "df": df,
        "hvns": compute_hvns(df.tail(504), top_nodes=hvn_count, decay_days=hvn_decay_days),
        "analogs": scan_similar_setups(
            df,
            compression_tolerance_pp,
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
        .block-container {
            max-width: 1280px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
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
        div[data-testid="stTabs"] div[role="tabpanel"] {
            padding-top: 0.25rem;
        }
        .mock-card {
            border: 1px solid #dfe5ef;
            border-radius: 10px;
            background: white;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(20, 30, 60, 0.03);
            margin-bottom: 14px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0;
            text-align: center;
        }
        .metric-box {
            border-right: 1px solid #e7ebf3;
            min-height: 78px;
            padding: 5px 10px;
        }
        .metric-box:last-child {
            border-right: 0;
        }
        .metric-label {
            color: #0f1730;
            font-size: 0.83rem;
            font-weight: 750;
            margin-bottom: 10px;
        }
        .metric-value {
            color: #061027;
            font-size: 1.92rem;
            font-weight: 850;
            line-height: 1.1;
        }
        .red {
            color: red !important;
        }
        .green {
            color: green !important;
        }
        .outcome-card {
            border: 1px solid #dfe5ef;
            border-radius: 10px;
            padding: 28px 24px;
            text-align: center;
            min-height: 238px;
        }
        .outcome-title {
            color: #0f1730;
            font-size: 1.14rem;
            margin-bottom: 10px;
        }
        .outcome-number {
            font-size: 1.75rem;
            font-weight: 850;
            margin-bottom: 28px;
        }
        .divider {
            border-top: 1px solid #d5dbe7;
            margin: 6px 0 24px;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_analog_chart(profile):
    analogs = profile["analogs"]
    ticker = profile["ticker"]
    latest_close = float(profile["df"].iloc[-1]["Close"])

    fig, ax = plt.subplots(figsize=(13, 5.8))

    if analogs.empty:
        ax.text(0.5, 0.5, "No historical analogs found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    chart_df = analogs.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    chart_df = chart_df.sort_values("Date")

    ax.plot(chart_df["Date"], chart_df["Future_Close_5D"], linewidth=1.1, alpha=0.65)
    ax.scatter(chart_df["Date"], chart_df["Future_Close_5D"], s=70, label="Close 5 trading days later", zorder=3)

    best_idx = chart_df["Dollar_Change_5D"].idxmax()
    worst_idx = chart_df["Dollar_Change_5D"].idxmin()
    best = chart_df.loc[best_idx]
    worst = chart_df.loc[worst_idx]
    ax.scatter([pd.to_datetime(best["Date"])], [best["Future_Close_5D"]], s=160, color="green", label="Biggest advance", zorder=4)
    ax.scatter([pd.to_datetime(worst["Date"])], [worst["Future_Close_5D"]], s=160, color="red", label="Biggest decline", zorder=4)

    ax.axhline(latest_close, linestyle="--", linewidth=1.4, color="#2166ff", label=f"Current close {money0(latest_close)}")
    ax.text(chart_df["Date"].max(), latest_close + 1, f"Current close {money0(latest_close)}", color="#2166ff", ha="right", va="bottom", fontsize=10, fontweight="bold")

    top_hvns = profile["hvns"][:3]
    for h in top_hvns:
        ax.axhline(h["price"], linestyle="--", linewidth=1.2, color="green", alpha=0.85)
        ax.text(chart_df["Date"].max(), h["price"], money2(h["price"]), color="green", ha="left", va="center", fontsize=10, fontweight="bold")

    ax.plot([], [], linestyle="--", color="green", label="Top 3 HVNs")

    ax.set_title(f"{ticker} (Apple Inc.) : Historical Similar Compression Setups — Price 5 Days Later", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Analog Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share Price 5 Days Later", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    return fig


def render_summary_metrics(profile):
    analogs = profile["analogs"]

    if analogs.empty:
        values = ["0", "N/A", "N/A", "N/A", "N/A", "N/A"]
        classes = ["", "", "", "", "", ""]
    else:
        avg_ret = analogs["Forward_Return_5D"].mean()
        med_ret = analogs["Forward_Return_5D"].median()
        win_rate = (analogs["Forward_Return_5D"] > 0).mean() * 100
        avg_change = analogs["Dollar_Change_5D"].mean()
        min_change = analogs["Dollar_Change_5D"].min()
        max_change = analogs["Dollar_Change_5D"].max()
        values = [
            f"{len(analogs)}",
            signed_pct1(avg_ret),
            signed_pct1(med_ret),
            f"{win_rate:.1f}%",
            signed_money0(avg_change),
            f"{signed_money0(min_change)} to {signed_money0(max_change)}",
        ]
        classes = ["", "green" if avg_ret > 0 else "red", "green" if med_ret > 0 else "red", "", "green" if avg_change > 0 else "red", ""]

    labels = ["Analog count", "Avg 5D return", "Median 5D return", "Win rate", "Avg $ change", "Change Range"]

    boxes = "".join(
        f"""
        <div class="metric-box">
            <div class="metric-label">{html.escape(labels[i])}</div>
            <div class="metric-value {classes[i]}">{html.escape(values[i])}</div>
        </div>
        """
        for i in range(6)
    )

    st.markdown(f'<div class="mock-card"><div class="metric-grid">{boxes}</div></div>', unsafe_allow_html=True)


def build_distribution_chart(profile, bins_count=12):
    analogs = profile["analogs"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    labels = ["≥ $20", "$15 to $20", "$10 to $15", "$5 to $10", "$0 to $5", "-$5 to $0", "-$10 to -$5", "-$15 to -$10", "-$20 to -$15", "≤ -$20"]

    if analogs.empty:
        counts = [0] * len(labels)
    else:
        s = analogs["Dollar_Change_5D"]
        counts = [
            int((s >= 20).sum()),
            int(((s >= 15) & (s < 20)).sum()),
            int(((s >= 10) & (s < 15)).sum()),
            int(((s >= 5) & (s < 10)).sum()),
            int(((s >= 0) & (s < 5)).sum()),
            int(((s >= -5) & (s < 0)).sum()),
            int(((s >= -10) & (s < -5)).sum()),
            int(((s >= -15) & (s < -10)).sum()),
            int(((s >= -20) & (s < -15)).sum()),
            int((s < -20).sum()),
        ]

    y = np.arange(len(labels))
    ax.barh(y, counts, color="#0d5bd6")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dollar Change (5D)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)

    for i, count in enumerate(counts):
        ax.text(count + 0.2, i, str(count), va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, max(max(counts) + 4, 10))
    fig.tight_layout()
    return fig


def render_distribution(profile):
    analogs = profile["analogs"]
    positives = int((analogs["Dollar_Change_5D"] > 0).sum()) if not analogs.empty else 0
    negatives = int((analogs["Dollar_Change_5D"] <= 0).sum()) if not analogs.empty else 0
    total = max(len(analogs), 1)

    st.markdown('<div class="mock-card">', unsafe_allow_html=True)
    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.markdown('<div class="section-title">Distribution of 5-Day Dollar Change ⓘ</div>', unsafe_allow_html=True)
    with top_right:
        st.slider("Number of bins", 6, 24, 12, 1)

    c1, c2 = st.columns([3.4, 1.25])
    with c1:
        fig = build_distribution_chart(profile)
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

    st.markdown("</div>", unsafe_allow_html=True)


def render_analogs_table(profile):
    analogs = profile["analogs"].head(10).copy()
    if analogs.empty:
        st.info("No similar historical setup was found with the current filters.")
        return

    styled = analogs.style.format(
        {
            "Close": "{:.2f}",
            "Future_Close_5D": "{:.2f}",
            "Dollar_Change_5D": "{:.2f}",
            "Forward_Return_5D": "{:.2f}%",
            "Compression_Percentile": "{:.1f}",
            "CLV_Trend": "{:.2f}",
            "RS_20": "{:.1f}",
        }
    ).map(
        lambda v: "color: green; font-weight: 800;" if float(v) > 0 else "color: red; font-weight: 800;",
        subset=["Dollar_Change_5D", "Forward_Return_5D"],
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=310)
    st.caption(f"Showing 1 to {min(10, len(profile['analogs']))} of {len(profile['analogs'])} entries")


def render_hvn_section(profile):
    st.markdown('<div class="section-title">HVN (High Volume Nodes) ⓘ</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 2.55])

    with left:
        st.markdown('<div class="mock-card">', unsafe_allow_html=True)
        st.selectbox("HVN Selection", ["Top 10 by Volume", "Top 5 by Volume", "Top 20 by Volume"], index=0)
        st.slider("Minimum Volume Percentile", 50, 99, 85, 1)
        st.slider("Node Decay (Days)", 30, 365, 180, 1)
        st.info("Node decay reduces the influence of older price activity. Lower values focus on recent data; higher values include more historical data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        hvn_table = pd.DataFrame(profile["hvns"])
        if hvn_table.empty:
            st.info("No HVNs were found.")
            return

        hvn_table = hvn_table.rename(
            columns={
                "price": "Price",
                "weighted_volume": "Volume (Weighted)",
                "percent_total": "Percent of Total",
                "touches": "Touches",
                "strength": "Strength",
            }
        )[["Price", "Volume (Weighted)", "Percent of Total", "Touches", "Strength"]]

        styled = hvn_table.style.format(
            {
                "Price": "${:,.2f}",
                "Volume (Weighted)": "{:,.0f}",
                "Percent of Total": "{:.2f}%",
                "Touches": "{:,.0f}",
            }
        ).map(
            lambda v: "color: green; font-weight: 800;" if v in ["Very Strong", "Strong"] else "color: #ff6a00; font-weight: 800;",
            subset=["Strength"],
        )

        st.dataframe(styled, use_container_width=True, hide_index=True, height=365)
        st.caption("Sorted by price (high to low)")


def render_profile(profile):
    fig = build_analog_chart(profile)
    st.markdown('<div class="mock-card">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    plt.close(fig)

    render_summary_metrics(profile)
    render_distribution(profile)
    render_analogs_table(profile)
    render_hvn_section(profile)


def main():
    st.set_page_config(page_title="Stock Setup Profiler", layout="wide")
    inject_custom_css()

    st.title("Stock Setup Profiler")
    st.caption("Find historical matches and outcomes for current market conditions")

    with st.expander("Controls", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Ticker", value="AAPL").strip().upper()
        benchmark = c2.text_input("Benchmark", value="SPY").strip().upper()
        period = c3.selectbox("History", ["2y", "5y", "10y", "max"], index=1)
        compression_tolerance_pp = c4.slider("Compression tolerance", 1, 25, 5, 1)

        c5, c6, c7, c8 = st.columns(4)
        use_clv = c5.toggle("Match CLV trend", value=False)
        clv_tolerance = c6.slider("CLV tolerance", 0.01, 0.50, 0.10, 0.01)
        use_rs = c7.toggle("Match relative strength", value=False)
        rs_tolerance_pp = c8.slider("RS tolerance", 1, 30, 5, 1)

        c9, c10, c11 = st.columns(3)
        use_volume = c9.toggle("Match volume support", value=False)
        volume_tolerance_pct = c10.slider("Volume tolerance", 5, 100, 25, 5)
        hvn_decay_days = c11.slider("Default HVN decay", 30, 365, 180, 1)

    selected_filters = []
    if use_clv:
        selected_filters.append("CLV trend")
    if use_volume:
        selected_filters.append("Volume support")
    if use_rs:
        selected_filters.append("Relative strength")

    try:
        with st.spinner("Building profile..."):
            benchmark_df = download_ohlcv(benchmark, period)
            profile = build_profile(
                ticker=ticker,
                benchmark_df=benchmark_df,
                period=period,
                compression_tolerance_pp=compression_tolerance_pp,
                selected_filters=selected_filters,
                clv_tolerance=clv_tolerance,
                volume_tolerance_pct=volume_tolerance_pct,
                rs_tolerance_pp=rs_tolerance_pp,
                hvn_count=10,
                hvn_decay_days=hvn_decay_days,
            )

        tab1, = st.tabs(["Similar setup outcomes"])
        with tab1:
            render_profile(profile)

    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()