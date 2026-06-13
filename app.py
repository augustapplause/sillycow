import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ----------------------------
# Helpers
# ----------------------------


def money0(x):
    if x is None or pd.isna(x):
        return "Not available"
    x = float(x)
    if x < 0:
        return f"-${abs(x):,.0f}"
    return f"${x:,.0f}"


def pct1(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.1f}%"


def normalize_ohlcv_columns(data, ticker=None):
    """Normalize yfinance output into Open/High/Low/Close/Volume columns."""
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


# ----------------------------
# Core calculations
# ----------------------------


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
    """Percentile rank of latest value within each rolling window, 0-100."""
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

    # Volatility percentile is high when ATR/price is high.
    # Compression percentile is inverse: high means unusually compressed.
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
        stock_60 = aligned["Stock_Close"].pct_change(60)
        bench_60 = aligned["Benchmark_Close"].pct_change(60)
        stock_20 = aligned["Stock_Close"].pct_change(20)
        bench_20 = aligned["Benchmark_Close"].pct_change(20)
        df["RS_60"] = (stock_60 - bench_60) * 100
        df["RS_20"] = (stock_20 - bench_20) * 100
    else:
        df["RS_60"] = np.nan
        df["RS_20"] = np.nan

    return df


# ----------------------------
# HVN engine
# ----------------------------


def compute_hvns(df, bins=140, top_nodes=20, decay_days=180):
    if df.empty:
        return []

    min_price = df["Low"].min()
    max_price = df["High"].max()
    if not np.isfinite(min_price) or not np.isfinite(max_price) or max_price <= min_price:
        return []

    bin_edges = np.linspace(min_price, max_price, bins + 1)
    volume_profile = np.zeros(bins)
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

    peaks = []
    for i in range(1, len(volume_profile) - 1):
        if volume_profile[i] > volume_profile[i - 1] and volume_profile[i] > volume_profile[i + 1]:
            peaks.append(
                {
                    "rank": None,
                    "price": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                    "weighted_volume": float(volume_profile[i]),
                }
            )

    if not peaks:
        for idx in np.argsort(volume_profile)[::-1][:top_nodes]:
            peaks.append(
                {
                    "rank": None,
                    "price": float((bin_edges[idx] + bin_edges[idx + 1]) / 2),
                    "weighted_volume": float(volume_profile[idx]),
                }
            )

    ranked = sorted(peaks, key=lambda x: x["weighted_volume"], reverse=True)[:top_nodes]
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i

    return sorted(ranked, key=lambda x: x["rank"])


# ----------------------------
# Similar setup scan
# ----------------------------


def state_label(value: float, kind: str) -> str:
    if pd.isna(value):
        return "N/A"
    if kind == "clv":
        if value > 0.10:
            return "Bullish"
        if value < -0.10:
            return "Bearish"
        return "Neutral"
    if kind == "volume":
        if value >= 1.20:
            return "Supportive"
        if value <= 0.80:
            return "Weak"
        return "Neutral"
    if kind == "rs":
        if value >= 10:
            return "Strong"
        if value >= 3:
            return "Positive"
        if value <= -10:
            return "Weak"
        if value <= -3:
            return "Negative"
        return "Neutral"
    return "N/A"


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
    current_date = clean.index[-1]

    # Exclude current row from historical analogs. Also Future_Close_5D dropna already removes last five bars.
    candidates = clean[clean.index < current_date].copy()

    mask = (candidates["Compression_Percentile"] - current["Compression_Percentile"]).abs() <= compression_tolerance_pp

    if "CLV trend" in selected_filters:
        mask &= (candidates["CLV_Trend"] - current["CLV_Trend"]).abs() <= clv_tolerance

    if "Volume support" in selected_filters:
        current_vol = current["Volume_Ratio"]
        vol_tol = volume_tolerance_pct / 100
        low = current_vol * (1 - vol_tol)
        high = current_vol * (1 + vol_tol)
        mask &= candidates["Volume_Ratio"].between(low, high)

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
    out["CLV_Trend"] = out["CLV_Trend"].round(3)
    out["Volume_Ratio"] = out["Volume_Ratio"].round(2)
    out["RS_60"] = out["RS_60"].round(1)

    return out[
        [
            "Date",
            "Close",
            "Future_Close_5D",
            "Dollar_Change_5D",
            "Forward_Return_5D",
            "Compression_Percentile",
            "CLV_Trend",
            "Volume_Ratio",
            "RS_60",
        ]
    ]


# ----------------------------
# Profile builder
# ----------------------------


def build_profile(
    ticker: str,
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    period: str,
    compression_tolerance_pp: float,
    selected_filters: List[str],
    clv_tolerance: float,
    volume_tolerance_pct: float,
    rs_tolerance_pp: float,
):
    raw = download_ohlcv(ticker, period)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if len(raw) < 320:
        raise ValueError(f"{ticker} needs at least ~320 daily bars for percentile-based setup scanning.")

    df = add_profile_columns(raw, benchmark_df)
    df = df.dropna(subset=["Compression_Percentile", "CLV_Trend", "Volume_Ratio"])

    hvns = compute_hvns(df.tail(504), top_nodes=20)
    analogs = scan_similar_setups(
        df,
        compression_tolerance_pp=compression_tolerance_pp,
        selected_filters=selected_filters,
        clv_tolerance=clv_tolerance,
        volume_tolerance_pct=volume_tolerance_pct,
        rs_tolerance_pp=rs_tolerance_pp,
    )

    latest = df.iloc[-1]
    rs_value = float(latest.get("RS_60", np.nan))
    volume_ratio = float(latest["Volume_Ratio"])
    clv_trend = float(latest["CLV_Trend"])
    compression_pct = float(latest["Compression_Percentile"])

    metrics = {
        "Ticker": ticker.upper(),
        "Last data bar": str(df.index[-1].date()),
        "Current price": money0(float(latest["Close"])),
        "20D compression percentile": pct1(compression_pct),
        "CLV trend": f"{clv_trend:+.3f} ({state_label(clv_trend, 'clv')})",
        "Volume support": f"{volume_ratio:.2f}x 20D avg ({state_label(volume_ratio, 'volume')})",
        f"Relative strength vs {benchmark_name}": f"{rs_value:+.1f}% over 60D ({state_label(rs_value, 'rs')})",
        "Historical setup count": len(analogs),
    }

    return {
        "ticker": ticker.upper(),
        "df": df,
        "hvns": hvns,
        "analogs": analogs,
        "metrics": metrics,
    }


# ----------------------------
# Rendering
# ----------------------------


def render_metrics(profile):
    m = profile["metrics"]
    st.markdown(f"### {profile['ticker']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current price", m["Current price"])
    c2.metric("Compression", m["20D compression percentile"])
    c3.metric("Analog count", m["Historical setup count"])
    c4.metric("Last bar", m["Last data bar"])

    rs_key = [k for k in m.keys() if k.startswith("Relative strength")][0]
    details = pd.DataFrame(
        [
            ["CLV trend", m["CLV trend"]],
            ["Volume support", m["Volume support"]],
            [rs_key, m[rs_key]],
        ],
        columns=["Metric", "Value"],
    )
    st.dataframe(details, use_container_width=True, hide_index=True)


def build_hvn_chart(profile):
    df = profile["df"]
    hvns = profile["hvns"]
    ticker = profile["ticker"]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(df.index, df["Close"], linewidth=1.8, label="Close")

    # Plot all top 20 HVNs. Label only top 8 to keep legend usable.
    for h in hvns:
        rank = h["rank"]
        price = h["price"]
        alpha = max(0.20, 0.95 - (rank - 1) * 0.035)
        lw = 2.2 if rank <= 5 else 1.0
        label = f"HVN #{rank}: {money0(price)}" if rank <= 8 else None
        ax.axhline(price, linestyle="--", linewidth=lw, alpha=alpha, label=label)

    ax.set_title(f"{ticker}: Top 20 HVNs on Price-Time Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def build_analog_chart(profile):
    analogs = profile["analogs"]
    ticker = profile["ticker"]
    latest_close = float(profile["df"].iloc[-1]["Close"])

    fig, ax = plt.subplots(figsize=(13, 4.8))

    if analogs.empty:
        ax.text(0.5, 0.5, "No historical analogs found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    chart_df = analogs.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
    ax.scatter(chart_df["Date"], chart_df["Future_Close_5D"], s=38, label="Close 5 trading days later")
    ax.plot(chart_df["Date"], chart_df["Future_Close_5D"], alpha=0.35)
    ax.axhline(latest_close, linestyle="--", linewidth=1.5, label=f"Current close {money0(latest_close)}")
    ax.set_title(f"{ticker}: Historical Similar Compression Setups — Price 5 Days Later")
    ax.set_xlabel("Analog Date")
    ax.set_ylabel("Share Price 5 Days Later")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def render_profile(profile):
    render_metrics(profile)

    tab1, tab2, tab3 = st.tabs(["HVN chart", "Similar setup outcomes", "Data tables"])

    with tab1:
        fig = build_hvn_chart(profile)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        hvn_table = pd.DataFrame(profile["hvns"])
        if not hvn_table.empty:
            hvn_table["price"] = hvn_table["price"].map(lambda x: f"${x:,.2f}")
            hvn_table["weighted_volume"] = hvn_table["weighted_volume"].map(lambda x: f"{x:,.0f}")
            st.dataframe(hvn_table, use_container_width=True, hide_index=True)

    with tab2:
        analogs = profile["analogs"]
        fig = build_analog_chart(profile)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if not analogs.empty:
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Analog count", len(analogs))
            c2.metric("Avg 5D return", f"{analogs['Forward_Return_5D'].mean():+.2f}%")
            c3.metric("Median 5D return", f"{analogs['Forward_Return_5D'].median():+.2f}%")
            c4.metric("Win rate", f"{(analogs['Forward_Return_5D'] > 0).mean() * 100:.1f}%")
            c5.metric("Avg $ change", money0(analogs['Dollar_Change_5D'].mean()))
            c6.metric("Min $ change", money0(analogs['Dollar_Change_5D'].min()))
            c7.metric("Max $ change", money0(analogs['Dollar_Change_5D'].max()))

            st.dataframe(analogs, use_container_width=True, hide_index=True)
            st.download_button(
                "Download similar setup results",
                data=analogs.to_csv(index=False),
                file_name=f"{profile['ticker']}_similar_setups.csv",
                mime="text/csv",
            )
        else:
            st.info("No similar historical setup was found with the current filters. Try widening the tolerance or removing an optional filter.")

    with tab3:
        st.caption("Current-state metrics")
        metrics_df = pd.DataFrame(profile["metrics"].items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def compare_profiles_table(profiles: List[dict]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        m = p["metrics"]
        rs_key = [k for k in m.keys() if k.startswith("Relative strength")][0]
        analogs = p["analogs"]
        rows.append(
            {
                "Ticker": p["ticker"],
                "Current price": m["Current price"],
                "Compression": m["20D compression percentile"],
                "CLV trend": m["CLV trend"],
                "Volume support": m["Volume support"],
                "Relative strength": m[rs_key],
                "Analog count": len(analogs),
                "Avg 5D return": "N/A" if analogs.empty else f"{analogs['Forward_Return_5D'].mean():+.2f}%",
                "Median 5D return": "N/A" if analogs.empty else f"{analogs['Forward_Return_5D'].median():+.2f}%",
                "Win rate": "N/A" if analogs.empty else f"{(analogs['Forward_Return_5D'] > 0).mean() * 100:.1f}%",
                "Avg $ change": "N/A" if analogs.empty else money0(analogs['Dollar_Change_5D'].mean()),
                "Min $ change": "N/A" if analogs.empty else money0(analogs['Dollar_Change_5D'].min()),
                "Max $ change": "N/A" if analogs.empty else money0(analogs['Dollar_Change_5D'].max()),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# Streamlit app
# ----------------------------


def main():
    st.set_page_config(page_title="Historical Setup Profiler", layout="wide")
    st.title("Historical Setup Profiler")
    st.caption(
        "Find historical analogs for the current compression / CLV / volume / relative-strength setup, "
        "then inspect what happened 5 trading days later."
    )

    with st.sidebar:
        st.header("Inputs")
        ticker_1 = st.text_input("Ticker 1", value="AAPL").strip().upper()
        ticker_2 = st.text_input("Ticker 2 / Peer", value="MSFT").strip().upper()
        benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
        period = st.selectbox("History", ["2y", "5y", "10y", "max"], index=1)

        st.header("Similarity matching")
        st.caption("Compression percentile is always matched. Turn on extra filters to narrow the historical analogs.")
        compression_tolerance_pp = st.slider("Compression percentile tolerance (+/- points)", 1, 25, 5, 1)

        st.markdown("**Additional matching filters**")
        use_clv = st.toggle("Match CLV trend", value=False)
        clv_tolerance = st.slider("CLV trend tolerance", 0.01, 0.50, 0.10, 0.01, disabled=not use_clv)

        use_volume = st.toggle("Match volume support", value=False)
        volume_tolerance_pct = st.slider("Volume ratio tolerance (+/- %)", 5, 100, 25, 5, disabled=not use_volume)

        use_rs = st.toggle("Match relative strength", value=False)
        rs_tolerance_pp = st.slider("Relative strength tolerance (+/- percentage points)", 1, 30, 5, 1, disabled=not use_rs)

        selected_filters = []
        if use_clv:
            selected_filters.append("CLV trend")
        if use_volume:
            selected_filters.append("Volume support")
        if use_rs:
            selected_filters.append("Relative strength")

        run_button = st.button("Run profile", type="primary")

    if not run_button:
        st.info("Enter one or two tickers, choose filters, then click Run profile.")
        return

    if not ticker_1:
        st.error("Ticker 1 is required.")
        return

    try:
        with st.spinner("Downloading benchmark data..."):
            benchmark_df = download_ohlcv(benchmark, period)
            if benchmark_df.empty:
                raise ValueError(f"No data returned for benchmark {benchmark}.")

        profiles = []
        for ticker in [ticker_1, ticker_2]:
            if ticker:
                with st.spinner(f"Profiling {ticker}..."):
                    profiles.append(
                        build_profile(
                            ticker=ticker,
                            benchmark_df=benchmark_df,
                            benchmark_name=benchmark,
                            period=period,
                            compression_tolerance_pp=compression_tolerance_pp,
                            selected_filters=selected_filters,
                            clv_tolerance=clv_tolerance,
                            volume_tolerance_pct=volume_tolerance_pct,
                            rs_tolerance_pp=rs_tolerance_pp,
                        )
                    )

        st.subheader("Side-by-side setup comparison")
        criteria = [f"Compression percentile ±{compression_tolerance_pp} pts"]
        if use_clv:
            criteria.append(f"CLV trend ±{clv_tolerance:.2f}")
        if use_volume:
            criteria.append(f"Volume ratio ±{volume_tolerance_pct}%")
        if use_rs:
            criteria.append(f"Relative strength ±{rs_tolerance_pp} pts")

        st.markdown(
            f"""
            <div style="
                font-size:0.60rem;
                color:#888888;
                margin-top:-12px;
                margin-bottom:8px;
            ">
            Matched on: {'; '.join(criteria)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        compare_df = compare_profiles_table(profiles)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download side-by-side comparison",
            data=compare_df.to_csv(index=False),
            file_name="setup_comparison.csv",
            mime="text/csv",
        )

        if len(profiles) == 1:
            render_profile(profiles[0])
        else:
            tab_labels = [p["ticker"] for p in profiles]
            tabs = st.tabs(tab_labels)
            for tab, profile in zip(tabs, profiles):
                with tab:
                    render_profile(profile)

    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
