import html
from typing import List, Optional

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


def money2(x):
    if x is None or pd.isna(x):
        return "Not available"
    x = float(x)
    if x < 0:
        return f"-${abs(x):,.2f}"
    return f"${x:,.2f}"


def signed_money0(x):
    if x is None or pd.isna(x):
        return "N/A"
    return money0(x)


def pct1(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.1f}%"


def signed_pct2(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:+.2f}%"


def value_color(x):
    if x is None or pd.isna(x):
        return "#111827"
    return "#008a16" if float(x) > 0 else "#d00000" if float(x) < 0 else "#111827"


def colored_span(text, color):
    return f'<span style="color:{color};">{html.escape(str(text))}</span>'


def colored_money0(x):
    return colored_span(money0(x), value_color(x))


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


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_name(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    try:
        info = yf.Ticker(ticker).get_info()
        name = info.get("shortName") or info.get("longName") or ""
        name = str(name).strip()
        if name:
            return name
    except Exception:
        pass
    return ticker


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

    total_volume = float(volume_profile.sum()) if float(volume_profile.sum()) > 0 else 1.0
    ranked_idx = sorted(peaks, key=lambda i: volume_profile[i], reverse=True)[:top_nodes]

    rows = []
    max_node_vol = max([float(volume_profile[i]) for i in ranked_idx], default=0.0)
    for rank, idx in enumerate(ranked_idx, start=1):
        pct_total = float(volume_profile[idx] / total_volume * 100)
        strength = "Very Strong" if rank <= 2 else "Strong" if rank <= 5 else "Moderate"
        rows.append(
            {
                "rank": rank,
                "price": float((bin_edges[idx] + bin_edges[idx + 1]) / 2),
                "weighted_volume": float(volume_profile[idx]),
                "percent_of_total": pct_total,
                "touches": int(touch_counts[idx]),
                "strength": strength,
                "node_volume_percentile": 0.0 if max_node_vol <= 0 else float(volume_profile[idx] / max_node_vol * 100),
            }
        )

    return rows


def filter_hvns(hvns: List[dict], selection: str, min_volume_percentile: int) -> List[dict]:
    if not hvns:
        return []
    if selection == "Top 5 by Volume":
        selected = hvns[:5]
    elif selection == "Top 10 by Volume":
        selected = hvns[:10]
    else:
        selected = [h for h in hvns if h.get("node_volume_percentile", 0) >= min_volume_percentile]
    return sorted(selected, key=lambda x: x["price"], reverse=True)


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
    hvn_bins: int,
    hvn_decay_days: int,
):
    raw = download_ohlcv(ticker, period)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if len(raw) < 320:
        raise ValueError(f"{ticker} needs at least ~320 daily bars for percentile-based setup scanning.")

    df = add_profile_columns(raw, benchmark_df)
    df = df.dropna(subset=["Compression_Percentile", "CLV_Trend", "Volume_Ratio"])

    hvns = compute_hvns(df.tail(504), bins=hvn_bins, top_nodes=20, decay_days=hvn_decay_days)
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
        "Stock name": get_stock_name(ticker),
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
        "stock_name": metrics["Stock name"],
        "df": df,
        "hvns": hvns,
        "analogs": analogs,
        "metrics": metrics,
    }


# ----------------------------
# CSS / HTML rendering helpers
# ----------------------------


def inject_custom_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        div[data-testid="stMetric"] { min-width: 0; }
        div[data-testid="stMetricLabel"] {
            font-size: clamp(1.05rem, 1.25vw, 1.45rem);
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        div[data-testid="stMetricValue"] {
            font-size: clamp(2.0rem, 3.25vw, 3.6rem);
            line-height: 1.05;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .section-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 18px 22px;
            margin: 14px 0 24px 0;
            background: white;
        }
        .spacer-lg { height: 28px; }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 18px 22px;
            margin: 8px 0 28px 0;
            background: white;
        }
        .metric-item { text-align: center; min-width: 0; }
        .metric-label {
            font-size: clamp(1.0rem, 1.25vw, 1.35rem);
            font-weight: 700;
            color: #111827;
            margin-bottom: 8px;
            white-space: nowrap;
        }
        .metric-value {
            font-size: clamp(2.15rem, 3.25vw, 3.65rem);
            line-height: 1.05;
            color: #111827;
            white-space: nowrap;
        }
        .big-section-title {
            font-size: clamp(1.7rem, 2.4vw, 2.5rem);
            font-weight: 800;
            margin: 12px 0 0 0;
            color: #111827;
        }
        .table-wrap {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            overflow-x: auto;
            margin: 22px 0 30px 0;
        }
        table.big-readable-table {
            border-collapse: collapse;
            width: 100%;
            font-size: clamp(1.15rem, 1.55vw, 1.75rem);
        }
        table.big-readable-table th {
            background: #f8fafc;
            color: #111827;
            font-weight: 800;
            text-align: right;
            padding: 15px 16px;
            border-bottom: 1px solid #e5e7eb;
            white-space: nowrap;
        }
        table.big-readable-table th:first-child,
        table.big-readable-table td:first-child { text-align: left; }
        table.big-readable-table td {
            padding: 13px 16px;
            border-bottom: 1px solid #e5e7eb;
            text-align: right;
            white-space: nowrap;
            font-weight: 650;
        }
        table.big-readable-table tr:last-child td { border-bottom: 0; }
        .hvn-note {
            background: #eaf3ff;
            border-radius: 10px;
            color: #075fd3;
            padding: 14px 16px;
            font-size: clamp(1.05rem, 1.35vw, 1.45rem);
            line-height: 1.35;
            margin-top: 18px;
        }
        @media (max-width: 900px) {
            .metric-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_strip(analogs: pd.DataFrame):
    avg_return = analogs["Forward_Return_5D"].mean()
    median_return = analogs["Forward_Return_5D"].median()
    win_rate = (analogs["Forward_Return_5D"] > 0).mean() * 100
    avg_change = analogs["Dollar_Change_5D"].mean()
    min_change = analogs["Dollar_Change_5D"].min()
    max_change = analogs["Dollar_Change_5D"].max()

    html_block = f"""
    <div class="metric-strip">
        <div class="metric-item"><div class="metric-label">Analog count</div><div class="metric-value">{len(analogs)}</div></div>
        <div class="metric-item"><div class="metric-label">Avg 5D return</div><div class="metric-value" style="color:{value_color(avg_return)};">{avg_return:+.1f}%</div></div>
        <div class="metric-item"><div class="metric-label">Median 5D return</div><div class="metric-value" style="color:{value_color(median_return)};">{median_return:+.1f}%</div></div>
        <div class="metric-item"><div class="metric-label">Win rate</div><div class="metric-value">{win_rate:.1f}%</div></div>
        <div class="metric-item"><div class="metric-label">Avg $ change</div><div class="metric-value" style="color:{value_color(avg_change)};">{money0(avg_change)}</div></div>
        <div class="metric-item"><div class="metric-label">Change Range</div><div class="metric-value">{colored_money0(min_change)} <span style="color:#111827;">to</span> {colored_money0(max_change)}</div></div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def format_analog_table(analogs: pd.DataFrame) -> pd.DataFrame:
    table = analogs.copy()
    table["Date"] = table["Date"].astype(str)
    table["Close"] = table["Close"].map(lambda x: f"{x:,.2f}")
    table["Future_Close_5D"] = table["Future_Close_5D"].map(lambda x: f"{x:,.2f}")
    table["Dollar_Change_5D"] = table["Dollar_Change_5D"].map(lambda x: colored_span(f"{x:,.2f}", value_color(x)))
    table["Forward_Return_5D"] = table["Forward_Return_5D"].map(lambda x: colored_span(f"{x:+.2f}%", value_color(x)))
    table["Compression_Percentile"] = table["Compression_Percentile"].map(lambda x: f"{x:,.1f}")
    table["CLV_Trend"] = table["CLV_Trend"].map(lambda x: f"{x:,.3f}")
    table["Volume_Ratio"] = table["Volume_Ratio"].map(lambda x: f"{x:,.2f}")
    table["RS_60"] = table["RS_60"].map(lambda x: f"{x:,.1f}")
    return table


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{str(v)}</td>" for v in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table class="big-readable-table"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ----------------------------
# Charts
# ----------------------------


def apply_large_chart_fonts(ax, title_size=28, label_size=25, tick_size=21, legend_size=18):
    ax.title.set_fontsize(title_size)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)


def build_analog_chart(profile, selected_hvns):
    analogs = profile["analogs"]
    ticker = profile["ticker"]
    stock_name = profile.get("stock_name", ticker)
    latest_close = float(profile["df"].iloc[-1]["Close"])

    fig, ax = plt.subplots(figsize=(16, 8.2))

    if analogs.empty:
        ax.text(0.5, 0.5, "No historical analogs found", ha="center", va="center", transform=ax.transAxes, fontsize=24)
        ax.set_axis_off()
        return fig

    chart_df = analogs.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"])

    ax.plot(chart_df["Date"], chart_df["Future_Close_5D"], alpha=0.55, linewidth=2.2)
    ax.scatter(chart_df["Date"], chart_df["Future_Close_5D"], s=150, label="Close 5 trading days later", zorder=3)

    # Mark biggest 5D dollar advance/decline with larger colored dots.
    max_idx = chart_df["Dollar_Change_5D"].idxmax()
    min_idx = chart_df["Dollar_Change_5D"].idxmin()
    ax.scatter(
        chart_df.loc[max_idx, "Date"],
        chart_df.loc[max_idx, "Future_Close_5D"],
        s=300,
        color="green",
        edgecolor="black",
        linewidth=0.5,
        label="Biggest advance",
        zorder=5,
    )
    ax.scatter(
        chart_df.loc[min_idx, "Date"],
        chart_df.loc[min_idx, "Future_Close_5D"],
        s=300,
        color="red",
        edgecolor="black",
        linewidth=0.5,
        label="Biggest decline",
        zorder=5,
    )

    ax.axhline(latest_close, linestyle="--", linewidth=2.2, label=f"Current close {money0(latest_close)}")
    ax.text(chart_df["Date"].max(), latest_close, f"  Current close {money0(latest_close)}", va="bottom", fontsize=18, fontweight="bold", color="#0057ff")

    top_three = sorted(selected_hvns, key=lambda x: x.get("rank", 999))[:3]
    for i, h in enumerate(top_three):
        label = "Top 3 HVNs" if i == 0 else None
        ax.axhline(h["price"], color="green", linestyle="--", linewidth=2.0, alpha=0.9, label=label)
        ax.text(chart_df["Date"].max(), h["price"], f"  {money2(h['price'])}", va="bottom", fontsize=18, fontweight="bold", color="green")

    ax.set_title(f"{ticker} ({stock_name}) : Historical Similar Compression Setups — Price 5 Days Later", pad=18)
    ax.set_xlabel("Analog Date")
    ax.set_ylabel("Share Price 5 Days Later")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    apply_large_chart_fonts(ax)
    fig.tight_layout(pad=2.0)
    return fig


def build_distribution_chart(analogs: pd.DataFrame, bin_count: int):
    fig, ax = plt.subplots(figsize=(16, 7.2))

    if analogs.empty:
        ax.text(0.5, 0.5, "No historical analogs found", ha="center", va="center", transform=ax.transAxes, fontsize=24)
        ax.set_axis_off()
        return fig

    changes = analogs["Dollar_Change_5D"].astype(float)
    min_val = float(changes.min())
    max_val = float(changes.max())
    if np.isclose(min_val, max_val):
        min_val -= 1.0
        max_val += 1.0

    edges = np.linspace(min_val, max_val, int(bin_count) + 1)
    counts, edges = np.histogram(changes, bins=edges)

    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f"{money0(left)} to {money0(right)}")

    y = np.arange(len(counts))
    ax.barh(y, counts, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Dollar Change (5D)")
    ax.set_title("Distribution of 5-Day Dollar Change", pad=16)
    ax.grid(True, axis="x", alpha=0.22)

    max_count = int(max(counts)) if len(counts) else 0
    for yi, count in zip(y, counts):
        ax.text(count + max(0.15, max_count * 0.015), yi, f"{int(count)}", va="center", fontsize=20, fontweight="bold")

    pos = int((changes > 0).sum())
    neg = int((changes < 0).sum())
    total = max(len(changes), 1)
    inset_text = (
        "Positive outcomes\n"
        f"{pos} ({pos / total * 100:.1f}%)\n\n"
        "Negative outcomes\n"
        f"{neg} ({neg / total * 100:.1f}%)"
    )
    ax.text(
        0.965,
        0.60,
        inset_text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=23,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.85", facecolor="white", edgecolor="#e5e7eb", alpha=0.96),
    )

    apply_large_chart_fonts(ax)
    fig.tight_layout(pad=2.0)
    return fig


# ----------------------------
# Page sections
# ----------------------------


def render_analogs_table(analogs: pd.DataFrame):
    display_df = format_analog_table(analogs)
    st.markdown(dataframe_to_html_table(display_df), unsafe_allow_html=True)


def render_hvn_section(profile, hvn_selection: str, min_volume_percentile: int):
    st.markdown('<div class="big-section-title">HVN (High Volume Nodes) <span style="color:#94a3b8;font-size:1.25rem;">ⓘ</span></div>', unsafe_allow_html=True)
    selected = filter_hvns(profile["hvns"], hvn_selection, min_volume_percentile)

    c1, c2 = st.columns([1.05, 3.2], gap="large")
    with c1:
        st.selectbox("HVN Selection", ["Top 10 by Volume", "Top 5 by Volume", "Minimum Volume Percentile"], index=["Top 10 by Volume", "Top 5 by Volume", "Minimum Volume Percentile"].index(hvn_selection), key=f"hvn_select_{profile['ticker']}", disabled=True)
        st.slider("Minimum Volume Percentile", 50, 99, int(min_volume_percentile), 1, key=f"hvn_pct_display_{profile['ticker']}", disabled=True)
        st.slider("Node Decay (Days)", 30, 365, int(st.session_state.get("hvn_decay_days", 180)), 5, key=f"hvn_decay_display_{profile['ticker']}", disabled=True)
        st.markdown(
            '<div class="hvn-note">Node decay reduces the influence of older price activity. Lower values focus on recent data; higher values include more historical data.</div>',
            unsafe_allow_html=True,
        )

    with c2:
        if not selected:
            st.info("No HVNs meet the current selection.")
            return
        rows = []
        for h in selected:
            rows.append(
                {
                    "Price": money2(h["price"]),
                    "Volume (Weighted)": f"{h['weighted_volume']:,.0f}",
                    "Percent of Total": f"{h['percent_of_total']:.2f}%",
                    "Touches": f"{h['touches']:,}",
                    "Strength": colored_span(h["strength"], "#008a16" if h["strength"] != "Moderate" else "#ff6b00"),
                }
            )
        hvn_df = pd.DataFrame(rows)
        st.markdown(dataframe_to_html_table(hvn_df), unsafe_allow_html=True)
        st.caption("Sorted by price (high to low)")


def render_profile(profile, distribution_bins: int, hvn_selection: str, min_volume_percentile: int):
    analogs = profile["analogs"]
    selected_hvns = filter_hvns(profile["hvns"], hvn_selection, min_volume_percentile)

    # One profile page: no internal tab switcher. Similar setup outcomes is the default/primary view.
    st.markdown('<div style="font-size:1.3rem;font-weight:800;color:#ff0000;border-bottom:3px solid #ff0000;display:inline-block;padding-bottom:10px;margin-bottom:10px;">Similar setup outcomes</div>', unsafe_allow_html=True)

    fig = build_analog_chart(profile, selected_hvns)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

    if not analogs.empty:
        render_metric_strip(analogs)

        top_row = st.columns([1.35, 1.0])
        with top_row[0]:
            st.markdown('<div class="big-section-title">Distribution of 5-Day Dollar Change <span style="color:#94a3b8;font-size:1.25rem;">ⓘ</span></div>', unsafe_allow_html=True)
        with top_row[1]:
            distribution_bins = st.slider(
                "Number of bins",
                min_value=4,
                max_value=24,
                value=int(distribution_bins),
                step=1,
                key=f"dist_bins_{profile['ticker']}",
            )

        fig = build_distribution_chart(analogs, distribution_bins)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

        render_analogs_table(analogs)
        st.download_button(
            "Download similar setup results",
            data=analogs.to_csv(index=False),
            file_name=f"{profile['ticker']}_similar_setups.csv",
            mime="text/csv",
        )
    else:
        st.info("No similar historical setup was found with the current filters. Try widening the tolerance or removing an optional filter.")

    render_hvn_section(profile, hvn_selection, min_volume_percentile)


def compare_profiles_table(profiles: List[dict]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        m = p["metrics"]
        rs_key = [k for k in m.keys() if k.startswith("Relative strength")][0]
        analogs = p["analogs"]
        rows.append(
            {
                "Ticker": p["ticker"],
                "Name": p.get("stock_name", p["ticker"]),
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
                "Change range": "N/A" if analogs.empty else f"{money0(analogs['Dollar_Change_5D'].min())} to {money0(analogs['Dollar_Change_5D'].max())}",
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# Streamlit app
# ----------------------------


def main():
    st.set_page_config(page_title="Stock Setup Profiler", layout="wide")
    inject_custom_css()
    st.title("Stock Setup Profiler")
    st.caption("Find historical matches and outcomes for current market conditions")

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

        st.header("Distribution")
        distribution_bins = st.slider("Default number of bins", 4, 24, 12, 1)

        st.header("HVN settings")
        hvn_selection = st.selectbox("HVN Selection", ["Top 10 by Volume", "Top 5 by Volume", "Minimum Volume Percentile"], index=0)
        min_volume_percentile = st.slider("Minimum Volume Percentile", 50, 99, 85, 1)
        hvn_bins = st.slider("HVN price bins", 60, 240, 140, 10)
        hvn_decay_days = st.slider("Node Decay (Days)", 30, 365, 180, 5)
        st.session_state["hvn_decay_days"] = hvn_decay_days

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
                            hvn_bins=hvn_bins,
                            hvn_decay_days=hvn_decay_days,
                        )
                    )

        st.subheader("Side-by-side setup comparison")
        criteria = [f"Compression percentile +/- {compression_tolerance_pp} pts"]
        if use_clv:
            criteria.append(f"CLV trend +/- {clv_tolerance:.2f}")
        if use_volume:
            criteria.append(f"Volume ratio +/- {volume_tolerance_pct}%")
        if use_rs:
            criteria.append(f"Relative strength +/- {rs_tolerance_pp} pts")

        st.markdown(
            f"""
            <div style="font-size:1.05rem;color:#64748b;margin-top:-8px;margin-bottom:10px;">
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
            render_profile(profiles[0], distribution_bins, hvn_selection, min_volume_percentile)
        else:
            tab_labels = [p["ticker"] for p in profiles]
            tabs = st.tabs(tab_labels)
            for tab, profile in zip(tabs, profiles):
                with tab:
                    render_profile(profile, distribution_bins, hvn_selection, min_volume_percentile)

    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
