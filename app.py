import html
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



@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_display_name(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if not ticker:
        return ""

    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return ""

    if not isinstance(info, dict):
        return ""

    # Prefer company-style names for chart titles. Fall back quietly if yfinance
    # does not return name metadata for the ticker.
    for key in ["longName", "shortName", "displayName"]:
        value = info.get(key)
        if value and str(value).strip().upper() != ticker:
            return str(value).strip()

    return ""


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

    df["ATR10"] = atr(df, 10)
    df["ATR50"] = atr(df, 50)
    df["Compression_Ratio"] = df["ATR10"] / (df["ATR50"] + 1e-9)

    df["Volume_20_Mean"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_20_Mean"] + 1e-9)

    df["Future_Close_5D"] = df["Close"].shift(-5)
    df["Forward_Return_5D"] = (df["Future_Close_5D"] / df["Close"] - 1) * 100

    if benchmark_df is not None and not benchmark_df.empty:
        aligned = df[["Close"]].rename(columns={"Close": "Stock_Close"}).join(
            benchmark_df[["Close"]].rename(columns={"Close": "Benchmark_Close"}),
            how="left",
        )
        df["RS_60"] = (
            aligned["Stock_Close"].pct_change(60)
            - aligned["Benchmark_Close"].pct_change(60)
        ) * 100
        df["RS_20"] = (
            aligned["Stock_Close"].pct_change(20)
            - aligned["Benchmark_Close"].pct_change(20)
        ) * 100
    else:
        df["RS_60"] = np.nan
        df["RS_20"] = np.nan

    return df


def compute_hvns(df, bins=140, top_nodes=10, decay_days=180, min_volume_percentile=85):
    if df.empty:
        return []

    min_price = df["Low"].min()
    max_price = df["High"].max()
    if not np.isfinite(min_price) or not np.isfinite(max_price) or max_price <= min_price:
        return []

    bin_edges = np.linspace(min_price, max_price, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume_profile = np.zeros(bins)
    touch_counts = np.zeros(bins)
    latest_date = df.index[-1]

    for date, row in df.iterrows():
        low = float(row["Low"])
        high = float(row["High"])
        close = float(row["Close"])
        volume = float(row["Volume"])

        if high <= low or volume <= 0 or not np.isfinite(close):
            continue

        age_days = max((latest_date - date).days, 0)
        typical_price = (high + low + close) / 3
        dollar_volume = volume * typical_price
        weighted_volume = dollar_volume * np.exp(-age_days / decay_days)

        touched_bins = np.where((bin_edges[:-1] <= high) & (bin_edges[1:] >= low))[0]
        if len(touched_bins) == 0:
            continue

        # Daily OHLCV does not contain true volume-at-price. Instead of spreading
        # the day's volume evenly across the whole high-low range, bias it toward
        # the close. This uses close location value behavior as a practical proxy:
        # a close near the high concentrates more volume toward the upper range,
        # while a close near the low concentrates more volume toward the lower range.
        touched_centers = bin_centers[touched_bins]
        day_range = max(high - low, 1e-9)
        close_anchor = np.clip(close, low, high)
        sigma = max(day_range * 0.28, (bin_edges[1] - bin_edges[0]) * 1.5)
        distance_weights = np.exp(-0.5 * ((touched_centers - close_anchor) / sigma) ** 2)
        base_weights = np.ones_like(distance_weights) * 0.20
        allocation_weights = base_weights + distance_weights
        allocation_weights = allocation_weights / (allocation_weights.sum() + 1e-9)

        volume_profile[touched_bins] += weighted_volume * allocation_weights
        touch_counts[touched_bins] += 1

    active_profile = volume_profile[volume_profile > 0]
    if len(active_profile) == 0:
        return []
    percentile_cutoff = np.percentile(active_profile, min_volume_percentile)

    # Identify HVNs as tight high-volume zones, not single price bins. Start with
    # local peaks above the percentile cutoff, then expand each peak into a
    # surrounding cluster while nearby bins remain meaningful relative to that
    # peak. Overlapping clusters are merged so adjacent high-volume prices become
    # one HVN zone with a weighted-average price.
    peak_indices = []
    for i in range(1, len(volume_profile) - 1):
        if (
            volume_profile[i] >= percentile_cutoff
            and volume_profile[i] > volume_profile[i - 1]
            and volume_profile[i] > volume_profile[i + 1]
        ):
            peak_indices.append(i)

    if not peak_indices:
        peak_indices = [int(i) for i in np.argsort(volume_profile)[::-1] if volume_profile[i] > 0][:top_nodes]

    cluster_threshold_ratio = 0.60
    max_cluster_bins = 12
    raw_clusters = []

    for peak_idx in peak_indices:
        peak_volume = volume_profile[peak_idx]
        if peak_volume <= 0:
            continue

        cluster_cutoff = max(percentile_cutoff, peak_volume * cluster_threshold_ratio)
        left = peak_idx
        right = peak_idx

        while left > 0 and volume_profile[left - 1] >= cluster_cutoff:
            left -= 1
        while right < bins - 1 and volume_profile[right + 1] >= cluster_cutoff:
            right += 1

        # If the peak is extremely sharp, include the stronger immediate neighbor
        # where available so the output still represents a small zone rather than
        # a single-bin price point.
        if left == right:
            left_neighbor = volume_profile[left - 1] if left > 0 else -1
            right_neighbor = volume_profile[right + 1] if right < bins - 1 else -1
            if left_neighbor >= right_neighbor and left > 0 and left_neighbor > 0:
                left -= 1
            elif right < bins - 1 and right_neighbor > 0:
                right += 1

        if right - left + 1 > max_cluster_bins:
            half_width = max_cluster_bins // 2
            left = max(0, peak_idx - half_width)
            right = min(bins - 1, left + max_cluster_bins - 1)
            left = max(0, right - max_cluster_bins + 1)

        raw_clusters.append((left, right))

    if not raw_clusters:
        return []

    raw_clusters = sorted(raw_clusters)
    merged_clusters = []
    for left, right in raw_clusters:
        if not merged_clusters or left > merged_clusters[-1][1] + 1:
            merged_clusters.append([left, right])
        else:
            merged_clusters[-1][1] = max(merged_clusters[-1][1], right)

    total_volume = volume_profile.sum() + 1e-9
    cluster_rows = []
    for left, right in merged_clusters:
        idx = np.arange(left, right + 1)
        cluster_volumes = volume_profile[idx]
        cluster_volume = float(cluster_volumes.sum())
        if cluster_volume <= 0:
            continue

        cluster_touches = int(np.max(touch_counts[idx]))
        avg_price = float(np.average(bin_centers[idx], weights=cluster_volumes))
        zone_low = float(bin_edges[left])
        zone_high = float(bin_edges[right + 1])
        cluster_rows.append(
            {
                "price": avg_price,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "weighted_volume": cluster_volume,
                "percent_total": float(cluster_volume / total_volume * 100),
                "touches": cluster_touches,
            }
        )

    # Rank zones by a blended score, not raw weighted volume alone. Raw volume
    # can over-rank stale zones after a major repricing. The blended score keeps
    # volume as the primary factor, but gives meaningful credit to zones that are
    # recent and closer to the current market price.
    latest_close = float(df.iloc[-1]["Close"])
    max_cluster_volume = max((row["weighted_volume"] for row in cluster_rows), default=1.0)
    latest_ts = df.index[-1]

    for row in cluster_rows:
        volume_score = row["weighted_volume"] / (max_cluster_volume + 1e-9)

        zone_mask = (df["High"] >= row["zone_low"]) & (df["Low"] <= row["zone_high"])
        if zone_mask.any():
            most_recent_touch = df.loc[zone_mask].index[-1]
            days_since_touch = max((latest_ts - most_recent_touch).days, 0)
            recency_score = float(np.exp(-days_since_touch / max(decay_days, 1)))
        else:
            recency_score = 0.0

        distance_from_current = abs(row["price"] - latest_close) / (latest_close + 1e-9)
        proximity_score = float(np.exp(-distance_from_current / 0.35))

        row["hvn_score"] = (
            0.60 * volume_score
            + 0.25 * recency_score
            + 0.15 * proximity_score
        )

    cluster_rows = sorted(cluster_rows, key=lambda x: x["hvn_score"], reverse=True)[:top_nodes]

    out = []
    for rank, row in enumerate(cluster_rows, start=1):
        strength = "Very Strong" if rank <= 2 else "Strong" if rank <= 5 else "Moderate"
        row = dict(row)
        row["rank"] = rank
        row["strength"] = strength
        row.pop("hvn_score", None)
        out.append(row)

    return sorted(out, key=lambda x: x["price"], reverse=True)

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
        current_required.append("RS_60")

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
        candidate_required.append("RS_60")

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
        rs_low = current["RS_60"] - rs_tolerance_pp
        rs_high = current["RS_60"] + rs_tolerance_pp
        mask &= candidates["RS_60"].between(rs_low, rs_high)

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

    return out[
        [
            "Date",
            "Close",
            "Future_Close_5D",
            "Dollar_Change_5D",
            "Forward_Return_5D",
            "Compression_Ratio",
            "CLV_Trend",
            "RS_20",
            "Volume_Ratio",
        ]
    ]


def build_profile(
    ticker: str,
    benchmark_df: pd.DataFrame,
    period: str,
    compression_tolerance_ratio: float,
    selected_filters: List[str],
    clv_tolerance: float,
    volume_tolerance_pct: float,
    rs_tolerance_pp: float,
):
    raw = download_ohlcv(ticker, period)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if len(raw) < 320:
        raise ValueError(f"{ticker} needs at least ~320 daily bars.")

    df = add_profile_columns(raw, benchmark_df)
    df = df.dropna(subset=["Compression_Ratio", "CLV_Trend", "Volume_Ratio"])

    ticker_upper = ticker.upper()

    return {
        "ticker": ticker_upper,
        "stock_name": get_ticker_display_name(ticker_upper),
        "df": df,
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
        .red {
            color: red !important;
        }
        .green {
            color: green !important;
        }
        .outcome-card {
            border: 1px solid #dfe5ef;
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
            background: white;
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
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_active_hvns(profile, hvn_count, hvn_decay_days, min_volume_percentile):
    return compute_hvns(
        profile["df"].loc[profile["df"].index >= (profile["df"].index[-1] - pd.Timedelta(days=365))],
        top_nodes=hvn_count,
        decay_days=hvn_decay_days,
        min_volume_percentile=min_volume_percentile,
    )


def build_analog_chart(profile, hvns):
    analogs = profile["analogs"]
    ticker = profile["ticker"]
    stock_name = str(profile.get("stock_name") or "").strip()
    title_name = f"{ticker} ({stock_name})" if stock_name and stock_name.upper() != ticker.upper() else ticker
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
    for h in hvns[:3]:
        min_y = min(min_y, h["price"])
        max_y = max(max_y, h["price"])
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

    for h in hvns[:3]:
        ax.axhline(h["price"], linestyle="--", linewidth=1.4, color="green", alpha=0.85)
        label_items.append(
            {
                "y": h["price"],
                "text": money2(h["price"]),
                "color": "green",
                "ha": "left",
                "x": label_x,
                "anchor_x": max_date,
            }
        )

    # Prevent current-close and HVN labels from sitting on top of one another.
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

    if hvns:
        ax.plot([], [], linestyle="--", color="green", label="Top 3 HVNs")

    label_y_values = [item.get("display_y", item["y"]) for item in label_items]
    adjusted_min_y = min(min_y, min(label_y_values))
    adjusted_max_y = max(max_y, max(label_y_values))
    adjusted_span = max(adjusted_max_y - adjusted_min_y, 1)
    ax.set_ylim(adjusted_min_y - adjusted_span * 0.08, adjusted_max_y + adjusted_span * 0.08)
    ax.set_title(
        f"{title_name}: Historical Similar ATR10/ATR50 Compression Setups — Price 5 Days Later",
        fontsize=20,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Analog Date", fontsize=21, fontweight="bold")
    ax.set_ylabel("Share Price 5 Days Later", fontsize=21, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(min_date - pd.Timedelta(days=x_padding_days), label_x + pd.Timedelta(days=x_padding_days))
    ax.grid(True, alpha=0.24)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        frameon=True,
        fontsize=16,
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


def render_summary_metrics(profile):
    analogs = profile["analogs"].copy()
    latest = profile["df"].iloc[-1]

    st.markdown('<div class="section-title">Summary of Analogs</div>', unsafe_allow_html=True)

    rows = [
        "Number of Analogs",
        "Compression",
        "Relative Strength",
        "CLV",
        "Volume Support",
        "Avg $ Change",
        "% Win",
        "$ Change Range",
    ]
    columns = [
        "Metric",
        "Current Setup",
        "Avg Positive Setup",
        "Avg Negative Setup",
        "Overall Avg Setup",
    ]

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
                "compression": np.nan,
                "rs": np.nan,
                "clv": np.nan,
                "volume": np.nan,
                "avg_change": np.nan,
                "win_pct": np.nan,
                "range": "",
            }
        return {
            "count": len(subset),
            "compression": subset["Compression_Ratio"].mean(),
            "rs": subset["RS_20"].mean(),
            "clv": subset["CLV_Trend"].mean(),
            "volume": subset["Volume_Ratio"].mean(),
            "avg_change": subset["Dollar_Change_5D"].mean(),
            "win_pct": (subset["Dollar_Change_5D"] > 0).mean() * 100,
            "range": _fmt_range(subset["Dollar_Change_5D"]),
        }

    positives = analogs[analogs["Dollar_Change_5D"] > 0] if not analogs.empty else analogs
    negatives = analogs[analogs["Dollar_Change_5D"] <= 0] if not analogs.empty else analogs

    positive_stats = _subset_stats(positives)
    negative_stats = _subset_stats(negatives)
    overall_stats = _subset_stats(analogs)

    raw_rows = [
        {
            "Metric": "Number of Analogs",
            "Current Setup": "",
            "Avg Positive Setup": positive_stats["count"],
            "Avg Negative Setup": negative_stats["count"],
            "Overall Avg Setup": overall_stats["count"],
        },
        {
            "Metric": "Compression",
            "Current Setup": latest.get("Compression_Ratio", np.nan),
            "Avg Positive Setup": positive_stats["compression"],
            "Avg Negative Setup": negative_stats["compression"],
            "Overall Avg Setup": overall_stats["compression"],
        },
        {
            "Metric": "Relative Strength",
            "Current Setup": latest.get("RS_20", np.nan),
            "Avg Positive Setup": positive_stats["rs"],
            "Avg Negative Setup": negative_stats["rs"],
            "Overall Avg Setup": overall_stats["rs"],
        },
        {
            "Metric": "CLV",
            "Current Setup": latest.get("CLV_Trend", np.nan),
            "Avg Positive Setup": positive_stats["clv"],
            "Avg Negative Setup": negative_stats["clv"],
            "Overall Avg Setup": overall_stats["clv"],
        },
        {
            "Metric": "Volume Support",
            "Current Setup": latest.get("Volume_Ratio", np.nan),
            "Avg Positive Setup": positive_stats["volume"],
            "Avg Negative Setup": negative_stats["volume"],
            "Overall Avg Setup": overall_stats["volume"],
        },
        {
            "Metric": "Avg $ Change",
            "Current Setup": "",
            "Avg Positive Setup": positive_stats["avg_change"],
            "Avg Negative Setup": negative_stats["avg_change"],
            "Overall Avg Setup": overall_stats["avg_change"],
        },
        {
            "Metric": "% Win",
            "Current Setup": "",
            "Avg Positive Setup": positive_stats["win_pct"],
            "Avg Negative Setup": negative_stats["win_pct"],
            "Overall Avg Setup": overall_stats["win_pct"],
        },
        {
            "Metric": "$ Change Range",
            "Current Setup": "",
            "Avg Positive Setup": positive_stats["range"],
            "Avg Negative Setup": negative_stats["range"],
            "Overall Avg Setup": overall_stats["range"],
        },
    ]

    display = pd.DataFrame(raw_rows, columns=columns).astype(object)
    for idx, row in display.iterrows():
        metric = row["Metric"]
        for col in columns[1:]:
            value = row[col]
            if metric == "Number of Analogs":
                display.at[idx, col] = _fmt_count(value)
            elif metric == "Relative Strength":
                display.at[idx, col] = _fmt_number(value, 1)
            elif metric == "Volume Support":
                formatted = _fmt_number(value, 2)
                display.at[idx, col] = f"{formatted}x" if formatted else ""
            elif metric in ["Compression", "CLV"]:
                display.at[idx, col] = _fmt_number(value, 2)
            elif metric == "Avg $ Change":
                display.at[idx, col] = _fmt_money(value)
            elif metric == "% Win":
                display.at[idx, col] = _fmt_pct(value)
            elif metric == "$ Change Range":
                display.at[idx, col] = value

    def _style_summary_table(data):
        styles = pd.DataFrame("font-weight: 850;", index=data.index, columns=data.columns)
        for col in ["Avg Positive Setup", "Avg Negative Setup", "Overall Avg Setup"]:
            for idx, row in data.iterrows():
                metric = row["Metric"]
                raw_value = raw_rows[idx].get(col, "")
                if metric in ["Avg $ Change", "$ Change Range"]:
                    if col == "Avg Positive Setup":
                        styles.at[idx, col] += " color: green;"
                    elif col == "Avg Negative Setup":
                        styles.at[idx, col] += " color: red;"
                    elif metric == "Avg $ Change" and raw_value not in ["", None] and not pd.isna(raw_value):
                        styles.at[idx, col] += " color: green;" if float(raw_value) > 0 else " color: red;" if float(raw_value) < 0 else ""
                elif col == "Avg Positive Setup":
                    styles.at[idx, col] += " color: green;"
                elif col == "Avg Negative Setup":
                    styles.at[idx, col] += " color: red;"
        return styles

    styled = display.style.apply(_style_summary_table, axis=None)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=345)

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

    analogs = analogs.rename(columns={"Compression_Ratio": "Compression"})

    styled = analogs.style.format(
        {
            "Close": "{:.2f}",
            "Future_Close_5D": "{:.2f}",
            "Dollar_Change_5D": "{:.2f}",
            "Forward_Return_5D": "{:.2f}%",
            "Compression": "{:.2f}",
            "CLV_Trend": "{:.2f}",
            "RS_20": "{:.1f}",
        }
    ).map(
        lambda v: "color: green; font-weight: 800;" if float(v) > 0 else "color: red; font-weight: 800;",
        subset=["Dollar_Change_5D", "Forward_Return_5D"],
    )

    visible_rows = min(len(analogs), 10)
    height = 38 * (visible_rows + 1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=height)
    st.caption(f"Showing all {len(analogs)} entries. Scroll inside the table to view rows beyond the first 10.")


def get_hvn_settings_from_state(key_prefix="main"):
    selection_key = f"{key_prefix}_hvn_selection"
    min_percentile_key = f"{key_prefix}_hvn_min_volume_percentile"
    decay_key = f"{key_prefix}_hvn_decay_days"

    hvn_selection = st.session_state.get(selection_key, "Top 10 by Volume")
    min_volume_percentile = st.session_state.get(min_percentile_key, 85)
    hvn_decay_days = st.session_state.get(decay_key, 180)

    try:
        hvn_count = int(str(hvn_selection).split()[1])
    except Exception:
        hvn_count = 10

    return hvn_count, hvn_decay_days, min_volume_percentile


def render_hvn_controls(key_prefix="main"):
    hvn_selection = st.selectbox(
        "HVN Selection",
        ["Top 5 by Volume", "Top 10 by Volume", "Top 20 by Volume"],
        index=1,
        key=f"{key_prefix}_hvn_selection",
    )
    min_volume_percentile = st.slider(
        "Minimum Volume Percentile",
        50,
        99,
        85,
        1,
        key=f"{key_prefix}_hvn_min_volume_percentile",
    )
    hvn_decay_days = st.slider(
        "Node Decay (Days)",
        30,
        365,
        180,
        1,
        key=f"{key_prefix}_hvn_decay_days",
    )
    st.info(
        "Node decay reduces the influence of older price activity. Lower values focus on recent data; "
        "higher values include more historical data."
    )

    hvn_count = int(hvn_selection.split()[1])
    return hvn_count, hvn_decay_days, min_volume_percentile


def render_hvn_table(hvns):
    hvn_table = pd.DataFrame(hvns)
    if hvn_table.empty:
        st.info("No HVNs were found.")
        return

    hvn_table = hvn_table.rename(
        columns={
            "price": "HVN Avg Price",
            "zone_low": "Zone Low",
            "zone_high": "Zone High",
            "weighted_volume": "Weighted Dollar Volume",
            "percent_total": "Percent of Total",
            "touches": "Touches",
            "strength": "Strength",
        }
    )[
        [
            "HVN Avg Price",
            "Zone Low",
            "Zone High",
            "Weighted Dollar Volume",
            "Percent of Total",
            "Touches",
            "Strength",
        ]
    ]

    styled = hvn_table.style.format(
        {
            "HVN Avg Price": "${:,.2f}",
            "Zone Low": "${:,.2f}",
            "Zone High": "${:,.2f}",
            "Weighted Dollar Volume": "${:,.0f}",
            "Percent of Total": "{:.2f}%",
            "Touches": "{:,.0f}",
        }
    ).map(
        lambda v: "color: green; font-weight: 800;" if v in ["Very Strong", "Strong"] else "color: #ff6a00; font-weight: 800;",
        subset=["Strength"],
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=365)
    st.caption("Sorted by HVN average price (high to low)")


def render_hvn_section(profile, key_prefix="main"):
    st.markdown('<div class="section-title">HVN (High Volume Nodes) ⓘ</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 2.55])
    with left:
        hvn_count, hvn_decay_days, min_volume_percentile = render_hvn_controls(key_prefix=key_prefix)

    active_hvns = get_active_hvns(profile, hvn_count, hvn_decay_days, min_volume_percentile)

    with right:
        render_hvn_table(active_hvns)


def render_major_events_section(profile, key_prefix="main"):
    st.markdown('<div class="section-title">Major Events ⓘ</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 2.55])
    with left:
        threshold_pct = st.slider(
            "Major event threshold (+/- %)",
            1,
            75,
            10,
            1,
            key=f"{key_prefix}_major_event_threshold",
        )
        st.info(
            "Major events are dates where the close on the next trading day versus "
            "the close on the prior trading day moved by at least the selected threshold. "
            "Setup metrics shown for historical events are from the prior trading day."
        )

    df = profile["df"].copy()
    df["Dollar_Change_Event_Window"] = df["Close"].shift(-1) - df["Close"].shift(1)
    df["Percent_Change_Event_Window"] = (df["Close"].shift(-1) / df["Close"].shift(1) - 1) * 100

    metric_cols = ["Compression_Ratio", "CLV_Trend", "RS_20", "Volume_Ratio"]
    setup_cols = []
    for col in metric_cols:
        setup_col = f"Setup_{col}"
        df[setup_col] = df[col].shift(1)
        setup_cols.append(setup_col)

    events = df.dropna(
        subset=["Dollar_Change_Event_Window", "Percent_Change_Event_Window"] + setup_cols
    ).copy()
    events = events[events["Percent_Change_Event_Window"].abs() >= threshold_pct]
    events = events.sort_index(ascending=False)

    latest = df.iloc[-1]
    current_vector = pd.Series(
        {
            "Compression": latest.get("Compression_Ratio", np.nan),
            "CLV": latest.get("CLV_Trend", np.nan),
            "Relative Strength": latest.get("RS_20", np.nan),
            "Volume Support": latest.get("Volume_Ratio", np.nan),
        }
    )

    event_metric_map = {
        "Setup_Compression_Ratio": "Compression",
        "Setup_CLV_Trend": "CLV",
        "Setup_RS_20": "Relative Strength",
        "Setup_Volume_Ratio": "Volume Support",
    }

    def _similarity_pct(current_values: pd.Series, target_values: pd.Series, reference_values: pd.DataFrame):
        if target_values is None or target_values.empty or reference_values.empty:
            return np.nan
        std = reference_values.std(ddof=0).replace(0, np.nan)
        std = std.fillna(reference_values.abs().mean()).replace(0, 1.0).fillna(1.0)
        aligned_current = current_values[target_values.index].astype(float)
        aligned_target = target_values.astype(float)
        z_distance = ((aligned_current - aligned_target) / std[target_values.index]).pow(2).mean() ** 0.5
        return float(np.exp(-z_distance) * 100)

    event_metrics = pd.DataFrame()
    if not events.empty:
        event_metrics = events[list(event_metric_map.keys())].rename(columns=event_metric_map)

    positive_events = events[events["Percent_Change_Event_Window"] > 0]
    negative_events = events[events["Percent_Change_Event_Window"] < 0]

    positive_avg = pd.Series(dtype=float)
    negative_avg = pd.Series(dtype=float)
    positive_similarity = np.nan
    negative_similarity = np.nan

    if not positive_events.empty:
        positive_avg = positive_events[list(event_metric_map.keys())].rename(columns=event_metric_map).mean()
        positive_similarity = _similarity_pct(current_vector, positive_avg, event_metrics)

    if not negative_events.empty:
        negative_avg = negative_events[list(event_metric_map.keys())].rename(columns=event_metric_map).mean()
        negative_similarity = _similarity_pct(current_vector, negative_avg, event_metrics)

    positive_event_count = int(len(positive_events))
    negative_event_count = int(len(negative_events))
    positive_avg_dollar_change = (
        float(positive_events["Dollar_Change_Event_Window"].mean())
        if positive_event_count > 0
        else np.nan
    )
    negative_avg_dollar_change = (
        float(negative_events["Dollar_Change_Event_Window"].mean())
        if negative_event_count > 0
        else np.nan
    )

    def _series_value(series, key):
        if series is None or series.empty or key not in series.index:
            return np.nan
        return series[key]

    def _metric_avg(series, key):
        if series is None or series.empty or key not in series.index:
            return np.nan
        return series[key]

    def _closest_profile(metric_name: str):
        current_value = current_vector.get(metric_name, np.nan)
        positive_value = _metric_avg(positive_avg, metric_name)
        negative_value = _metric_avg(negative_avg, metric_name)

        if pd.isna(current_value) or (pd.isna(positive_value) and pd.isna(negative_value)):
            return ""
        if pd.isna(positive_value):
            return "Negative"
        if pd.isna(negative_value):
            return "Positive"

        positive_distance = abs(float(current_value) - float(positive_value))
        negative_distance = abs(float(current_value) - float(negative_value))
        if abs(positive_distance - negative_distance) < 1e-9:
            return "Tie"
        return "Positive" if positive_distance < negative_distance else "Negative"

    setup_summary = pd.DataFrame(
        [
            {
                "Metric": "Compression",
                "Current Setup": current_vector["Compression"],
                "Avg Positive Setup": _metric_avg(positive_avg, "Compression"),
                "Avg Negative Setup": _metric_avg(negative_avg, "Compression"),
                "Closest": _closest_profile("Compression"),
            },
            {
                "Metric": "CLV",
                "Current Setup": current_vector["CLV"],
                "Avg Positive Setup": _metric_avg(positive_avg, "CLV"),
                "Avg Negative Setup": _metric_avg(negative_avg, "CLV"),
                "Closest": _closest_profile("CLV"),
            },
            {
                "Metric": "Relative Strength",
                "Current Setup": current_vector["Relative Strength"],
                "Avg Positive Setup": _metric_avg(positive_avg, "Relative Strength"),
                "Avg Negative Setup": _metric_avg(negative_avg, "Relative Strength"),
                "Closest": _closest_profile("Relative Strength"),
            },
            {
                "Metric": "Volume Support",
                "Current Setup": current_vector["Volume Support"],
                "Avg Positive Setup": _metric_avg(positive_avg, "Volume Support"),
                "Avg Negative Setup": _metric_avg(negative_avg, "Volume Support"),
                "Closest": _closest_profile("Volume Support"),
            },
            {
                "Metric": "Event Count",
                "Current Setup": "",
                "Avg Positive Setup": positive_event_count if positive_event_count > 0 else np.nan,
                "Avg Negative Setup": negative_event_count if negative_event_count > 0 else np.nan,
                "Closest": "",
            },
            {
                "Metric": "Avg $ Change",
                "Current Setup": "",
                "Avg Positive Setup": positive_avg_dollar_change,
                "Avg Negative Setup": negative_avg_dollar_change,
                "Closest": "",
            },
            {
                "Metric": "Similarity",
                "Current Setup": "",
                "Avg Positive Setup": positive_similarity,
                "Avg Negative Setup": negative_similarity,
                "Closest": (
                    "Positive"
                    if pd.notna(positive_similarity)
                    and (pd.isna(negative_similarity) or positive_similarity > negative_similarity)
                    else "Negative"
                    if pd.notna(negative_similarity)
                    and (pd.isna(positive_similarity) or negative_similarity > positive_similarity)
                    else "Tie"
                    if pd.notna(positive_similarity) and pd.notna(negative_similarity)
                    else ""
                ),
            },
        ]
    )

    def _fmt_money(value):
        if value == "" or pd.isna(value):
            return ""
        return signed_money0(value)

    def _fmt_pct(value):
        if value == "" or pd.isna(value):
            return ""
        return signed_pct1(value)

    def _fmt_number(value, decimals=1):
        if value == "" or pd.isna(value):
            return ""
        return f"{float(value):.{decimals}f}"

    def _fmt_volume_support(value):
        if value == "" or pd.isna(value):
            return ""
        return f"{float(value):.2f}x"

    def _fmt_similarity(value):
        if value == "" or pd.isna(value):
            return ""
        return f"{float(value):.1f}%"

    def _fmt_setup_cell(value, metric_name):
        if value == "" or pd.isna(value):
            return ""
        if metric_name == "Volume Support":
            return _fmt_volume_support(value)
        if metric_name == "Relative Strength":
            return _fmt_number(value, 1)
        if metric_name == "Event Count":
            return f"{int(float(value)):,}"
        if metric_name == "Avg $ Change":
            return signed_money0(value)
        if metric_name == "Similarity":
            return _fmt_similarity(value)
        return _fmt_number(value, 2)

    def _summary_row_style(row):
        styles = ["font-weight: 850;" for _ in row]
        columns = list(row.index)
        for idx, col in enumerate(columns):
            if col == "Avg Positive Setup":
                styles[idx] += " color: green;"
            elif col == "Avg Negative Setup":
                styles[idx] += " color: red;"
            elif col == "Closest":
                closest_value = str(row.get("Closest", ""))
                if closest_value == "Positive":
                    styles[idx] += " color: green;"
                elif closest_value == "Negative":
                    styles[idx] += " color: red;"
        return styles

    summary_styled = setup_summary.style.format(
        {
            "Current Setup": lambda v: _fmt_setup_cell(v, setup_summary.loc[v.name, "Metric"]) if hasattr(v, "name") else v,
            "Avg Positive Setup": lambda v: v,
            "Avg Negative Setup": lambda v: v,
        }
    )

    # Pandas 2.x / Python 3.14 can reject assigning formatted strings into
    # float columns. Build a separate object-typed display table before styling.
    setup_summary_display = setup_summary.copy().astype(object)
    for idx, row in setup_summary.iterrows():
        metric_name = row["Metric"]
        for col in ["Current Setup", "Avg Positive Setup", "Avg Negative Setup"]:
            setup_summary_display.at[idx, col] = _fmt_setup_cell(row[col], metric_name)

    summary_styled = setup_summary_display.style.apply(_summary_row_style, axis=1)

    detail_rows = []
    for event_date, row in events.iterrows():
        detail_rows.append(
            {
                "Event Date": event_date.date(),
                "$ Change": row["Dollar_Change_Event_Window"],
                "% Change": row["Percent_Change_Event_Window"],
                "Compression": row["Setup_Compression_Ratio"],
                "CLV": row["Setup_CLV_Trend"],
                "Relative Strength": row["Setup_RS_20"],
                "Volume Support": row["Setup_Volume_Ratio"],
            }
        )

    display_table = pd.DataFrame(detail_rows)
    if display_table.empty:
        display_table = pd.DataFrame(
            columns=[
                "Event Date",
                "$ Change",
                "% Change",
                "Compression",
                "CLV",
                "Relative Strength",
                "Volume Support",
            ]
        )

    detailed_styled = display_table.style.format(
        {
            "$ Change": _fmt_money,
            "% Change": _fmt_pct,
            "Compression": lambda v: _fmt_number(v, 2),
            "CLV": lambda v: _fmt_number(v, 2),
            "Relative Strength": lambda v: _fmt_number(v, 1),
            "Volume Support": _fmt_volume_support,
        }
    ).map(
        lambda v: "color: green; font-weight: 800;" if isinstance(v, (int, float, np.number)) and float(v) > 0 else (
            "color: red; font-weight: 800;" if isinstance(v, (int, float, np.number)) and float(v) < 0 else ""
        ),
        subset=["$ Change", "% Change"],
    )

    visible_rows = min(max(len(display_table), 1), 10)
    height = 38 * (visible_rows + 1)
    with right:
        st.markdown("**Setup Profile Summary**")
        st.dataframe(summary_styled, use_container_width=True, hide_index=True, height=245)
        st.caption(
            "Similarity uses a standardized vector distance across Compression, CLV, Relative Strength, and Volume Support. "
            "Higher percentages mean the current setup is closer to the historical average setup."
        )
        st.markdown("**Major Event Detail**")
        st.dataframe(detailed_styled, use_container_width=True, hide_index=True, height=height)
        event_count = len(display_table)
        st.caption(
            f"{event_count} major events where next-day close versus prior-day close moved +/- {threshold_pct}% or greater. "
            "Historical event metrics show the setup from the prior trading day. Scroll inside the table to view additional events."
        )


def render_profile(profile, key_prefix="main"):
    # HVN widgets are displayed beside the HVN table below, but Streamlit stores
    # their latest values in session state. Reading that state here lets the line
    # graph update on rerun without placing the controls above the graph.
    hvn_count, hvn_decay_days, min_volume_percentile = get_hvn_settings_from_state(key_prefix=key_prefix)
    active_hvns = get_active_hvns(profile, hvn_count, hvn_decay_days, min_volume_percentile)

    fig = build_analog_chart(profile, active_hvns)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    render_summary_metrics(profile)
    render_distribution(profile, key_prefix=key_prefix)
    render_analogs_table(profile)
    render_hvn_section(profile, key_prefix=key_prefix)
    render_major_events_section(profile, key_prefix=key_prefix)

def main():
    st.set_page_config(page_title="Stock Setup Profiler", layout="wide")
    inject_custom_css()

    st.title("Stock Setup Profiler")
    st.caption("Find historical matches and outcomes for current market conditions")

    with st.sidebar:
        st.header("Inputs (v12.7)")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        comparison_ticker = st.text_input("Second ticker for comparison", value="MSFT").strip().upper()
        benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
        period = st.selectbox("History", ["2y", "5y", "10y", "max"], index=1)

        st.header("Similarity matching")
        compression_tolerance_ratio = st.slider("Compression ratio tolerance (+/-)", 0.05, 2.00, 0.20, 0.05)
        st.caption("Compression ratio = ATR10 / ATR50. Lower values mean compression; higher values mean expansion.")

        st.markdown("**Additional matching filters**")
        use_clv = st.toggle("Match CLV trend", value=False)
        clv_tolerance = st.slider("CLV trend tolerance", 0.01, 0.50, 0.10, 0.01, disabled=not use_clv)

        use_volume = st.toggle("Match volume support", value=False)
        volume_tolerance_pct = st.slider("Volume ratio tolerance (+/- %)", 5, 100, 25, 5, disabled=not use_volume)

        use_rs = st.toggle("Match relative strength", value=False)
        rs_tolerance_pp = st.slider("Relative strength tolerance (+/- percentage points)", 1, 30, 5, 1, disabled=not use_rs)

        run_button = st.button("Run profile", type="primary")

    selected_filters = []
    if use_clv:
        selected_filters.append("CLV trend")
    if use_volume:
        selected_filters.append("Volume support")
    if use_rs:
        selected_filters.append("Relative strength")

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
        benchmark,
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
                benchmark_df = download_ohlcv(benchmark, period)
                profiles = []
                for requested_ticker in requested_tickers:
                    profiles.append(
                        build_profile(
                            ticker=requested_ticker,
                            benchmark_df=benchmark_df,
                            period=period,
                            compression_tolerance_ratio=compression_tolerance_ratio,
                            selected_filters=selected_filters,
                            clv_tolerance=clv_tolerance,
                            volume_tolerance_pct=volume_tolerance_pct,
                            rs_tolerance_pp=rs_tolerance_pp,
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
