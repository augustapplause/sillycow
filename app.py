import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import html


BENCHMARK_TICKER = "SPY"


def money0(x):
    if x is None:
        return "Not established"
    return f"${x:,.0f}"


def clv(df):
    return ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"] + 1e-9
    )


def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_hvns(df, bins=140, top_nodes=20, decay_days=180):
    min_price = df["Low"].min()
    max_price = df["High"].max()
    bin_edges = np.linspace(min_price, max_price, bins + 1)
    volume_profile = np.zeros(bins)
    latest_date = df.index[-1]

    for date, row in df.iterrows():
        low = row["Low"]
        high = row["High"]
        volume = row["Volume"]

        if high <= low or volume <= 0:
            continue

        age_days = max((latest_date - date).days, 0)
        weighted_volume = volume * np.exp(-age_days / decay_days)

        touched_bins = np.where(
            (bin_edges[:-1] <= high) &
            (bin_edges[1:] >= low)
        )[0]

        if len(touched_bins) == 0:
            continue

        volume_per_bin = weighted_volume / len(touched_bins)

        for idx in touched_bins:
            volume_profile[idx] += volume_per_bin

    peaks = []

    for i in range(1, len(volume_profile) - 1):
        if volume_profile[i] > volume_profile[i - 1] and volume_profile[i] > volume_profile[i + 1]:
            peaks.append({
                "price": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "volume": float(volume_profile[i])
            })

    if not peaks:
        for idx in np.argsort(volume_profile)[::-1][:top_nodes]:
            peaks.append({
                "price": float((bin_edges[idx] + bin_edges[idx + 1]) / 2),
                "volume": float(volume_profile[idx])
            })

    top_peaks = sorted(peaks, key=lambda x: x["volume"], reverse=True)[:top_nodes]
    return sorted(top_peaks, key=lambda x: x["price"])


def get_hvn_levels(hvns, price):
    below = [h for h in hvns if h["price"] < price]
    above = [h for h in hvns if h["price"] > price]

    nearest_below = below[-1]["price"] if below else None
    nearest_above = above[0]["price"] if above else None
    nearest_hvn = min(hvns, key=lambda h: abs(h["price"] - price))["price"]

    return nearest_hvn, nearest_above, nearest_below


def add_base_columns(df):
    df = df.copy()

    df["CLV"] = clv(df)
    df["ATR"] = atr(df)

    df["Volume_20_Mean"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_20_Mean"] + 1e-9)

    df["ATR_20_Mean"] = df["ATR"].rolling(20).mean()
    df["ATR_Ratio"] = df["ATR"] / (df["ATR_20_Mean"] + 1e-9)

    return df.dropna()


def relative_strength_model(stock_df, benchmark_df, benchmark_name="SPY"):
    stock = stock_df[["Close"]].copy().rename(columns={"Close": "Stock_Close"})
    bench = benchmark_df[["Close"]].copy().rename(columns={"Close": "Benchmark_Close"})

    combined = stock.join(bench, how="inner").dropna()

    if len(combined) < 80:
        return {
            "value": "UNAVAILABLE",
            "score": "",
            "days": "",
            "reason": "Not enough benchmark history.",
            "price_impact": "="
        }

    stock_20 = combined["Stock_Close"].iloc[-1] / combined["Stock_Close"].iloc[-21] - 1
    bench_20 = combined["Benchmark_Close"].iloc[-1] / combined["Benchmark_Close"].iloc[-21] - 1
    rs_20 = stock_20 - bench_20

    stock_60 = combined["Stock_Close"].iloc[-1] / combined["Stock_Close"].iloc[-61] - 1
    bench_60 = combined["Benchmark_Close"].iloc[-1] / combined["Benchmark_Close"].iloc[-61] - 1
    rs_60 = stock_60 - bench_60

    if rs_60 >= 0.10:
        value = "STRONG"
        impact = "+"
    elif rs_60 >= 0.03:
        value = "POSITIVE"
        impact = "+"
    elif rs_60 <= -0.10:
        value = "WEAK"
        impact = "-"
    elif rs_60 <= -0.03:
        value = "NEGATIVE"
        impact = "-"
    else:
        value = "NEUTRAL"
        impact = "="

    rs_series = combined["Stock_Close"].pct_change(20) - combined["Benchmark_Close"].pct_change(20)

    states = []
    for x in rs_series.dropna():
        if x >= 0.10:
            states.append("STRONG")
        elif x >= 0.03:
            states.append("POSITIVE")
        elif x <= -0.10:
            states.append("WEAK")
        elif x <= -0.03:
            states.append("NEGATIVE")
        else:
            states.append("NEUTRAL")

    days = 0
    if states:
        current = states[-1]
        for s in reversed(states):
            if s == current:
                days += 1
            else:
                break

    reason = (
        f"Outperforming {benchmark_name} by {rs_60 * 100:.1f}% over 60 days."
        if rs_60 > 0
        else f"Underperforming {benchmark_name} by {abs(rs_60) * 100:.1f}% over 60 days."
        if rs_60 < 0
        else f"In line with {benchmark_name} over 60 days."
    )

    return {
        "value": value,
        "score": f"{rs_60 * 100:+.1f}%",
        "days": days,
        "reason": reason,
        "price_impact": impact,
        "rs_20": float(rs_20),
        "rs_60": float(rs_60)
    }


def institutional_participation_model(df):
    latest = df.iloc[-1]

    volume_ratio = float(latest["Volume_Ratio"])
    atr_ratio = float(latest["ATR_Ratio"])
    clv_value = float(latest["CLV"])

    if volume_ratio >= 1.50 and abs(clv_value) >= 0.30:
        value = "STRONG"
        score = "3/3"
    elif volume_ratio >= 1.10 and (abs(clv_value) >= 0.20 or atr_ratio >= 1.10):
        value = "MODERATE"
        score = "2/3"
    else:
        value = "WEAK"
        score = "1/3"

    reason = []

    if volume_ratio >= 1.50:
        reason.append(f"volume strong at {volume_ratio:.2f}x 20-day average")
    elif volume_ratio >= 1.10:
        reason.append(f"volume moderately elevated at {volume_ratio:.2f}x 20-day average")
    else:
        reason.append(f"volume weak at {volume_ratio:.2f}x 20-day average")

    if clv_value > 0.30:
        reason.append("CLV shows buyer control")
    elif clv_value < -0.30:
        reason.append("CLV shows seller control")
    else:
        reason.append("CLV not strongly directional")

    if atr_ratio >= 1.10:
        reason.append(f"ATR expanding at {atr_ratio:.2f}x")
    else:
        reason.append(f"ATR not expanding meaningfully at {atr_ratio:.2f}x")

    reason.append("volume is the primary participation filter")

    return {
        "value": value,
        "score": score,
        "reason": "; ".join(reason),
        "volume_ratio": volume_ratio,
        "atr_ratio": atr_ratio,
        "clv": clv_value
    }


def supply_exhaustion_model(df, clv_5, clv_10, price):
    recent = df.tail(10)
    down_bars = recent[recent["Close"] < recent["Open"]]

    sellers_losing_control = clv_5 > clv_10 and clv_5 > -0.15

    down_volume_falling = (
        len(down_bars) >= 3
        and down_bars["Volume"].iloc[-1] < down_bars["Volume"].mean()
    )

    near_lower_range = price <= df["Close"].tail(20).quantile(0.30)

    score_num = sum([
        sellers_losing_control,
        down_volume_falling,
        near_lower_range
    ])

    if score_num >= 3:
        value = "LIKELY"
    elif score_num >= 2:
        value = "NEARING"
    else:
        value = "NOT_EVIDENT"

    reason = [
        "CLV improving versus 10-day control" if sellers_losing_control else "CLV not yet improving versus 10-day control",
        "down-volume fading" if down_volume_falling else "down-volume not fading",
        "price near lower 20-day range" if near_lower_range else "price not near lower 20-day range"
    ]

    return {
        "value": value,
        "score": f"{score_num}/3",
        "reason": "; ".join(reason),
        "sellers_losing_control": sellers_losing_control,
        "down_volume_falling": down_volume_falling,
        "near_lower_range": near_lower_range
    }


def near_term_bias_model(clv_5, volume_ratio, hvn_distance_pct, supply, institutional, relative_strength=None):
    volume_component = 0.0

    if volume_ratio > 1.10 and clv_5 > 0:
        volume_component = 0.20
    elif volume_ratio > 1.10 and clv_5 < 0:
        volume_component = -0.20

    supply_support = 0.0

    if supply["value"] == "LIKELY" and clv_5 > -0.10:
        supply_support = 0.25
    elif supply["value"] == "NEARING" and clv_5 > -0.10:
        supply_support = 0.15

    institutional_component = 0.0

    if institutional["value"] == "STRONG" and clv_5 > 0:
        institutional_component = 0.20
    elif institutional["value"] == "STRONG" and clv_5 < 0:
        institutional_component = -0.20
    elif institutional["value"] == "MODERATE" and clv_5 > 0:
        institutional_component = 0.10
    elif institutional["value"] == "MODERATE" and clv_5 < 0:
        institutional_component = -0.10

    rs_component = 0.0
    if relative_strength is not None:
        if relative_strength["value"] in ["STRONG", "POSITIVE"]:
            rs_component = 0.15
        elif relative_strength["value"] in ["WEAK", "NEGATIVE"]:
            rs_component = -0.15

    hvn_component = np.tanh(-(hvn_distance_pct / 100) * 5)

    score = (
        0.35 * clv_5
        + 0.17 * volume_component
        + 0.13 * hvn_component
        + 0.13 * supply_support
        + 0.10 * institutional_component
        + 0.12 * rs_component
    )

    if score > 0.15:
        value = "BULLISH"
        reason = "Buyers have near-term control."
    elif score < -0.15:
        value = "BEARISH"
        reason = "Sellers have near-term control."
    else:
        value = "NEUTRAL"
        reason = "Directional control is mixed."

    return {
        "value": value,
        "score": round(float(score), 3),
        "reason": reason
    }


def expected_auction_range_model(
    price,
    atr_now,
    clv_5,
    volume_ratio,
    nearest_hvn_above,
    nearest_hvn_below
):
    atr_up_distance = atr_now
    atr_down_distance = atr_now

    if nearest_hvn_above is not None:
        hvn_up_distance = max(nearest_hvn_above - price, 0)
    else:
        hvn_up_distance = atr_now * 1.50

    if nearest_hvn_below is not None:
        hvn_down_distance = max(price - nearest_hvn_below, 0)
    else:
        hvn_down_distance = atr_now * 1.50

    base_up_distance = (0.65 * atr_up_distance) + (0.35 * hvn_up_distance)
    base_down_distance = (0.65 * atr_down_distance) + (0.35 * hvn_down_distance)

    skew = 0.0
    skew_reasons = []

    if clv_5 > 0.25:
        skew += 0.25
        skew_reasons.append("CLV favors upside")
    elif clv_5 < -0.25:
        skew -= 0.25
        skew_reasons.append("CLV favors downside")
    else:
        skew_reasons.append("CLV mixed")

    if volume_ratio > 1.20 and clv_5 > 0:
        skew += 0.25
        skew_reasons.append("volume confirms upside")
    elif volume_ratio > 1.20 and clv_5 < 0:
        skew -= 0.25
        skew_reasons.append("volume confirms downside")
    else:
        skew_reasons.append("volume not strongly directional")

    if skew > 0:
        up_distance = base_up_distance * (1 + skew)
        down_distance = base_down_distance * 0.85
        reason = "Blended ATR/HVN range skewed upward; " + "; ".join(skew_reasons)
    elif skew < 0:
        up_distance = base_up_distance * 0.85
        down_distance = base_down_distance * (1 + abs(skew))
        reason = "Blended ATR/HVN range skewed downward; " + "; ".join(skew_reasons)
    else:
        up_distance = base_up_distance
        down_distance = base_down_distance
        reason = "Blended ATR/HVN range is balanced; " + "; ".join(skew_reasons)

    low = price - down_distance
    high = price + up_distance

    return {
        "value": f"{money0(low)} to {money0(high)}",
        "score": f"Skew {skew:.2f}",
        "low": float(low),
        "high": float(high),
        "skew": float(skew),
        "reason": reason
    }


def expansion_potential_model(
    df,
    price,
    atr_now,
    clv_5,
    hvn_distance_pct,
    bias,
    volume_ratio,
    supply,
    institutional,
    relative_strength
):
    compression_ratio = atr_now / (df["ATR"].tail(20).mean() + 1e-9)

    compressed = compression_ratio < 0.85
    volume_expanding = volume_ratio > 1.20
    clv_directional = abs(clv_5) > 0.35
    leaving_balance = abs(hvn_distance_pct) > 1.00
    institutional_support = institutional["value"] in ["MODERATE", "STRONG"]
    supply_confirming_upside = (
        supply["value"] in ["NEARING", "LIKELY"]
        and bias["value"] == "BULLISH"
    )
    rs_support = relative_strength["value"] in ["STRONG", "POSITIVE"]

    score_items = {
        "compression": compressed,
        "volume": volume_expanding,
        "CLV": clv_directional,
        "leaving HVN": leaving_balance,
        "participation": institutional_support,
        "supply": supply_confirming_upside,
        "relative strength": rs_support
    }

    score_num = sum(score_items.values())

    if score_num >= 6:
        status = "HIGH"
    elif score_num in [4, 5]:
        status = "ELEVATED"
    elif score_num in [2, 3]:
        status = "WATCH"
    else:
        status = "LOW"

    if bias["value"] == "BULLISH":
        direction = "UPSIDE"
    elif bias["value"] == "BEARISH":
        direction = "DOWNSIDE"
    else:
        direction = "UNCONFIRMED"

    expansion_move = atr_now * 1.5

    passed = [k for k, v in score_items.items() if v]
    failed = [k for k, v in score_items.items() if not v]

    reason = (
        f"Passed: {', '.join(passed) if passed else 'none'}; "
        f"Missing: {', '.join(failed) if failed else 'none'}."
    )

    return {
        "value": f"{status} / {direction}",
        "status": status,
        "direction": direction,
        "score": f"{score_num}/7",
        "expansion_move": float(expansion_move),
        "upside_target": float(price + expansion_move),
        "downside_target": float(price - expansion_move),
        "reason": reason,
        "score_items": score_items
    }


def discovery_state_model(df, hvns, price):
    dominant_hvn = max(hvns, key=lambda h: h["volume"])
    prior_value = dominant_hvn["price"]

    closes = df["Close"]
    volumes = df["Volume"]

    if price > prior_value:
        mask = closes > prior_value
        direction = "above"
    elif price < prior_value:
        mask = closes < prior_value
        direction = "below"
    else:
        mask = abs(closes - prior_value) / prior_value <= 0.01
        direction = "near"

    days = 0
    for x in reversed(mask.tolist()):
        if x:
            days += 1
        else:
            break

    accepted_pct = float((volumes[mask].sum() / (volumes.sum() + 1e-9)) * 100)

    if direction == "above":
        new_hvns = [h for h in hvns if h["price"] > prior_value]
    elif direction == "below":
        new_hvns = [h for h in hvns if h["price"] < prior_value]
    else:
        new_hvns = []

    new_hvn_count = len(new_hvns)

    score_num = (
        min(days / 120 * 30, 30)
        + min(accepted_pct / 50 * 40, 40)
        + min(new_hvn_count / 5 * 30, 30)
    )

    if score_num >= 76:
        value = "NEW_VALUE_ESTABLISHED"
    elif score_num >= 51:
        value = "ESTABLISHING_NEW_VALUE"
    elif score_num >= 26:
        value = "EARLY_DISCOVERY"
    else:
        value = "BALANCE"

    if value == "BALANCE":
        reason = "Price remains anchored to prior value."
    else:
        reason = f"{accepted_pct:.0f}% accepted volume; {new_hvn_count} new HVNs."

    return {
        "value": value,
        "score": f"{score_num:.0f}/100",
        "days": int(days),
        "reason": reason,
        "direction": direction
    }


def auction_model_base(raw_df, benchmark_df, hvn_window=252):
    df = add_base_columns(raw_df)

    price = float(df.iloc[-1]["Close"])

    hvn_df = df.tail(hvn_window)
    hvns = compute_hvns(hvn_df)

    nearest_hvn, nearest_hvn_above, nearest_hvn_below = get_hvn_levels(hvns, price)

    hvn_distance_pct = ((price - nearest_hvn) / nearest_hvn) * 100

    if nearest_hvn_above is None and price > max(h["price"] for h in hvns):
        hvn_status = "PRICE_DISCOVERY_ABOVE"
    elif nearest_hvn_below is None and price < min(h["price"] for h in hvns):
        hvn_status = "PRICE_DISCOVERY_BELOW"
    elif abs(hvn_distance_pct) <= 1:
        hvn_status = "IN_HVN"
    elif abs(hvn_distance_pct) <= 3:
        hvn_status = "NEARING_HVN"
    else:
        hvn_status = "AWAY_FROM_HVN"

    clv_5 = df["CLV"].tail(5).mean()
    clv_10 = df["CLV"].tail(10).mean()

    volume_ratio = float(df.iloc[-1]["Volume_Ratio"])
    atr_now = float(df.iloc[-1]["ATR"])

    relative_strength = relative_strength_model(raw_df, benchmark_df, BENCHMARK_TICKER)

    institutional = institutional_participation_model(df)

    supply = supply_exhaustion_model(
        df=df,
        clv_5=clv_5,
        clv_10=clv_10,
        price=price
    )

    bias = near_term_bias_model(
        clv_5=clv_5,
        volume_ratio=volume_ratio,
        hvn_distance_pct=hvn_distance_pct,
        supply=supply,
        institutional=institutional,
        relative_strength=relative_strength
    )

    expected_range = expected_auction_range_model(
        price=price,
        atr_now=atr_now,
        clv_5=clv_5,
        volume_ratio=volume_ratio,
        nearest_hvn_above=nearest_hvn_above,
        nearest_hvn_below=nearest_hvn_below
    )

    expansion = expansion_potential_model(
        df=df,
        price=price,
        atr_now=atr_now,
        clv_5=clv_5,
        hvn_distance_pct=hvn_distance_pct,
        bias=bias,
        volume_ratio=volume_ratio,
        supply=supply,
        institutional=institutional,
        relative_strength=relative_strength
    )

    discovery = discovery_state_model(hvn_df, hvns, price)

    return {
        "ticker_price": price,
        "last_bar_date": df.index[-1].date(),
        "near_term_bias": bias,
        "relative_strength": relative_strength,
        "institutional_participation": institutional,
        "expected_range": expected_range,
        "hvn_analysis": {
            "value": hvn_status,
            "score": f"{hvn_distance_pct:.2f}%",
            "reason": "Current price versus nearest accepted value area."
        },
        "supply_exhaustion": supply,
        "expansion_potential": expansion,
        "discovery_state": discovery,
        "nearest_hvn": nearest_hvn,
        "nearest_hvn_above": nearest_hvn_above,
        "nearest_hvn_below": nearest_hvn_below,
        "hvn_distance_pct": hvn_distance_pct
    }


def count_streak(states):
    if not states:
        return ""

    current = states[-1]
    streak = 0

    for s in reversed(states):
        if s == current:
            streak += 1
        else:
            break

    return streak


def compute_streaks(raw_df, benchmark_df, lookback=20):
    states = {
        "near_term_bias": [],
        "institutional_participation": [],
        "supply_exhaustion": [],
        "expansion_potential": []
    }

    start = max(120, len(raw_df) - lookback)

    for i in range(start, len(raw_df) + 1):
        try:
            partial_stock = raw_df.iloc[:i].copy()
            last_date = partial_stock.index[-1]
            partial_benchmark = benchmark_df[benchmark_df.index <= last_date].copy()

            r = auction_model_base(partial_stock, partial_benchmark)

            states["near_term_bias"].append(r["near_term_bias"]["value"])
            states["institutional_participation"].append(r["institutional_participation"]["value"])
            states["supply_exhaustion"].append(r["supply_exhaustion"]["value"])
            states["expansion_potential"].append(r["expansion_potential"]["status"])
        except Exception:
            continue

    return {k: count_streak(v) for k, v in states.items()}


def auction_model(raw_df, benchmark_df):
    result = auction_model_base(raw_df, benchmark_df)
    streaks = compute_streaks(raw_df, benchmark_df, lookback=20)

    result["near_term_bias"]["days"] = streaks["near_term_bias"]
    result["institutional_participation"]["days"] = streaks["institutional_participation"]
    result["supply_exhaustion"]["days"] = streaks["supply_exhaustion"]
    result["expansion_potential"]["days"] = streaks["expansion_potential"]

    return result


def price_impact_for_row(attribute, result):
    if attribute == "5-day auction range":
        skew = result["expected_range"]["skew"]
        if skew > 0:
            return "+"
        elif skew < 0:
            return "-"
        return "="

    if attribute == "Relative strength":
        return result["relative_strength"]["price_impact"]

    if attribute == "Institutional participation":
        value = result["institutional_participation"]["value"]
        bias = result["near_term_bias"]["value"]

        if value in ["MODERATE", "STRONG"] and bias == "BULLISH":
            return "+"
        elif value in ["MODERATE", "STRONG"] and bias == "BEARISH":
            return "-"
        return "="

    if attribute == "Supply exhaustion":
        if result["supply_exhaustion"]["value"] in ["NEARING", "LIKELY"]:
            return "+"
        return "="

    if attribute == "Expansion potential":
        status = result["expansion_potential"]["status"]
        direction = result["expansion_potential"]["direction"]

        if status in ["WATCH", "ELEVATED", "HIGH"] and direction == "UPSIDE":
            return "+"
        elif status in ["WATCH", "ELEVATED", "HIGH"] and direction == "DOWNSIDE":
            return "-"
        return "="

    if attribute == "Discovery state":
        value = result["discovery_state"]["value"]
        direction = result["discovery_state"]["direction"]

        if value in ["ESTABLISHING_NEW_VALUE", "NEW_VALUE_ESTABLISHED"] and direction == "above":
            return "+"
        elif value in ["ESTABLISHING_NEW_VALUE", "NEW_VALUE_ESTABLISHED"] and direction == "below":
            return "-"
        return "="

    if attribute == "HVN status":
        value = result["hvn_analysis"]["value"]

        if value == "PRICE_DISCOVERY_ABOVE":
            return "+"
        elif value == "PRICE_DISCOVERY_BELOW":
            return "-"
        return "="

    return ""


def make_summary_table(result):
    rows = [
        [
            "Near-term bias",
            result["near_term_bias"]["value"],
            result["near_term_bias"]["score"],
            result["near_term_bias"]["days"],
            "",
            result["near_term_bias"]["reason"]
        ],
        [
            "5-day auction range",
            result["expected_range"]["value"],
            result["expected_range"]["score"],
            "",
            price_impact_for_row("5-day auction range", result),
            result["expected_range"]["reason"]
        ],
        [
            "Relative strength",
            result["relative_strength"]["value"],
            result["relative_strength"]["score"],
            result["relative_strength"]["days"],
            price_impact_for_row("Relative strength", result),
            result["relative_strength"]["reason"]
        ],
        [
            "Institutional participation",
            result["institutional_participation"]["value"],
            result["institutional_participation"]["score"],
            result["institutional_participation"]["days"],
            price_impact_for_row("Institutional participation", result),
            result["institutional_participation"]["reason"]
        ],
        [
            "Supply exhaustion",
            result["supply_exhaustion"]["value"],
            result["supply_exhaustion"]["score"],
            result["supply_exhaustion"]["days"],
            price_impact_for_row("Supply exhaustion", result),
            result["supply_exhaustion"]["reason"]
        ],
        [
            "Expansion potential",
            result["expansion_potential"]["value"],
            result["expansion_potential"]["score"],
            result["expansion_potential"]["days"],
            price_impact_for_row("Expansion potential", result),
            result["expansion_potential"]["reason"]
        ],
        [
            "Discovery state",
            result["discovery_state"]["value"],
            result["discovery_state"]["score"],
            result["discovery_state"]["days"],
            price_impact_for_row("Discovery state", result),
            result["discovery_state"]["reason"]
        ],
        [
            "HVN status",
            result["hvn_analysis"]["value"],
            result["hvn_analysis"]["score"],
            "",
            price_impact_for_row("HVN status", result),
            result["hvn_analysis"]["reason"]
        ],
        [
            "HVNs nearest",
            f"{money0(result['nearest_hvn_below'])} to {money0(result['nearest_hvn_above'])}",
            "",
            "",
            "",
            "Nearest current-acceptance value areas."
        ],
        [
            "1.5x ATR expansion levels",
            f"{money0(result['expansion_potential']['downside_target'])} to {money0(result['expansion_potential']['upside_target'])}",
            money0(result["expansion_potential"]["expansion_move"]),
            "",
            "",
            "Reference levels for a 1.5x ATR move."
        ]
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Attribute",
            "Value",
            "Score",
            "Days",
            "Price impact",
            "Reason"
        ]
    )



def normalize_ohlcv_columns(data, ticker=None):
    """Normalize yfinance output into Open/High/Low/Close/Volume columns."""
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        required = {"Open", "High", "Low", "Close", "Volume"}
        level0 = set(map(str, data.columns.get_level_values(0)))
        level_last = set(map(str, data.columns.get_level_values(-1)))

        if required.issubset(level0):
            data.columns = data.columns.get_level_values(0)
        elif required.issubset(level_last):
            data.columns = data.columns.get_level_values(-1)
        elif ticker is not None:
            ticker_upper = str(ticker).upper()
            for level in range(data.columns.nlevels):
                labels = [str(x).upper() for x in data.columns.get_level_values(level)]
                if ticker_upper in labels:
                    try:
                        data = data.xs(
                            key=data.columns.get_level_values(level)[labels.index(ticker_upper)],
                            axis=1,
                            level=level
                        )
                    except Exception:
                        pass
                    break

    data = data.loc[:, ~data.columns.duplicated()]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    return data[required_cols].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlcv(ticker):
    ticker = str(ticker).strip().upper()

    data = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    return normalize_ohlcv_columns(data, ticker=ticker)


def render_summary_table(summary_table):
    """Render a readable table with wrapped Reason text for Streamlit."""
    df = summary_table.fillna("").copy()

    st.markdown(
        """
        <style>
        .apm-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid rgba(250, 250, 250, 0.15);
            border-radius: 8px;
        }
        table.apm-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.92rem;
        }
        table.apm-table th, table.apm-table td {
            border-bottom: 1px solid rgba(250, 250, 250, 0.12);
            padding: 0.55rem 0.65rem;
            vertical-align: top;
            white-space: normal;
            word-break: break-word;
        }
        table.apm-table th {
            font-weight: 700;
            text-align: left;
            background: rgba(128, 128, 128, 0.16);
        }
        table.apm-table th:nth-child(1), table.apm-table td:nth-child(1) { width: 14%; }
        table.apm-table th:nth-child(2), table.apm-table td:nth-child(2) { width: 15%; }
        table.apm-table th:nth-child(3), table.apm-table td:nth-child(3) { width: 8%; }
        table.apm-table th:nth-child(4), table.apm-table td:nth-child(4) { width: 7%; text-align: center; }
        table.apm-table th:nth-child(5), table.apm-table td:nth-child(5) { width: 8%; text-align: center; font-weight: 700; }
        table.apm-table th:nth-child(6), table.apm-table td:nth-child(6) { width: 48%; }
        </style>
        """,
        unsafe_allow_html=True
    )

    html_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            cells.append(f"<td>{html.escape(str(row[col]))}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    table_html = (
        '<div class="apm-table-wrap">'
        '<table class="apm-table">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(html_rows)}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def build_chart(df, result, ticker):
    expansion = result["expansion_potential"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["Close"], linewidth=2, label="Close")

    ax.axhline(
        result["nearest_hvn"],
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f'Nearest HVN {money0(result["nearest_hvn"])}'
    )

    if result["nearest_hvn_above"] is not None:
        ax.axhline(
            result["nearest_hvn_above"],
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f'HVN Above {money0(result["nearest_hvn_above"])}'
        )

    if result["nearest_hvn_below"] is not None:
        ax.axhline(
            result["nearest_hvn_below"],
            color="red",
            linestyle="-.",
            linewidth=2,
            label=f'HVN Below {money0(result["nearest_hvn_below"])}'
        )

    ax.axhline(
        result["expected_range"]["low"],
        color="red",
        linestyle=":",
        linewidth=2,
        label=f'5D Auction Low {money0(result["expected_range"]["low"])}'
    )

    ax.axhline(
        result["expected_range"]["high"],
        color="green",
        linestyle=":",
        linewidth=2,
        label=f'5D Auction High {money0(result["expected_range"]["high"])}'
    )

    ax.axhline(
        expansion["upside_target"],
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.45,
        label=f'1.5x ATR Upside {money0(expansion["upside_target"])}'
    )

    ax.axhline(
        expansion["downside_target"],
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.45,
        label=f'1.5x ATR Downside {money0(expansion["downside_target"])}'
    )

    ax.set_title(f"{ticker} Auction Pressure Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig




def evaluate_signal_accuracy(signal_row, future_df, neutral_band=0.01):
    entry_price = signal_row["entry_price"]
    future_close = float(future_df["Close"].iloc[-1])
    future_high = float(future_df["High"].max())
    future_low = float(future_df["Low"].min())
    forward_return = (future_close / entry_price) - 1

    bias = signal_row["near_term_bias"]
    if bias == "BULLISH":
        bias_correct = forward_return > 0
    elif bias == "BEARISH":
        bias_correct = forward_return < 0
    else:
        bias_correct = abs(forward_return) <= neutral_band

    directional_scored = bias in ["BULLISH", "BEARISH"]

    range_low = signal_row["range_low"]
    range_high = signal_row["range_high"]
    range_close_hit = range_low <= future_close <= range_high
    range_full_hit = future_low >= range_low and future_high <= range_high

    expansion_status = signal_row["expansion_status"]
    expansion_direction = signal_row["expansion_direction"]
    expansion_scored = expansion_status in ["WATCH", "ELEVATED", "HIGH"]

    if not expansion_scored:
        expansion_hit = None
    elif expansion_direction == "UPSIDE":
        expansion_hit = future_high >= signal_row["expansion_upside"]
    elif expansion_direction == "DOWNSIDE":
        expansion_hit = future_low <= signal_row["expansion_downside"]
    else:
        expansion_hit = (
            future_high >= signal_row["expansion_upside"]
            or future_low <= signal_row["expansion_downside"]
        )

    return {
        "forward_close": future_close,
        "forward_high": future_high,
        "forward_low": future_low,
        "forward_5d_return_%": forward_return * 100,
        "bias_correct": bool(bias_correct),
        "directional_scored": bool(directional_scored),
        "range_close_hit": bool(range_close_hit),
        "range_full_hit": bool(range_full_hit),
        "expansion_scored": bool(expansion_scored),
        "expansion_hit": expansion_hit,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def run_backtest_cached(stock_df, benchmark_df, lookback_days=80, horizon_days=5):
    records = []

    min_history = 140
    if len(stock_df) < min_history + horizon_days + 1:
        return pd.DataFrame()

    start_idx = max(min_history, len(stock_df) - lookback_days - horizon_days)
    end_idx = len(stock_df) - horizon_days

    for i in range(start_idx, end_idx):
        try:
            signal_df = stock_df.iloc[:i].copy()
            future_df = stock_df.iloc[i:i + horizon_days].copy()
            signal_date = signal_df.index[-1]

            partial_benchmark = benchmark_df[benchmark_df.index <= signal_date].copy()
            if len(partial_benchmark) < min_history:
                continue

            result = auction_model_base(signal_df, partial_benchmark)

            row = {
                "signal_date": signal_date.date(),
                "entry_price": float(result["ticker_price"]),
                "near_term_bias": result["near_term_bias"]["value"],
                "bias_score": result["near_term_bias"]["score"],
                "range_low": float(result["expected_range"]["low"]),
                "range_high": float(result["expected_range"]["high"]),
                "relative_strength": result["relative_strength"]["value"],
                "institutional_participation": result["institutional_participation"]["value"],
                "supply_exhaustion": result["supply_exhaustion"]["value"],
                "expansion_status": result["expansion_potential"]["status"],
                "expansion_direction": result["expansion_potential"]["direction"],
                "expansion_upside": float(result["expansion_potential"]["upside_target"]),
                "expansion_downside": float(result["expansion_potential"]["downside_target"]),
                "discovery_state": result["discovery_state"]["value"],
            }

            outcome = evaluate_signal_accuracy(row, future_df, neutral_band=0.01)
            row.update(outcome)
            records.append(row)
        except Exception:
            continue

    return pd.DataFrame(records)


def pct_text(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.0f}%"


def render_backtest_dashboard(backtest_df, ticker, horizon_days=5):
    st.subheader("Backtest Dashboard")

    if backtest_df.empty:
        st.warning("Not enough valid historical signals to backtest this ticker.")
        return

    n = len(backtest_df)

    directional = backtest_df[backtest_df["directional_scored"]]
    directional_accuracy = directional["bias_correct"].mean() if len(directional) else np.nan

    all_bias_accuracy = backtest_df["bias_correct"].mean()
    range_close_hit = backtest_df["range_close_hit"].mean()
    range_full_hit = backtest_df["range_full_hit"].mean()

    expansion = backtest_df[backtest_df["expansion_scored"]].copy()
    expansion_hit_rate = expansion["expansion_hit"].mean() if len(expansion) else np.nan

    avg_forward_return = backtest_df["forward_5d_return_%"].mean()
    median_forward_return = backtest_df["forward_5d_return_%"].median()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signals tested", f"{n}")
    c2.metric("Directional accuracy", pct_text(directional_accuracy))
    c3.metric("Range close hit", pct_text(range_close_hit))
    c4.metric("Range full containment", pct_text(range_full_hit))
    c5.metric("Expansion hit rate", pct_text(expansion_hit_rate))

    st.caption(
        "Directional accuracy scores BULLISH/BEARISH calls against the next 5-day close direction. "
        "Range close hit checks whether the next 5-day close landed inside the model range. "
        "Full containment checks whether the entire next 5-day high-low path stayed inside the range."
    )

    accuracy_rows = [
        {
            "Measure": "Directional accuracy",
            "Result": pct_text(directional_accuracy),
            "Sample": len(directional),
            "Definition": "BULLISH correct if next 5-day close is higher; BEARISH correct if lower."
        },
        {
            "Measure": "All bias accuracy",
            "Result": pct_text(all_bias_accuracy),
            "Sample": n,
            "Definition": "Includes NEUTRAL as correct only when next 5-day return is within +/-1%."
        },
        {
            "Measure": "5-day range close hit",
            "Result": pct_text(range_close_hit),
            "Sample": n,
            "Definition": "Next 5-day close finished inside the auction range."
        },
        {
            "Measure": "5-day range full containment",
            "Result": pct_text(range_full_hit),
            "Sample": n,
            "Definition": "Entire next 5-day high-low path stayed inside the auction range."
        },
        {
            "Measure": "Expansion hit rate",
            "Result": pct_text(expansion_hit_rate),
            "Sample": len(expansion),
            "Definition": "For WATCH/ELEVATED/HIGH setups, price touched the indicated 1.5x ATR expansion side."
        },
        {
            "Measure": "Average next 5-day return",
            "Result": f"{avg_forward_return:+.2f}%",
            "Sample": n,
            "Definition": "Average forward close-to-close return after each historical signal."
        },
        {
            "Measure": "Median next 5-day return",
            "Result": f"{median_forward_return:+.2f}%",
            "Sample": n,
            "Definition": "Median forward close-to-close return after each historical signal."
        },
    ]

    st.markdown("#### Accuracy Summary")
    st.dataframe(pd.DataFrame(accuracy_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Recent Backtest Signals")
    display_cols = [
        "signal_date",
        "entry_price",
        "near_term_bias",
        "forward_5d_return_%",
        "bias_correct",
        "range_close_hit",
        "range_full_hit",
        "expansion_status",
        "expansion_direction",
        "expansion_hit",
        "relative_strength",
        "institutional_participation",
        "supply_exhaustion",
    ]

    display_df = backtest_df[display_cols].tail(30).copy()
    display_df["entry_price"] = display_df["entry_price"].map(lambda x: f"${x:,.0f}")
    display_df["forward_5d_return_%"] = display_df["forward_5d_return_%"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = backtest_df.to_csv(index=False)
    st.download_button(
        "Download backtest results as CSV",
        data=csv,
        file_name=f"{ticker}_backtest.csv",
        mime="text/csv"
    )

def run_model_streamlit(ticker, benchmark):
    global BENCHMARK_TICKER
    BENCHMARK_TICKER = benchmark.upper().strip()

    df = download_ohlcv(ticker)
    benchmark_df = download_ohlcv(BENCHMARK_TICKER)

    if df.empty:
        raise ValueError("No stock data returned. Check the ticker.")
    if benchmark_df.empty:
        raise ValueError("No benchmark data returned. Check the benchmark ticker.")
    if len(df) < 120:
        raise ValueError("Not enough stock history.")
    if len(benchmark_df) < 120:
        raise ValueError("Not enough benchmark history.")

    result = auction_model(df, benchmark_df)
    summary_table = make_summary_table(result)
    return result, df, benchmark_df, summary_table


def main():
    st.set_page_config(
        page_title="Auction Pressure Model",
        layout="wide"
    )

    st.title("Auction Pressure Model")

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
        run_backtest = st.checkbox("Run backtest dashboard", value=True)
        backtest_lookback = st.slider("Backtest signals", min_value=30, max_value=150, value=80, step=10)
        run_button = st.button("Run model", type="primary")

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    if run_button:
        try:
            with st.spinner(f"Downloading data and running model for {ticker}..."):
                result, df, benchmark_df, summary_table = run_model_streamlit(ticker, benchmark)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ticker", ticker)
            col2.metric("Benchmark", benchmark)
            col3.metric("Current price", money0(result["ticker_price"]))
            col4.metric("Last data bar", str(result["last_bar_date"]))

            st.subheader("Model Output")
            render_summary_table(summary_table)

            with st.expander("Show as downloadable dataframe"):
                st.dataframe(summary_table, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download table as CSV",
                    data=summary_table.to_csv(index=False),
                    file_name=f"{ticker}_auction_pressure_model.csv",
                    mime="text/csv"
                )

            st.subheader("Chart")
            fig = build_chart(df, result, ticker)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if run_backtest:
                with st.spinner(f"Running {backtest_lookback}-signal backtest for {ticker}..."):
                    backtest_df = run_backtest_cached(df, benchmark_df, lookback_days=backtest_lookback, horizon_days=5)
                render_backtest_dashboard(backtest_df, ticker, horizon_days=5)

        except Exception as e:
            st.error(str(e))
    else:
        st.caption("Enter a ticker and click Run model.")


if __name__ == "__main__":
    main()
