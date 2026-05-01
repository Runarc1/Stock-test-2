"""
Technical indicator calculations.

All functions accept a pandas DataFrame with at minimum columns:
  open, high, low, close, volume  (lowercase, adjusted prices preferred)

They return either a scalar (latest value) or a Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(period).mean()


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    return _ema(df["close"], period)


def is_above_sma(df: pd.DataFrame, period: int) -> bool:
    """Return True if the latest close is above the SMA(period)."""
    s = sma(df, period)
    if s.isna().iloc[-1]:
        return False
    return float(df["close"].iloc[-1]) > float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Momentum — RSI
# ---------------------------------------------------------------------------

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def current_rsi(df: pd.DataFrame, period: int = 14) -> float:
    r = rsi(df, period)
    return float(r.iloc[-1]) if not r.isna().iloc[-1] else float("nan")


# ---------------------------------------------------------------------------
# Momentum — MACD
# ---------------------------------------------------------------------------

def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    fast_ema = _ema(df["close"], fast)
    slow_ema = _ema(df["close"], slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_bullish_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    """True if MACD line crossed above signal line within the last 3 bars."""
    line, sig, _ = macd(df, fast, slow, signal)
    if len(line) < 4:
        return False
    for i in range(-3, 0):
        if line.iloc[i - 1] < sig.iloc[i - 1] and line.iloc[i] >= sig.iloc[i]:
            return True
    return False


def macd_is_positive(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    """True if the MACD histogram is positive (bullish momentum)."""
    _, _, hist = macd(df, fast, slow, signal)
    return float(hist.iloc[-1]) > 0


# ---------------------------------------------------------------------------
# Volume analysis
# ---------------------------------------------------------------------------

def volume_ratio(df: pd.DataFrame, avg_period: int = 20) -> float:
    """Today's volume divided by the rolling average volume."""
    avg_vol = df["volume"].rolling(avg_period).mean().iloc[-1]
    if avg_vol == 0 or np.isnan(avg_vol):
        return 0.0
    return float(df["volume"].iloc[-1]) / float(avg_vol)


def avg_volume(df: pd.DataFrame, period: int = 20) -> float:
    return float(df["volume"].rolling(period).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Volatility — Average True Range
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def current_atr(df: pd.DataFrame, period: int = 14) -> float:
    a = atr(df, period)
    return float(a.iloc[-1]) if not a.isna().iloc[-1] else float("nan")


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper, middle, lower) bands."""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


def near_upper_band(df: pd.DataFrame, period: int = 20) -> bool:
    """True if price is within 1% of the upper Bollinger Band (breakout zone)."""
    upper, _, _ = bollinger_bands(df, period)
    close = float(df["close"].iloc[-1])
    u = float(upper.iloc[-1])
    return close >= u * 0.99


# ---------------------------------------------------------------------------
# 52-week high proximity
# ---------------------------------------------------------------------------

def pct_from_52w_high(df: pd.DataFrame) -> float:
    """Fraction of 52-week high that current price represents (1.0 = at high)."""
    period = min(252, len(df))
    high_52w = df["high"].rolling(period).max().iloc[-1]
    if high_52w == 0:
        return 0.0
    return float(df["close"].iloc[-1]) / float(high_52w)


# ---------------------------------------------------------------------------
# Relative Strength vs market
# ---------------------------------------------------------------------------

def relative_strength(stock_df: pd.DataFrame, market_df: pd.DataFrame, period: int = 20) -> float:
    """
    RS = stock % change over period / market % change over period.
    Values > 1.0 mean the stock is outperforming the market.
    """
    if len(stock_df) < period or len(market_df) < period:
        return 0.0
    stock_chg = (
        float(stock_df["close"].iloc[-1]) / float(stock_df["close"].iloc[-period]) - 1
    )
    mkt_chg = (
        float(market_df["close"].iloc[-1]) / float(market_df["close"].iloc[-period]) - 1
    )
    if mkt_chg == 0:
        return 1.0
    return stock_chg / abs(mkt_chg)


# ---------------------------------------------------------------------------
# Composite score helpers
# ---------------------------------------------------------------------------

def momentum_score(df: pd.DataFrame) -> float:
    """
    0–100 composite score combining RSI, MACD, volume, and proximity to 52w high.
    Higher is stronger momentum.
    """
    score = 0.0

    r = current_rsi(df)
    if 50 <= r <= 80:
        score += 25 * min((r - 50) / 30, 1.0)

    if macd_is_positive(df):
        score += 25

    vol_r = volume_ratio(df)
    if vol_r > 1.0:
        score += 25 * min((vol_r - 1.0) / 2.0, 1.0)

    near_high = pct_from_52w_high(df)
    score += 25 * max(near_high - 0.75, 0) / 0.25

    return min(score, 100.0)
