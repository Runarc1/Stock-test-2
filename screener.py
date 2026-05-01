"""
Stock screener.

Fetches price/volume history and fundamental data via yfinance, then filters
the universe down to two candidate buckets:
  - momentum_candidates  : breakout stocks with strong trend & volume
  - squeeze_candidates   : high short-interest stocks showing early squeeze signs

Both lists are sorted by score (highest first) so the caller can take the top N.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

import config as cfg
import indicators as ind

log = logging.getLogger(__name__)


@dataclass
class CandidateStock:
    ticker: str
    price: float
    market_cap: float
    avg_volume: float
    rsi: float
    volume_ratio: float
    short_float: float       # fraction of float that is short (0.15 = 15 %)
    days_to_cover: float
    momentum_score: float
    strategy: str            # "momentum" | "squeeze"
    score: float = field(init=False)

    def __post_init__(self) -> None:
        # Blended score used for ranking within each strategy bucket.
        if self.strategy == "squeeze":
            # Reward high short float, high days-to-cover, strong volume spike.
            self.score = (
                min(self.short_float / 0.30, 1.0) * 40
                + min(self.days_to_cover / 10.0, 1.0) * 30
                + min((self.volume_ratio - 1.0) / 3.0, 1.0) * 30
            )
        else:
            self.score = self.momentum_score


def _fetch_history(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """Download daily OHLCV for `ticker`. Returns None on failure."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        # yfinance sometimes returns MultiIndex columns; flatten if so.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        return df
    except Exception as exc:
        log.debug("yfinance error for %s: %s", ticker, exc)
        return None


def _fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _passes_basic_filters(info: dict, df: pd.DataFrame) -> tuple[bool, str]:
    """Return (passes, reason). Reason is non-empty if it fails."""
    price = float(df["close"].iloc[-1])
    if price < cfg.MIN_PRICE or price > cfg.MAX_PRICE:
        return False, f"price {price:.2f} out of range"

    mkt_cap = info.get("marketCap") or 0
    if mkt_cap < cfg.MIN_MARKET_CAP:
        return False, f"market cap {mkt_cap:,} too small"

    avg_vol = ind.avg_volume(df)
    if avg_vol < cfg.MIN_AVG_VOLUME:
        return False, f"avg volume {avg_vol:,.0f} too low"

    return True, ""


def _evaluate_momentum(ticker: str, df: pd.DataFrame, info: dict) -> Optional[CandidateStock]:
    """Return a CandidateStock if the stock passes momentum criteria, else None."""
    r = ind.current_rsi(df, cfg.RSI_PERIOD)
    if not (cfg.MOMENTUM_RSI_MIN <= r <= cfg.MOMENTUM_RSI_MAX):
        log.debug("%s RSI %.1f not in momentum range", ticker, r)
        return None

    if not ind.is_above_sma(df, cfg.MOMENTUM_SMA_PERIOD):
        log.debug("%s below 50 SMA", ticker)
        return None

    if not ind.is_above_sma(df, cfg.MOMENTUM_SMA_SHORT):
        log.debug("%s below 20 SMA", ticker)
        return None

    vol_r = ind.volume_ratio(df)
    if vol_r < cfg.BREAKOUT_VOLUME_MULTIPLIER:
        log.debug("%s volume ratio %.2f below threshold", ticker, vol_r)
        return None

    if ind.pct_from_52w_high(df) < cfg.NEAR_HIGH_THRESHOLD:
        log.debug("%s not near 52w high", ticker)
        return None

    if not ind.macd_bullish_crossover(df, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL):
        if not ind.macd_is_positive(df, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL):
            log.debug("%s MACD not bullish", ticker)
            return None

    price = float(df["close"].iloc[-1])
    mkt_cap = info.get("marketCap") or 0
    short_float = float(info.get("shortPercentOfFloat") or 0)
    dtc = float(info.get("shortRatio") or 0)

    return CandidateStock(
        ticker=ticker,
        price=price,
        market_cap=mkt_cap,
        avg_volume=ind.avg_volume(df),
        rsi=r,
        volume_ratio=vol_r,
        short_float=short_float,
        days_to_cover=dtc,
        momentum_score=ind.momentum_score(df),
        strategy="momentum",
    )


def _evaluate_squeeze(ticker: str, df: pd.DataFrame, info: dict) -> Optional[CandidateStock]:
    """Return a CandidateStock if the stock is a short-squeeze candidate, else None."""
    short_float = float(info.get("shortPercentOfFloat") or 0)
    if short_float < cfg.MIN_SHORT_FLOAT:
        log.debug("%s short float %.1f%% below threshold", ticker, short_float * 100)
        return None

    dtc = float(info.get("shortRatio") or 0)
    if dtc < cfg.MIN_DAYS_TO_COVER:
        log.debug("%s days-to-cover %.1f below threshold", ticker, dtc)
        return None

    vol_r = ind.volume_ratio(df)
    if vol_r < cfg.SQUEEZE_VOLUME_MULTIPLIER:
        log.debug("%s volume ratio %.2f below squeeze threshold", ticker, vol_r)
        return None

    r = ind.current_rsi(df, cfg.RSI_PERIOD)
    if r > cfg.SQUEEZE_MAX_RSI:
        log.debug("%s RSI %.1f too high for squeeze entry", ticker, r)
        return None

    # Require price to at least be above the 20 SMA (short-term uptrend starting).
    if not ind.is_above_sma(df, cfg.MOMENTUM_SMA_SHORT):
        log.debug("%s below 20 SMA, squeeze not confirmed", ticker)
        return None

    price = float(df["close"].iloc[-1])
    mkt_cap = info.get("marketCap") or 0

    return CandidateStock(
        ticker=ticker,
        price=price,
        market_cap=mkt_cap,
        avg_volume=ind.avg_volume(df),
        rsi=r,
        volume_ratio=vol_r,
        short_float=short_float,
        days_to_cover=dtc,
        momentum_score=ind.momentum_score(df),
        strategy="squeeze",
    )


def screen_universe(
    tickers: list[str] | None = None,
) -> tuple[list[CandidateStock], list[CandidateStock]]:
    """
    Screen the universe and return (momentum_candidates, squeeze_candidates).
    Both lists are sorted by score descending.
    """
    universe = list({*(tickers or []), *cfg.DEFAULT_UNIVERSE, *cfg.EXTRA_WATCHLIST})
    log.info("Screening %d tickers …", len(universe))

    momentum: list[CandidateStock] = []
    squeeze: list[CandidateStock] = []

    for ticker in universe:
        df = _fetch_history(ticker)
        if df is None:
            log.debug("Skipping %s — no history", ticker)
            continue

        info = _fetch_info(ticker)
        ok, reason = _passes_basic_filters(info, df)
        if not ok:
            log.debug("Skipping %s — %s", ticker, reason)
            continue

        # Evaluate as squeeze first (higher priority / more explosive).
        sq = _evaluate_squeeze(ticker, df, info)
        if sq is not None:
            squeeze.append(sq)
            log.info("  SQUEEZE candidate: %s  score=%.1f  short=%.0f%%  dtc=%.1f  volR=%.1fx",
                     ticker, sq.score, sq.short_float * 100, sq.days_to_cover, sq.volume_ratio)
            continue  # Don't double-classify

        mo = _evaluate_momentum(ticker, df, info)
        if mo is not None:
            momentum.append(mo)
            log.info("  MOMENTUM candidate: %s  score=%.1f  RSI=%.1f  volR=%.1fx",
                     ticker, mo.score, mo.rsi, mo.volume_ratio)

    momentum.sort(key=lambda c: c.score, reverse=True)
    squeeze.sort(key=lambda c: c.score, reverse=True)

    log.info(
        "Screening complete — %d momentum, %d squeeze candidates",
        len(momentum), len(squeeze),
    )
    return momentum, squeeze
