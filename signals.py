"""
Buy and sell signal generation.

A Signal captures everything the risk manager and broker need to act:
the ticker, direction, strategy type, and current prices.
Exit signals check open positions against stop-loss, take-profit, and
trailing-stop rules every loop iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yfinance as yf
import pandas as pd

import config as cfg
import indicators as ind
from screener import CandidateStock

log = logging.getLogger(__name__)


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    EOD_CLOSE = "EOD_CLOSE"
    MANUAL = "MANUAL"


@dataclass
class EntrySignal:
    ticker: str
    direction: Direction
    strategy: str          # "momentum" | "squeeze"
    price: float           # current ask price (approximate)
    stop_price: float      # initial hard stop
    target_price: float    # initial take-profit target
    score: float           # screener score (for position-size weighting)


@dataclass
class ExitSignal:
    ticker: str
    direction: Direction
    reason: ExitReason
    current_price: float
    entry_price: float
    pnl_pct: float


@dataclass
class OpenPosition:
    ticker: str
    entry_price: float
    shares: int
    strategy: str
    peak_price: float = field(init=False)
    trailing_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.peak_price = self.entry_price


def generate_entry_signals(
    candidates: list[CandidateStock],
    open_positions: dict[str, OpenPosition],
    available_slots: int,
) -> list[EntrySignal]:
    """
    Convert screener candidates into actionable entry signals.
    Skips tickers already held. Returns at most `available_slots` signals.
    """
    signals: list[EntrySignal] = []
    for candidate in candidates:
        if available_slots <= 0:
            break
        if candidate.ticker in open_positions:
            log.debug("Already holding %s, skipping entry", candidate.ticker)
            continue

        stop = candidate.price * (1 - cfg.STOP_LOSS_PCT)
        target = candidate.price * (1 + cfg.TAKE_PROFIT_PCT)

        signals.append(
            EntrySignal(
                ticker=candidate.ticker,
                direction=Direction.BUY,
                strategy=candidate.strategy,
                price=candidate.price,
                stop_price=stop,
                target_price=target,
                score=candidate.score,
            )
        )
        available_slots -= 1

    return signals


def _get_current_price(ticker: str) -> Optional[float]:
    """Fetch the latest price for an open-position check."""
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if data.empty:
            return None
        # Flatten possible MultiIndex
        data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
        return float(data["close"].iloc[-1])
    except Exception as exc:
        log.warning("Could not fetch price for %s: %s", ticker, exc)
        return None


def generate_exit_signals(
    open_positions: dict[str, OpenPosition],
    eod_close: bool = False,
) -> list[ExitSignal]:
    """
    Scan open positions and return exit signals for any that have hit
    stop-loss, take-profit, trailing-stop, or the end-of-day close flag.
    """
    exit_signals: list[ExitSignal] = []

    for ticker, pos in list(open_positions.items()):
        if eod_close:
            price = _get_current_price(ticker) or pos.entry_price
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            exit_signals.append(
                ExitSignal(
                    ticker=ticker,
                    direction=Direction.SELL,
                    reason=ExitReason.EOD_CLOSE,
                    current_price=price,
                    entry_price=pos.entry_price,
                    pnl_pct=pnl_pct,
                )
            )
            continue

        price = _get_current_price(ticker)
        if price is None:
            continue

        pnl_pct = (price - pos.entry_price) / pos.entry_price

        # Update peak for trailing stop.
        if price > pos.peak_price:
            pos.peak_price = price

        # Activate trailing stop once gain threshold is hit.
        if pnl_pct >= cfg.TRAIL_ACTIVATE_PCT:
            pos.trailing_active = True

        # Check trailing stop.
        if pos.trailing_active:
            trail_stop = pos.peak_price * (1 - cfg.TRAIL_STOP_PCT)
            if price <= trail_stop:
                log.info(
                    "%s trailing stop triggered: price=%.2f  peak=%.2f  trail=%.2f",
                    ticker, price, pos.peak_price, trail_stop,
                )
                exit_signals.append(
                    ExitSignal(
                        ticker=ticker,
                        direction=Direction.SELL,
                        reason=ExitReason.TRAILING_STOP,
                        current_price=price,
                        entry_price=pos.entry_price,
                        pnl_pct=pnl_pct,
                    )
                )
                continue

        # Hard stop-loss.
        if pnl_pct <= -cfg.STOP_LOSS_PCT:
            log.info(
                "%s stop-loss triggered: entry=%.2f  current=%.2f  loss=%.1f%%",
                ticker, pos.entry_price, price, pnl_pct * 100,
            )
            exit_signals.append(
                ExitSignal(
                    ticker=ticker,
                    direction=Direction.SELL,
                    reason=ExitReason.STOP_LOSS,
                    current_price=price,
                    entry_price=pos.entry_price,
                    pnl_pct=pnl_pct,
                )
            )
            continue

        # Take-profit.
        if pnl_pct >= cfg.TAKE_PROFIT_PCT:
            log.info(
                "%s take-profit triggered: entry=%.2f  current=%.2f  gain=%.1f%%",
                ticker, pos.entry_price, price, pnl_pct * 100,
            )
            exit_signals.append(
                ExitSignal(
                    ticker=ticker,
                    direction=Direction.SELL,
                    reason=ExitReason.TAKE_PROFIT,
                    current_price=price,
                    entry_price=pos.entry_price,
                    pnl_pct=pnl_pct,
                )
            )

    return exit_signals
