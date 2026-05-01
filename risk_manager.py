"""
Risk management and position sizing.

Kelly-fraction sizing: risk a fixed percentage of portfolio equity per trade,
scaled by the entry score so higher-conviction setups get slightly more capital.
Hard guardrails:
  - Never exceed MAX_POSITION_PCT of portfolio in one name.
  - Never open new positions if daily loss limit is hit.
  - Never hold more than MAX_POSITIONS simultaneously.
"""

from __future__ import annotations

import logging
from typing import Optional

import config as cfg
from signals import EntrySignal, OpenPosition

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, portfolio_value: float) -> None:
        self._starting_day_value = portfolio_value
        self._portfolio_value = portfolio_value

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    def start_new_day(self, portfolio_value: float) -> None:
        self._starting_day_value = portfolio_value
        self._portfolio_value = portfolio_value
        log.info("New day — portfolio value: $%.2f", portfolio_value)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    @property
    def daily_pnl_pct(self) -> float:
        return (self._portfolio_value - self._starting_day_value) / self._starting_day_value

    def daily_loss_limit_hit(self) -> bool:
        hit = self.daily_pnl_pct <= -cfg.DAILY_LOSS_LIMIT_PCT
        if hit:
            log.warning(
                "Daily loss limit hit: %.2f%% (limit %.2f%%)",
                self.daily_pnl_pct * 100, cfg.DAILY_LOSS_LIMIT_PCT * 100,
            )
        return hit

    def slots_available(self, open_positions: dict[str, OpenPosition]) -> int:
        return max(0, cfg.MAX_POSITIONS - len(open_positions))

    def can_open_position(self, open_positions: dict[str, OpenPosition]) -> bool:
        if self.daily_loss_limit_hit():
            return False
        if self.slots_available(open_positions) == 0:
            log.info("Max positions (%d) reached", cfg.MAX_POSITIONS)
            return False
        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def shares_to_buy(
        self,
        signal: EntrySignal,
        open_positions: dict[str, OpenPosition],
    ) -> Optional[int]:
        """
        Return the number of whole shares to buy, or None if the position
        should not be opened (size would be too small or guardrails fire).
        """
        if not self.can_open_position(open_positions):
            return None

        # Dollar risk per share = distance from entry to stop.
        risk_per_share = signal.price - signal.stop_price
        if risk_per_share <= 0:
            log.warning("%s: invalid stop price (%.2f >= entry %.2f)", signal.ticker,
                        signal.stop_price, signal.price)
            return None

        # Base dollar risk = MAX_RISK_PER_TRADE_PCT of portfolio.
        # Scale by score (0–100) so top setups get up to 1.5× the base risk.
        score_multiplier = 0.75 + 0.75 * min(signal.score / 100.0, 1.0)
        dollar_risk = self._portfolio_value * cfg.MAX_RISK_PER_TRADE_PCT * score_multiplier

        shares = int(dollar_risk / risk_per_share)
        if shares < 1:
            log.debug("%s: position size rounds to 0 shares", signal.ticker)
            return None

        # Cap at MAX_POSITION_PCT of portfolio.
        max_dollars = self._portfolio_value * cfg.MAX_POSITION_PCT
        capped_shares = int(max_dollars / signal.price)
        shares = min(shares, capped_shares)

        if shares < 1:
            log.debug("%s: capped position size is 0 shares", signal.ticker)
            return None

        position_value = shares * signal.price
        log.info(
            "Position size for %s: %d shares @ $%.2f = $%.2f  "
            "(%.1f%% of portfolio, risk=$%.2f)",
            signal.ticker, shares, signal.price, position_value,
            position_value / self._portfolio_value * 100,
            shares * risk_per_share,
        )
        return shares

    # ------------------------------------------------------------------
    # Market-regime filter
    # ------------------------------------------------------------------

    @staticmethod
    def market_is_bullish(market_df) -> bool:  # type: ignore[no-untyped-def]
        """
        Simple regime filter: SPY must be above its 50 SMA and RSI > 45.
        When the overall market is bearish, the bot won't open new longs.
        """
        import indicators as ind  # avoid circular at module level
        above_sma = ind.is_above_sma(market_df, 50)
        r = ind.current_rsi(market_df)
        bullish = above_sma and r > 45
        if not bullish:
            log.warning(
                "Market regime filter: SPY SMA50=%s  RSI=%.1f — no new entries",
                above_sma, r,
            )
        return bullish
