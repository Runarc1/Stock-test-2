"""
Main trading bot loop.

Flow each SCAN_INTERVAL_MINUTES cycle:
  1. Check market hours / daily loss limit.
  2. Check open positions for exit signals (stop, take-profit, trailing stop).
  3. If near market close, force-close all positions.
  4. Screen universe for new candidates.
  5. Check market regime (SPY SMA/RSI filter).
  6. Generate entry signals and size positions.
  7. Place orders via the broker.
  8. Sleep until next cycle.

Run:
    python bot.py               # live (requires .env with credentials)
    DRY_RUN=true python bot.py  # paper mode (no real orders)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import schedule
import yfinance as yf

import config as cfg
from broker import Broker
from risk_manager import RiskManager
from screener import screen_universe
from signals import (
    EntrySignal,
    ExitReason,
    OpenPosition,
    generate_entry_signals,
    generate_exit_signals,
)

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# Bot state
# ---------------------------------------------------------------------------

class TradingBot:
    def __init__(self) -> None:
        self.broker = Broker()
        self.risk = RiskManager(portfolio_value=10_000.0)  # placeholder; updated on login
        self.open_positions: dict[str, OpenPosition] = {}
        self._trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _now_et(self) -> datetime:
        return datetime.now(ET)

    def _is_market_hours(self) -> bool:
        now = self._now_et()
        if now.weekday() >= 5:  # Saturday / Sunday
            return False
        open_time = now.replace(hour=cfg.MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
        close_time = now.replace(
            hour=cfg.MARKET_CLOSE_HOUR,
            minute=cfg.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        return open_time <= now <= close_time

    def _is_eod_window(self) -> bool:
        """True if within EOD_CLOSE_MINUTES_BEFORE of market close."""
        now = self._now_et()
        close_time = now.replace(
            hour=cfg.MARKET_CLOSE_HOUR,
            minute=cfg.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        delta_minutes = (close_time - now).total_seconds() / 60
        return 0 <= delta_minutes <= cfg.EOD_CLOSE_MINUTES_BEFORE

    def _fetch_market_df(self):
        try:
            df = yf.download(cfg.MARKET_ETF, period="3mo", auto_adjust=True, progress=False)
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.columns, "levels"):
                df.columns = [c[0].lower() for c in df.columns]
            return df
        except Exception as exc:
            log.warning("Could not fetch market data: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------

    def _sync_positions_from_broker(self) -> None:
        """
        Reconcile in-memory positions with what Robinhood reports.
        Useful on startup or after a crash.
        """
        if cfg.DRY_RUN:
            return
        live = self.broker.get_open_positions()
        for ticker, data in live.items():
            if ticker not in self.open_positions:
                self.open_positions[ticker] = OpenPosition(
                    ticker=ticker,
                    entry_price=data["average_buy_price"],
                    shares=int(data["shares"]),
                    strategy="unknown",
                )
        # Remove positions that were closed outside the bot.
        for ticker in list(self.open_positions.keys()):
            if ticker not in live:
                log.info("Position %s closed externally — removing from tracking", ticker)
                del self.open_positions[ticker]

    def _record_trade(self, ticker: str, side: str, shares: int, price: float,
                      reason: str = "") -> None:
        self._trade_log.append({
            "time": self._now_et().isoformat(),
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "price": price,
            "reason": reason,
        })

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def _handle_exits(self, eod: bool = False) -> None:
        if not self.open_positions:
            return

        exit_signals = generate_exit_signals(self.open_positions, eod_close=eod)
        for sig in exit_signals:
            pos = self.open_positions.get(sig.ticker)
            if pos is None:
                continue

            result = self.broker.sell_market(sig.ticker, pos.shares)
            if result is not None:
                pnl_str = f"{sig.pnl_pct * 100:+.2f}%"
                log.info(
                    "CLOSED %s  reason=%s  pnl=%s",
                    sig.ticker, sig.reason.value, pnl_str,
                )
                self._record_trade(sig.ticker, "SELL", pos.shares, sig.current_price, sig.reason.value)
                del self.open_positions[sig.ticker]

                # Update portfolio value estimate after close.
                pv = self.broker.get_portfolio_value() if not cfg.DRY_RUN else self.risk._portfolio_value
                self.risk.update_portfolio_value(pv)

    def _handle_entries(self) -> None:
        if not self.risk.can_open_position(self.open_positions):
            return

        market_df = self._fetch_market_df()
        if market_df is not None and not self.risk.market_is_bullish(market_df):
            log.info("Market regime filter active — skipping new entries this cycle")
            return

        momentum_candidates, squeeze_candidates = screen_universe()

        # Prioritize squeeze setups, then fill remaining slots with momentum.
        available = self.risk.slots_available(self.open_positions)
        all_candidates = squeeze_candidates[:available] + momentum_candidates[:available]

        entry_signals = generate_entry_signals(
            all_candidates,
            self.open_positions,
            available_slots=available,
        )

        for signal in entry_signals:
            shares = self.risk.shares_to_buy(signal, self.open_positions)
            if shares is None or shares < 1:
                continue

            result = self.broker.buy_market(signal.ticker, shares)
            if result is not None:
                actual_price = self.broker.get_current_price(signal.ticker) or signal.price
                self.open_positions[signal.ticker] = OpenPosition(
                    ticker=signal.ticker,
                    entry_price=actual_price,
                    shares=shares,
                    strategy=signal.strategy,
                )
                self._record_trade(signal.ticker, "BUY", shares, actual_price, signal.strategy)
                log.info(
                    "OPENED %s  strategy=%s  shares=%d  entry=$%.2f  "
                    "stop=$%.2f  target=$%.2f",
                    signal.ticker, signal.strategy, shares, actual_price,
                    signal.stop_price, signal.target_price,
                )

    def run_cycle(self) -> None:
        """One full scan+trade cycle."""
        if not self._is_market_hours():
            log.debug("Outside market hours — skipping cycle")
            return

        now = self._now_et()
        log.info("=== Cycle start: %s ===", now.strftime("%Y-%m-%d %H:%M ET"))

        # Sync with broker every cycle.
        self._sync_positions_from_broker()

        # Update portfolio value.
        pv = self.broker.get_portfolio_value() if not cfg.DRY_RUN else self.risk._portfolio_value
        self.risk.update_portfolio_value(pv)

        eod = self._is_eod_window()
        if eod:
            log.info("EOD window — closing all positions before market close")

        self._handle_exits(eod=eod)

        if not eod and not self.risk.daily_loss_limit_hit():
            self._handle_entries()

        log.info(
            "=== Cycle end — open positions: %d | daily P&L: %+.2f%% ===",
            len(self.open_positions),
            self.risk.daily_pnl_pct * 100,
        )

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def start(self) -> None:
        log.info("=" * 60)
        log.info("Robinhood Trading Bot starting")
        log.info("  DRY_RUN  : %s", cfg.DRY_RUN)
        log.info("  Strategy : momentum breakout + short squeeze")
        log.info("  Target   : ~2%% monthly")
        log.info("  Stop     : %.1f%%   Target: %.1f%%", cfg.STOP_LOSS_PCT * 100, cfg.TAKE_PROFIT_PCT * 100)
        log.info("=" * 60)

        if not cfg.DRY_RUN:
            ok = self.broker.login()
            if not ok:
                log.critical("Login failed — exiting")
                sys.exit(1)
            pv = self.broker.get_portfolio_value()
            self.risk = RiskManager(pv)
            self.risk.start_new_day(pv)
            log.info("Portfolio value: $%.2f", pv)
        else:
            log.info("DRY RUN mode — no real orders will be placed")

        self._sync_positions_from_broker()

        # Schedule the recurring cycle.
        schedule.every(cfg.SCAN_INTERVAL_MINUTES).minutes.do(self.run_cycle)
        # Also schedule a new-day reset at 9:31 ET (after open).
        schedule.every().day.at("09:31").do(self._new_day_reset)

        # Run immediately once on startup.
        self.run_cycle()

        log.info(
            "Bot running — scanning every %d minutes. Press Ctrl+C to stop.",
            cfg.SCAN_INTERVAL_MINUTES,
        )
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            self.shutdown()

    def _new_day_reset(self) -> None:
        pv = self.broker.get_portfolio_value() if not cfg.DRY_RUN else self.risk._portfolio_value
        self.risk.start_new_day(pv)
        log.info("New trading day — daily P&L reset. Portfolio: $%.2f", pv)

    def shutdown(self) -> None:
        log.info("Shutdown requested")
        self.broker.cancel_all_open_orders()
        self.broker.logout()
        self._print_summary()

    def _print_summary(self) -> None:
        log.info("-" * 50)
        log.info("Session trade log (%d trades):", len(self._trade_log))
        for trade in self._trade_log:
            log.info(
                "  %s  %-6s  %-8s  %d shares @ $%.2f  [%s]",
                trade["time"], trade["side"], trade["ticker"],
                trade["shares"], trade["price"], trade["reason"],
            )
        log.info("-" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
