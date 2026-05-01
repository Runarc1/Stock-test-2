"""
Robinhood broker wrapper.

Thin layer around robin_stocks that:
  - Handles login (with optional MFA/TOTP).
  - Exposes buy/sell in a dry-run-safe way.
  - Returns portfolio equity and current positions.

All order placement goes through `place_order()` which checks DRY_RUN and
logs the intent before touching the live API.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pyotp
import robin_stocks.robinhood as rh

import config as cfg

log = logging.getLogger(__name__)


class Broker:
    def __init__(self) -> None:
        self._logged_in = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self) -> bool:
        if not cfg.RH_USERNAME or not cfg.RH_PASSWORD:
            log.error("RH_USERNAME / RH_PASSWORD not set in .env")
            return False
        try:
            mfa_code: Optional[str] = None
            if cfg.RH_MFA_SECRET:
                totp = pyotp.TOTP(cfg.RH_MFA_SECRET)
                mfa_code = totp.now()
                log.debug("Generated TOTP code: %s", mfa_code)

            rh.login(
                username=cfg.RH_USERNAME,
                password=cfg.RH_PASSWORD,
                mfa_code=mfa_code,
                store_session=True,
            )
            self._logged_in = True
            log.info("Logged in to Robinhood as %s", cfg.RH_USERNAME)
            return True
        except Exception as exc:
            log.error("Robinhood login failed: %s", exc)
            return False

    def logout(self) -> None:
        if self._logged_in:
            rh.logout()
            self._logged_in = False
            log.info("Logged out of Robinhood")

    # ------------------------------------------------------------------
    # Portfolio / position queries
    # ------------------------------------------------------------------

    def get_portfolio_value(self) -> float:
        """Total equity (cash + positions) in the account."""
        try:
            profile = rh.load_portfolio_profile()
            equity = profile.get("equity") or profile.get("extended_hours_equity") or 0
            return float(equity)
        except Exception as exc:
            log.error("Could not fetch portfolio value: %s", exc)
            return 0.0

    def get_buying_power(self) -> float:
        """Cash available to trade."""
        try:
            profile = rh.load_account_profile()
            return float(profile.get("buying_power") or 0)
        except Exception as exc:
            log.error("Could not fetch buying power: %s", exc)
            return 0.0

    def get_open_positions(self) -> dict[str, dict]:
        """
        Return {ticker: {shares, average_buy_price}} for all non-zero positions.
        """
        try:
            raw = rh.get_open_stock_positions()
            positions: dict[str, dict] = {}
            for pos in raw:
                qty = float(pos.get("quantity") or 0)
                if qty <= 0:
                    continue
                instrument_url = pos.get("instrument", "")
                try:
                    info = rh.get_instrument_by_url(instrument_url)
                    ticker = info.get("symbol", "UNKNOWN")
                except Exception:
                    ticker = "UNKNOWN"
                positions[ticker] = {
                    "shares": qty,
                    "average_buy_price": float(pos.get("average_buy_price") or 0),
                }
            return positions
        except Exception as exc:
            log.error("Could not fetch positions: %s", exc)
            return {}

    def get_current_price(self, ticker: str) -> Optional[float]:
        try:
            data = rh.get_latest_price(ticker)
            if data and data[0] is not None:
                return float(data[0])
            return None
        except Exception as exc:
            log.warning("Could not fetch price for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: str,         # "buy" | "sell"
        shares: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "gfd",  # good for day
    ) -> Optional[dict]:
        """
        Place a market or limit order.
        In DRY_RUN mode, logs the order and returns a fake result dict.
        """
        shares = round(shares, 6)
        action = f"[DRY RUN] " if cfg.DRY_RUN else ""

        if order_type == "limit" and limit_price is not None:
            log.info(
                "%s%s %d shares of %s @ limit $%.2f (%s)",
                action, side.upper(), shares, ticker, limit_price, time_in_force,
            )
        else:
            log.info(
                "%s%s %d shares of %s @ market (%s)",
                action, side.upper(), shares, ticker, time_in_force,
            )

        if cfg.DRY_RUN:
            return {"id": "dry-run", "state": "filled", "ticker": ticker}

        try:
            if side == "buy":
                if order_type == "limit" and limit_price:
                    result = rh.order_buy_limit(ticker, shares, limit_price, time_in_force)
                else:
                    result = rh.order_buy_market(ticker, shares, time_in_force)
            else:
                if order_type == "limit" and limit_price:
                    result = rh.order_sell_limit(ticker, shares, limit_price, time_in_force)
                else:
                    result = rh.order_sell_market(ticker, shares, time_in_force)

            log.info("Order placed for %s: %s", ticker, result.get("id", "unknown"))
            return result
        except Exception as exc:
            log.error("Order failed for %s: %s", ticker, exc)
            return None

    def buy_market(self, ticker: str, shares: int) -> Optional[dict]:
        return self.place_order(ticker, "buy", shares, "market")

    def sell_market(self, ticker: str, shares: int) -> Optional[dict]:
        return self.place_order(ticker, "sell", shares, "market")

    def cancel_all_open_orders(self) -> None:
        if cfg.DRY_RUN:
            log.info("[DRY RUN] Would cancel all open orders")
            return
        try:
            rh.cancel_all_open_orders()
            log.info("All open orders cancelled")
        except Exception as exc:
            log.error("Could not cancel orders: %s", exc)
