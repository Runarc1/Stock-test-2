"""
Trading bot configuration and strategy parameters.

Target: ~2% monthly return via momentum breakouts and short-squeeze setups.
Adjust these values to tune the bot's behavior. All parameters are documented
with the reasoning behind the chosen defaults.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Robinhood credentials (loaded from .env)
# ---------------------------------------------------------------------------
RH_USERNAME: str = os.getenv("RH_USERNAME", "")
RH_PASSWORD: str = os.getenv("RH_PASSWORD", "")
RH_MFA_SECRET: str = os.getenv("RH_MFA_SECRET", "")

# When True the bot logs every decision but places no real orders.
# ALWAYS start with DRY_RUN=True until you've validated the strategy.
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() != "false"

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Universe / watchlist
# ---------------------------------------------------------------------------

# Hard-coded watchlist always included in scans (comma-separated in .env).
EXTRA_WATCHLIST: list[str] = [
    t.strip().upper()
    for t in os.getenv("WATCHLIST", "").split(",")
    if t.strip()
]

# These ETF tickers are used to gauge overall market direction.
# The bot will refuse to open new longs if SPY is in a confirmed downtrend.
MARKET_ETF: str = "SPY"

# Minimum average daily volume (shares). Below this the stock is too illiquid
# to trade safely without significant slippage.
MIN_AVG_VOLUME: int = 500_000

# Price range for eligible stocks. Avoid sub-$5 (penny stock volatility / manipulation)
# and above $400 (high capital requirement per share makes position sizing awkward).
MIN_PRICE: float = 5.0
MAX_PRICE: float = 400.0

# Minimum market cap in USD. Keeps the bot away from micro-caps that are easy to
# pump/dump and have unreliable short-interest data.
MIN_MARKET_CAP: int = 300_000_000  # $300 M

# ---------------------------------------------------------------------------
# Short-squeeze detection parameters
# ---------------------------------------------------------------------------

# Minimum short interest as a fraction of float to qualify as a squeeze candidate.
# 10 %+ means there's a real short base that can be forced to cover.
MIN_SHORT_FLOAT: float = 0.10  # 10 %

# Days-to-cover (short interest / avg daily volume). A high DTC means it would
# take many days for all shorts to cover, amplifying a squeeze.
MIN_DAYS_TO_COVER: float = 3.0

# Volume today must be at least this multiple of the 20-day average volume to
# signal that a squeeze may already be in progress.
SQUEEZE_VOLUME_MULTIPLIER: float = 2.0

# Maximum RSI still allowed for a squeeze entry. Above 78 the stock is likely
# already mid-squeeze and risk/reward deteriorates.
SQUEEZE_MAX_RSI: float = 78.0

# ---------------------------------------------------------------------------
# Momentum breakout parameters
# ---------------------------------------------------------------------------

# RSI window (standard 14 periods).
RSI_PERIOD: int = 14

# RSI must be in this range to confirm momentum without being overbought.
# 50–70 = building strength, not yet exhausted.
MOMENTUM_RSI_MIN: float = 50.0
MOMENTUM_RSI_MAX: float = 70.0

# MACD settings (fast EMA, slow EMA, signal line).
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# Price must be above the 50-day simple moving average for a valid momentum entry.
# This keeps the bot on the right side of the medium-term trend.
MOMENTUM_SMA_PERIOD: int = 50

# Price must also be above the 20-day SMA (short-term trend confirmation).
MOMENTUM_SMA_SHORT: int = 20

# A breakout day needs volume > this multiple of the 20-day average.
BREAKOUT_VOLUME_MULTIPLIER: float = 1.5

# Stock must be within this fraction of its 52-week high to be "breaking out".
# 0.90 = within 10 % of the 52-week high.
NEAR_HIGH_THRESHOLD: float = 0.90

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

# Stop-loss: close position if it falls more than this fraction from entry.
STOP_LOSS_PCT: float = 0.04   # -4 %

# Take-profit: close position if it gains more than this fraction from entry.
TAKE_PROFIT_PCT: float = 0.08  # +8 %   (2:1 reward-to-risk)

# Trailing stop activates once the position is up by TRAIL_ACTIVATE_PCT.
# After that, the stop trails the peak by TRAIL_STOP_PCT.
# Set TRAIL_ACTIVATE_PCT = None to disable trailing stop.
TRAIL_ACTIVATE_PCT: float = 0.05   # activate after +5 %
TRAIL_STOP_PCT: float = 0.03       # trail 3 % below peak

# Maximum number of simultaneous open positions.
MAX_POSITIONS: int = 5

# Maximum allocation per position as a fraction of total portfolio value.
# 15 % per position = never more than ~$1,500 in one name on a $10k account.
MAX_POSITION_PCT: float = 0.15

# If the portfolio loses more than this fraction in one day, stop trading for
# the rest of the day (circuit breaker).
DAILY_LOSS_LIMIT_PCT: float = 0.03  # -3 %

# Never risk more than this fraction of the portfolio in a single trade.
# Used by the Kelly-fraction position sizer.
MAX_RISK_PER_TRADE_PCT: float = 0.01  # 1 % of portfolio per trade

# ---------------------------------------------------------------------------
# Scheduling / timing
# ---------------------------------------------------------------------------

# Trading window (24-h Eastern time). The bot will only scan and place orders
# between these hours on weekdays.
MARKET_OPEN_HOUR: int = 10   # 10:00 AM ET  (skip first 30 min of chaos)
MARKET_CLOSE_HOUR: int = 15  # 3:00 PM ET   (stop before closing auction)
MARKET_CLOSE_MINUTE: int = 30

# How often (in minutes) the main loop re-scans the universe.
SCAN_INTERVAL_MINUTES: int = 15

# Force-close any open positions this many minutes before market close to avoid
# overnight gap risk.
EOD_CLOSE_MINUTES_BEFORE: int = 20

# ---------------------------------------------------------------------------
# S&P 500 and high-momentum universe (bot will scan these + EXTRA_WATCHLIST)
# Keeping this focused on liquid large/mid-caps reduces slippage risk.
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE: list[str] = [
    # High short-interest or recent squeeze candidates (updated manually)
    "GME", "AMC", "BBBY", "SPCE", "WISH",
    # High-momentum large-caps
    "NVDA", "AMD", "META", "TSLA", "AMZN", "GOOG", "MSFT", "AAPL",
    "NFLX", "CRM", "SHOP", "SNOW", "PLTR", "COIN", "MSTR",
    # Leveraged sector plays with momentum
    "SOXS", "SOXL", "TQQQ", "LABU",
    # Financials / energy with squeeze setups
    "SLB", "OXY", "DVN", "MRO",
    # Biotech (high short interest sector — use caution)
    "SGEN", "FATE", "BEAM", "EDIT",
]
