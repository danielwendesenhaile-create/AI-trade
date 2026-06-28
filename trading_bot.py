"""
MT5 Automated Trading Bot — XAUUSD M1 Scalper
Strategy: EMA 9/21 Cross + RSI(7) Momentum + ATR(14) Dynamic SL/TP
           with London/New York session filter and spread guard.
"""

import time
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SECTION 1 — CONFIGURATION
# ---------------------------------------------------------------------------

SYMBOL    = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
CANDLES   = 300           # Enough history for all indicators to warm up
LOOP_SLEEP = 60           # Seconds between loop iterations (1 candle = 60 s)

# Risk management
RISK_FRACTION = 0.20      # 20% of balance risked per trade

# ATR multipliers — SL = ATR * SL_ATR_MULT, TP = ATR * TP_ATR_MULT (1:2 R/R)
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0

# Spread guard: skip trade if live spread exceeds this many points
MAX_SPREAD_POINTS = 30    # ~3 pips on XAUUSD (adjust per broker)

# Lot size limits
LOT_MIN  = 0.01
LOT_MAX  = 100.0
LOT_STEP = 0.01

MAGIC_NUMBER = 20240102
SLIPPAGE     = 10         # Max deviation in points

# ---------------------------------------------------------------------------
# Active trading sessions (UTC hours, inclusive).
# Gold is most liquid during London (07–12) and New York (13–17) overlap.
# ---------------------------------------------------------------------------
SESSIONS = [
    (7, 12),   # London
    (13, 17),  # New York
]

# ---------------------------------------------------------------------------
# Strategy indicator parameters
# ---------------------------------------------------------------------------
EMA_FAST  = 9
EMA_SLOW  = 21
RSI_LEN   = 7
ATR_LEN   = 14

# RSI thresholds
RSI_BUY_MIN  = 50    # RSI must be above this to confirm BUY momentum
RSI_SELL_MAX = 50    # RSI must be below this to confirm SELL momentum
RSI_OB       = 75    # Overbought — suppress BUY signals
RSI_OS       = 25    # Oversold  — suppress SELL signals


# ---------------------------------------------------------------------------
# SECTION 2 — MT5 CONNECTION
# ---------------------------------------------------------------------------

def connect_mt5(login: Optional[int] = None,
                password: Optional[str] = None,
                server: Optional[str] = None) -> bool:
    """Initialize MT5 terminal connection. Returns True on success."""
    if not mt5.initialize(login=login, password=password, server=server):
        log.error("mt5.initialize() failed — %s", mt5.last_error())
        return False

    info = mt5.terminal_info()
    if info is None:
        log.error("terminal_info() returned None after init.")
        mt5.shutdown()
        return False

    log.info("MT5 connected | build=%s | connected=%s", info.build, info.connected)
    return True


def shutdown_mt5() -> None:
    mt5.shutdown()
    log.info("MT5 connection closed.")


# ---------------------------------------------------------------------------
# SECTION 3 — MARKET DATA
# ---------------------------------------------------------------------------

def fetch_candles(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    Fetch `count` completed OHLCV candles.
    The live (still-forming) candle is always stripped before returning.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
    if rates is None or len(rates) == 0:
        log.error("copy_rates_from_pos failed for %s — %s", symbol, mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df.iloc[:-1].reset_index(drop=True)   # drop live candle
    return df[["time", "open", "high", "low", "close", "volume"]]


def get_live_spread(symbol: str) -> Optional[float]:
    """Return current spread in points, or None on error."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return round((tick.ask - tick.bid) / info.point)


# ---------------------------------------------------------------------------
# SECTION 4 — INDICATORS
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI (standard implementation)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range."""
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA fast/slow, RSI, and ATR columns to the DataFrame."""
    df = df.copy()
    df["ema_fast"] = _ema(df["close"], EMA_FAST)
    df["ema_slow"] = _ema(df["close"], EMA_SLOW)
    df["rsi"]      = _rsi(df["close"], RSI_LEN)
    df["atr"]      = _atr(df, ATR_LEN)
    return df


# ---------------------------------------------------------------------------
# SECTION 5 — SESSION FILTER
# ---------------------------------------------------------------------------

def is_active_session() -> bool:
    """
    Return True only if the current UTC hour falls inside a configured
    liquid trading session (London or New York).
    Gold spreads widen sharply outside these windows, killing scalp edge.
    """
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    for start, end in SESSIONS:
        if start <= hour < end:
            return True
    return False


# ---------------------------------------------------------------------------
# SECTION 6 — STRATEGY SIGNAL ENGINE
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> Tuple[str, float]:
    """
    XAUUSD M1 Scalping Strategy — EMA 9/21 Cross + RSI(7) + ATR(14)

    Entry rules
    -----------
    BUY  : EMA9 crosses above EMA21 on the last closed candle
           AND RSI(7) > 50  (upside momentum confirmed)
           AND RSI(7) < 75  (not already overbought)

    SELL : EMA9 crosses below EMA21 on the last closed candle
           AND RSI(7) < 50  (downside momentum confirmed)
           AND RSI(7) > 25  (not already oversold)

    SL/TP distances (returned as price distance, not points)
    ---------------------------------------------------------
    SL = ATR(14) * SL_ATR_MULT  →  volatility-adaptive stop
    TP = ATR(14) * TP_ATR_MULT  →  2× the stop (1:2 R/R minimum)

    Returns
    -------
    (signal, atr_value) where signal ∈ {'BUY', 'SELL', 'HOLD'}
    atr_value is the latest ATR (0.0 when HOLD).
    """
    min_candles = max(EMA_SLOW, RSI_LEN, ATR_LEN) * 3
    if len(df) < min_candles:
        log.warning("Insufficient candles (%d < %d) — HOLD.", len(df), min_candles)
        return "HOLD", 0.0

    df = compute_indicators(df)

    # Latest and previous completed candle
    c  = df.iloc[-1]   # current (last closed)
    p  = df.iloc[-2]   # previous

    fast_cross_up   = (p["ema_fast"] <= p["ema_slow"]) and (c["ema_fast"] > c["ema_slow"])
    fast_cross_down = (p["ema_fast"] >= p["ema_slow"]) and (c["ema_fast"] < c["ema_slow"])

    rsi  = c["rsi"]
    atr  = c["atr"]

    log.info(
        "Indicators | EMA_fast=%.2f EMA_slow=%.2f RSI=%.1f ATR=%.4f",
        c["ema_fast"], c["ema_slow"], rsi, atr,
    )

    if fast_cross_up and RSI_BUY_MIN < rsi < RSI_OB:
        log.info("Signal: BUY  | EMA9 crossed above EMA21 | RSI=%.1f", rsi)
        return "BUY", atr

    if fast_cross_down and RSI_OS < rsi < RSI_SELL_MAX:
        log.info("Signal: SELL | EMA9 crossed below EMA21 | RSI=%.1f", rsi)
        return "SELL", atr

    log.info("Signal: HOLD | no valid crossover + RSI confluence")
    return "HOLD", 0.0


# ---------------------------------------------------------------------------
# SECTION 7 — DYNAMIC POSITION SIZING
# ---------------------------------------------------------------------------

def get_account_balance() -> Optional[float]:
    info = mt5.account_info()
    if info is None:
        log.error("account_info() failed — %s", mt5.last_error())
        return None
    return info.balance


def calculate_lot_size(symbol: str, sl_price_distance: float) -> Optional[float]:
    """
    Compute lot size so that hitting the SL costs exactly RISK_FRACTION
    of the current balance.

    sl_price_distance : SL distance expressed as a raw price difference
                        (e.g. ATR * SL_ATR_MULT). Converted to points internally.

    Formula:
        cash_at_risk  = balance × RISK_FRACTION
        sl_in_ticks   = sl_price_distance / tick_size
        risk_per_lot  = sl_in_ticks × tick_value
        lot           = cash_at_risk / risk_per_lot
    """
    balance = get_account_balance()
    if balance is None:
        return None

    sym = mt5.symbol_info(symbol)
    if sym is None:
        log.error("symbol_info() failed for %s — %s", symbol, mt5.last_error())
        return None

    if not sym.visible:
        if not mt5.symbol_select(symbol, True):
            log.error("Cannot select symbol %s.", symbol)
            return None

    tick_size  = sym.trade_tick_size
    tick_value = sym.trade_tick_value

    if tick_size == 0 or tick_value == 0:
        log.error("Bad tick data: tick_size=%s tick_value=%s", tick_size, tick_value)
        return None

    cash_at_risk = balance * RISK_FRACTION
    sl_in_ticks  = sl_price_distance / tick_size
    risk_per_lot = sl_in_ticks * tick_value

    if risk_per_lot <= 0:
        log.error("risk_per_lot non-positive (%.6f) — cannot size.", risk_per_lot)
        return None

    raw_lot = cash_at_risk / risk_per_lot
    lot = max(LOT_MIN, min(LOT_MAX, round(raw_lot - (raw_lot % LOT_STEP), 2)))

    log.info(
        "Sizing | balance=%.2f | risk=%.0f%% | cash_risk=%.2f | "
        "SL_dist=%.4f | risk/lot=%.4f | raw=%.4f | lot=%.2f",
        balance, RISK_FRACTION * 100, cash_at_risk,
        sl_price_distance, risk_per_lot, raw_lot, lot,
    )
    return lot


# ---------------------------------------------------------------------------
# SECTION 8 — POSITION GUARD
# ---------------------------------------------------------------------------

def has_open_position(symbol: str) -> bool:
    """True if this bot already holds an open position for `symbol`."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        log.warning("positions_get() returned None — %s", mt5.last_error())
        return True   # conservative: assume open
    return any(p.magic == MAGIC_NUMBER for p in positions)


# ---------------------------------------------------------------------------
# SECTION 9 — ORDER EXECUTION
# ---------------------------------------------------------------------------

def get_current_price(symbol: str, order_type: int) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("symbol_info_tick() failed — %s", mt5.last_error())
        return None
    return tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid


def execute_order(symbol: str, signal: str, lot: float, atr: float) -> bool:
    """
    Place a market order with ATR-based SL and TP.

    SL distance = ATR × SL_ATR_MULT
    TP distance = ATR × TP_ATR_MULT   (≥ 1:2 R/R by default)
    """
    if signal not in ("BUY", "SELL"):
        return False

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    price = get_current_price(symbol, order_type)
    if price is None:
        return False

    sym = mt5.symbol_info(symbol)
    if sym is None:
        log.error("symbol_info() failed — %s", mt5.last_error())
        return False

    digits    = sym.digits
    sl_dist   = round(atr * SL_ATR_MULT, digits)
    tp_dist   = round(atr * TP_ATR_MULT, digits)

    if signal == "BUY":
        sl = round(price - sl_dist, digits)
        tp = round(price + tp_dist, digits)
    else:
        sl = round(price + sl_dist, digits)
        tp = round(price - tp_dist, digits)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    SLIPPAGE,
        "magic":        MAGIC_NUMBER,
        "comment":      f"EMA-RSI-ATR {signal}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log.info(
        "ORDER %s | lot=%.2f | price=%.5f | sl=%.5f | tp=%.5f | "
        "sl_dist=%.4f | tp_dist=%.4f | ATR=%.4f",
        signal, lot, price, sl, tp, sl_dist, tp_dist, atr,
    )

    result = mt5.order_send(request)
    if result is None:
        log.error("order_send() returned None — %s", mt5.last_error())
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            "ACCEPTED | ticket=%s | deal=%s | fill_price=%.5f | lot=%.2f",
            result.order, result.deal, result.price, result.volume,
        )
        return True

    log.error(
        "REJECTED | retcode=%s (%s) | %s",
        result.retcode, _retcode_str(result.retcode), result.comment,
    )
    return False


def _retcode_str(code: int) -> str:
    table = {
        mt5.TRADE_RETCODE_DONE:              "Done",
        mt5.TRADE_RETCODE_REQUOTE:           "Requote",
        mt5.TRADE_RETCODE_REJECT:            "Rejected",
        mt5.TRADE_RETCODE_CANCEL:            "Cancelled",
        mt5.TRADE_RETCODE_PLACED:            "Placed",
        mt5.TRADE_RETCODE_DONE_PARTIAL:      "Partial fill",
        mt5.TRADE_RETCODE_ERROR:             "Error",
        mt5.TRADE_RETCODE_TIMEOUT:           "Timeout",
        mt5.TRADE_RETCODE_INVALID:           "Invalid request",
        mt5.TRADE_RETCODE_INVALID_VOLUME:    "Invalid volume",
        mt5.TRADE_RETCODE_INVALID_PRICE:     "Invalid price",
        mt5.TRADE_RETCODE_INVALID_STOPS:     "Invalid stops",
        mt5.TRADE_RETCODE_NO_MONEY:          "Insufficient funds",
        mt5.TRADE_RETCODE_PRICE_CHANGED:     "Price changed",
        mt5.TRADE_RETCODE_OFF_QUOTES:        "Off quotes",
        mt5.TRADE_RETCODE_CONNECTION:        "No connection",
        mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
    }
    return table.get(code, f"Unknown({code})")


# ---------------------------------------------------------------------------
# SECTION 10 — MAIN AUTOMATION LOOP
# ---------------------------------------------------------------------------

def run_bot() -> None:
    """
    Main execution loop (runs every 60 seconds — one M1 candle):

      1. Session filter  — skip dead-market hours
      2. Spread guard    — skip if broker spread is too wide
      3. Fetch candles   — 300 completed M1 bars
      4. Compute signal  — EMA cross + RSI + ATR
      5. Position guard  — no duplicate entries
      6. Size lot        — ATR-based SL, 20% balance risk
      7. Execute order   — market order with SL/TP attached
      8. Sleep           — align to next candle close
    """
    log.info("=" * 65)
    log.info("  XAUUSD M1 SCALPER  |  %s UTC", datetime.utcnow().isoformat())
    log.info("  Strategy : EMA %d/%d Cross + RSI(%d) + ATR(%d) SL/TP",
             EMA_FAST, EMA_SLOW, RSI_LEN, ATR_LEN)
    log.info("  Risk     : %.0f%% per trade | SL×%.1f ATR | TP×%.1f ATR",
             RISK_FRACTION * 100, SL_ATR_MULT, TP_ATR_MULT)
    log.info("  Sessions : London 07–12 UTC | New York 13–17 UTC")
    log.info("=" * 65)

    if not connect_mt5():
        log.critical("MT5 connection failed. Exiting.")
        sys.exit(1)

    try:
        while True:
            tick_start = datetime.utcnow()
            log.info("── Tick %s ──", tick_start.strftime("%Y-%m-%d %H:%M:%S UTC"))

            # ── 1. Session filter ──────────────────────────────────────────
            if not is_active_session():
                log.info("Outside active session — sleeping 60 s.")
                time.sleep(LOOP_SLEEP)
                continue

            # ── 2. Spread guard ────────────────────────────────────────────
            spread = get_live_spread(SYMBOL)
            if spread is None:
                log.warning("Could not read spread — skipping tick.")
                time.sleep(LOOP_SLEEP)
                continue
            if spread > MAX_SPREAD_POINTS:
                log.info("Spread too wide (%d pts > %d) — skipping tick.",
                         spread, MAX_SPREAD_POINTS)
                time.sleep(LOOP_SLEEP)
                continue
            log.info("Spread OK: %d pts", spread)

            # ── 3. Fetch candles ───────────────────────────────────────────
            df = fetch_candles(SYMBOL, TIMEFRAME, CANDLES)
            if df is None:
                log.warning("Candle fetch failed — skipping tick.")
                time.sleep(LOOP_SLEEP)
                continue
            log.info("Candles: %d bars | last close: %s", len(df), df["time"].iloc[-1])

            # ── 4. Strategy signal ─────────────────────────────────────────
            signal, atr = compute_signal(df)

            if signal == "HOLD":
                time.sleep(LOOP_SLEEP)
                continue

            # ── 5. Position guard ──────────────────────────────────────────
            if has_open_position(SYMBOL):
                log.info("Position already open — no new entry.")
                time.sleep(LOOP_SLEEP)
                continue

            # ── 6. Lot sizing ──────────────────────────────────────────────
            sl_distance = atr * SL_ATR_MULT
            lot = calculate_lot_size(SYMBOL, sl_distance)
            if lot is None:
                log.warning("Lot sizing failed — skipping trade.")
                time.sleep(LOOP_SLEEP)
                continue

            # ── 7. Execute ─────────────────────────────────────────────────
            success = execute_order(SYMBOL, signal, lot, atr)
            if not success:
                log.warning("Order execution failed for signal=%s.", signal)

            # ── 8. Sleep to next candle ────────────────────────────────────
            elapsed = (datetime.utcnow() - tick_start).total_seconds()
            sleep_for = max(0.0, LOOP_SLEEP - elapsed)
            log.info("Sleeping %.1f s to next candle...", sleep_for)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    except Exception as exc:
        log.exception("Unhandled exception: %s", exc)
    finally:
        shutdown_mt5()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_bot()
