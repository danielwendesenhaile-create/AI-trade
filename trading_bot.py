"""
MT5 Automated Trading Bot
Modular, production-grade implementation with dynamic position sizing,
risk management, and a continuous execution loop.
"""

import time
import logging
import sys
from datetime import datetime
from typing import Optional

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

SYMBOL      = "XAUUSD"       # Trading instrument
TIMEFRAME   = mt5.TIMEFRAME_M15
CANDLES     = 200             # Historical candles to fetch for analysis
LOOP_SLEEP  = 60              # Seconds between each loop iteration

# Risk: fraction of balance to risk per trade
RISK_FRACTION = 0.20          # 20%

# Hard-coded SL / TP distances in POINTS (1 point = smallest price move)
# Override these per-symbol as needed.
SL_POINTS = 150               # Stop-loss distance in points
TP_POINTS = 300               # Take-profit distance in points  (1:2 R/R)

# Minimum/maximum lot sizes allowed (broker-specific, adjust as needed)
LOT_MIN  = 0.01
LOT_MAX  = 100.0
LOT_STEP = 0.01               # Lot rounding step

MAGIC_NUMBER = 20240101       # Unique magic number to identify bot orders
SLIPPAGE     = 10             # Max slippage in points


# ---------------------------------------------------------------------------
# SECTION 2 — MT5 CONNECTION HELPERS
# ---------------------------------------------------------------------------

def connect_mt5(login: Optional[int] = None,
                password: Optional[str] = None,
                server: Optional[str] = None) -> bool:
    """
    Initialize and authenticate with the MT5 terminal.

    Parameters can be omitted to reuse the terminal's active session.
    Returns True on success, False on failure.
    """
    if not mt5.initialize(login=login, password=password, server=server):
        log.error("mt5.initialize() failed — error: %s", mt5.last_error())
        return False

    info = mt5.terminal_info()
    if info is None:
        log.error("Could not retrieve terminal info after init.")
        mt5.shutdown()
        return False

    log.info("MT5 connected  |  build=%s  |  connected=%s", info.build, info.connected)
    return True


def shutdown_mt5() -> None:
    """Cleanly close the MT5 connection."""
    mt5.shutdown()
    log.info("MT5 connection closed.")


# ---------------------------------------------------------------------------
# SECTION 3 — MARKET DATA
# ---------------------------------------------------------------------------

def fetch_candles(symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
    """
    Fetch the most recent `count` completed OHLCV candles for `symbol`.

    Returns a DataFrame with columns: time, open, high, low, close, tick_volume.
    The most recent (potentially incomplete) candle is excluded by dropping index 0
    from the tail — MT5 returns candles newest-last, so we drop the last row.
    Returns None on error.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
    if rates is None or len(rates) == 0:
        log.error("copy_rates_from_pos failed for %s — %s", symbol, mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})

    # Drop the last row: it is the still-forming current candle
    df = df.iloc[:-1].reset_index(drop=True)

    return df[["time", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# SECTION 4 — STRATEGY / SIGNAL ENGINE
# ---------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> str:
    """
    Analyse the OHLCV DataFrame and return 'BUY', 'SELL', or 'HOLD'.

    This function implements a dual-EMA crossover strategy as a reference.
    Replace or extend with: ICT order blocks, liquidity sweeps, RSI divergence,
    Bollinger Bands, price-action patterns, or any custom logic.

    Strategy (EMA 20 / EMA 50 crossover):
      - BUY  when fast EMA crosses above slow EMA on the latest closed candle
      - SELL when fast EMA crosses below slow EMA on the latest closed candle
      - HOLD otherwise
    """
    if len(df) < 60:
        log.warning("Not enough candles to compute signal (%d < 60).", len(df))
        return "HOLD"

    close = df["close"]

    # --- Indicator computation ---
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()

    # Current vs previous bar values
    curr_fast, prev_fast = ema_fast.iloc[-1], ema_fast.iloc[-2]
    curr_slow, prev_slow = ema_slow.iloc[-1], ema_slow.iloc[-2]

    bullish_cross = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
    bearish_cross = (prev_fast >= prev_slow) and (curr_fast < curr_slow)

    # --- Optional: Add confluence filters here ---
    # e.g., RSI, ATR, session time, spread check, higher-timeframe bias, etc.

    if bullish_cross:
        log.info("Signal: BUY  (EMA%d crossed above EMA%d)", 20, 50)
        return "BUY"

    if bearish_cross:
        log.info("Signal: SELL (EMA%d crossed below EMA%d)", 20, 50)
        return "SELL"

    log.info("Signal: HOLD (no crossover detected)")
    return "HOLD"


# ---------------------------------------------------------------------------
# SECTION 5 — DYNAMIC POSITION SIZING & RISK MANAGEMENT
# ---------------------------------------------------------------------------

def get_account_balance() -> Optional[float]:
    """Fetch live account balance. Returns None on failure."""
    info = mt5.account_info()
    if info is None:
        log.error("mt5.account_info() failed — %s", mt5.last_error())
        return None
    return info.balance


def calculate_lot_size(symbol: str, sl_points: int) -> Optional[float]:
    """
    Calculate lot size so that a stop-loss hit equals exactly RISK_FRACTION
    of the current account balance.

    Formula:
        cash_at_risk  = balance * RISK_FRACTION
        pip_value     = (contract_size * point) / price   [for non-USD quote]
        lot_size      = cash_at_risk / (sl_points * value_per_point_per_lot)

    MT5 provides tick_value (profit/loss per 1 lot per 1 tick move) and
    tick_size (size of 1 tick in price terms). We convert sl_points to ticks:
        ticks_in_sl = sl_points * point / tick_size
        risk_per_lot = ticks_in_sl * tick_value

    Returns the rounded lot size, or None on any failure.
    """
    balance = get_account_balance()
    if balance is None:
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        log.error("symbol_info() failed for %s — %s", symbol, mt5.last_error())
        return None

    if not symbol_info.visible:
        # Attempt to make symbol visible in Market Watch
        if not mt5.symbol_select(symbol, True):
            log.error("Cannot select symbol %s.", symbol)
            return None

    point      = symbol_info.point          # Smallest price increment
    tick_size  = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value

    if tick_size == 0 or tick_value == 0:
        log.error("Invalid tick data for %s: tick_size=%s tick_value=%s",
                  symbol, tick_size, tick_value)
        return None

    cash_at_risk    = balance * RISK_FRACTION
    sl_price_range  = sl_points * point
    ticks_in_sl     = sl_price_range / tick_size
    risk_per_lot    = ticks_in_sl * tick_value

    if risk_per_lot <= 0:
        log.error("risk_per_lot is non-positive (%s); cannot size position.", risk_per_lot)
        return None

    raw_lot = cash_at_risk / risk_per_lot

    # Round down to the nearest LOT_STEP and clamp within broker limits
    lot = max(LOT_MIN, min(LOT_MAX, round(raw_lot - (raw_lot % LOT_STEP), 2)))

    log.info(
        "Position sizing | balance=%.2f | risk=%.2f%% | cash_at_risk=%.2f | "
        "sl_points=%d | risk_per_lot=%.4f | raw_lot=%.4f | lot=%.2f",
        balance, RISK_FRACTION * 100, cash_at_risk,
        sl_points, risk_per_lot, raw_lot, lot,
    )
    return lot


# ---------------------------------------------------------------------------
# SECTION 6 — OPEN POSITION DETECTION
# ---------------------------------------------------------------------------

def has_open_position(symbol: str) -> bool:
    """
    Return True if there is already an open position for `symbol` opened
    by this bot (identified by MAGIC_NUMBER).
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        # None means API error; treat conservatively as having a position
        log.warning("positions_get() returned None — %s", mt5.last_error())
        return True
    for pos in positions:
        if pos.magic == MAGIC_NUMBER:
            return True
    return False


# ---------------------------------------------------------------------------
# SECTION 7 — ORDER EXECUTION
# ---------------------------------------------------------------------------

def get_current_price(symbol: str, order_type: int) -> Optional[float]:
    """
    Return the current ASK (for BUY) or BID (for SELL) price.
    order_type: mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("symbol_info_tick() failed for %s — %s", symbol, mt5.last_error())
        return None
    return tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid


def execute_order(symbol: str, signal: str, lot: float) -> bool:
    """
    Send a market order based on the signal direction ('BUY' or 'SELL').

    Stop-loss and take-profit are calculated as fixed point distances from
    the entry price (SL_POINTS / TP_POINTS constants).

    Returns True if the order was successfully accepted, False otherwise.
    """
    if signal not in ("BUY", "SELL"):
        log.error("execute_order called with invalid signal: %s", signal)
        return False

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    price = get_current_price(symbol, order_type)
    if price is None:
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        log.error("symbol_info() failed — %s", mt5.last_error())
        return False

    point = symbol_info.point
    digits = symbol_info.digits

    if signal == "BUY":
        sl = round(price - SL_POINTS * point, digits)
        tp = round(price + TP_POINTS * point, digits)
    else:
        sl = round(price + SL_POINTS * point, digits)
        tp = round(price - TP_POINTS * point, digits)

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
        "comment":      f"Bot {signal}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log.info(
        "Sending %s order | symbol=%s | lot=%.2f | price=%s | sl=%s | tp=%s",
        signal, symbol, lot, price, sl, tp,
    )

    result = mt5.order_send(request)
    if result is None:
        log.error("order_send() returned None — %s", mt5.last_error())
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            "Order ACCEPTED | ticket=%s | deal=%s | price=%.5f | lot=%.2f",
            result.order, result.deal, result.price, result.volume,
        )
        return True

    log.error(
        "Order REJECTED | retcode=%s (%s) | comment=%s",
        result.retcode, _retcode_description(result.retcode), result.comment,
    )
    return False


def _retcode_description(retcode: int) -> str:
    """Map common MT5 return codes to human-readable strings."""
    codes = {
        mt5.TRADE_RETCODE_DONE:            "Done",
        mt5.TRADE_RETCODE_REQUOTE:         "Requote",
        mt5.TRADE_RETCODE_REJECT:          "Rejected",
        mt5.TRADE_RETCODE_CANCEL:          "Cancelled",
        mt5.TRADE_RETCODE_PLACED:          "Order placed",
        mt5.TRADE_RETCODE_DONE_PARTIAL:    "Partial fill",
        mt5.TRADE_RETCODE_ERROR:           "Common error",
        mt5.TRADE_RETCODE_TIMEOUT:         "Timeout",
        mt5.TRADE_RETCODE_INVALID:         "Invalid request",
        mt5.TRADE_RETCODE_INVALID_VOLUME:  "Invalid volume",
        mt5.TRADE_RETCODE_INVALID_PRICE:   "Invalid price",
        mt5.TRADE_RETCODE_INVALID_STOPS:   "Invalid stops",
        mt5.TRADE_RETCODE_NO_MONEY:        "Insufficient funds",
        mt5.TRADE_RETCODE_PRICE_CHANGED:   "Price changed",
        mt5.TRADE_RETCODE_OFF_QUOTES:      "Off quotes",
        mt5.TRADE_RETCODE_CONNECTION:      "No connection",
        mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
    }
    return codes.get(retcode, f"Unknown({retcode})")


# ---------------------------------------------------------------------------
# SECTION 8 — MAIN AUTOMATION LOOP
# ---------------------------------------------------------------------------

def run_bot() -> None:
    """
    Continuous execution loop:
      1. Fetch market data
      2. Compute trading signal
      3. Skip if a position is already open
      4. Size the position
      5. Execute the order
      6. Sleep until the next candle close
    """
    log.info("=" * 60)
    log.info("  MT5 Trading Bot started  |  %s", datetime.utcnow().isoformat())
    log.info("  Symbol=%s  Timeframe=%s  Risk=%.0f%%", SYMBOL, TIMEFRAME, RISK_FRACTION * 100)
    log.info("=" * 60)

    if not connect_mt5():
        log.critical("Failed to connect to MT5. Exiting.")
        sys.exit(1)

    try:
        while True:
            loop_start = datetime.utcnow()
            log.info("--- Loop tick: %s ---", loop_start.strftime("%Y-%m-%d %H:%M:%S UTC"))

            # Step 1: Fetch candle data
            df = fetch_candles(SYMBOL, TIMEFRAME, CANDLES)
            if df is None:
                log.warning("Skipping tick — could not fetch candles.")
                time.sleep(LOOP_SLEEP)
                continue

            log.info("Candles fetched: %d rows, latest close time: %s",
                     len(df), df["time"].iloc[-1])

            # Step 2: Generate trading signal
            signal = compute_signal(df)

            # Step 3: Skip HOLD or if position already open
            if signal == "HOLD":
                log.info("HOLD — no trade action taken.")
                time.sleep(LOOP_SLEEP)
                continue

            if has_open_position(SYMBOL):
                log.info("Position already open for %s — skipping new entry.", SYMBOL)
                time.sleep(LOOP_SLEEP)
                continue

            # Step 4: Calculate lot size with strict risk management
            lot = calculate_lot_size(SYMBOL, SL_POINTS)
            if lot is None:
                log.warning("Could not calculate lot size — skipping trade.")
                time.sleep(LOOP_SLEEP)
                continue

            # Step 5: Execute the order
            success = execute_order(SYMBOL, signal, lot)
            if not success:
                log.warning("Order execution failed for signal=%s.", signal)

            # Sleep until next iteration
            elapsed = (datetime.utcnow() - loop_start).total_seconds()
            sleep_for = max(0, LOOP_SLEEP - elapsed)
            log.info("Sleeping %.1f seconds until next tick...", sleep_for)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as exc:
        log.exception("Unhandled exception in main loop: %s", exc)
    finally:
        shutdown_mt5()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_bot()
