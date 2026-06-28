"""
Trading Bot Dashboard — Flask backend
Serves live metrics, trade history, and log tail from backtest results.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template

# Allow importing backtest from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

BASE = Path(__file__).parent.parent

# ── Run backtest once at startup and cache results ────────────────────────────
_cache: dict = {}

def load_results() -> dict:
    if _cache:
        return _cache

    import numpy as np
    import pandas as pd
    from backtest import (
        fetch_data, run_backtest,
        STARTING_BALANCE, SL_ATR_MULT, TP_ATR_MULT,
        EMA_FAST, EMA_SLOW, RSI_LEN, ATR_LEN, RISK_FRACTION,
    )

    df     = fetch_data()
    result = run_backtest(df)
    trades = result["trades"]
    final  = result["final_balance"]

    if not trades:
        _cache.update({"trades": [], "metrics": {}, "equity": [], "weekly": []})
        return _cache

    tdf = pd.DataFrame(trades)
    wins   = tdf[tdf["result"] == "WIN"]
    losses = tdf[tdf["result"] == "LOSS"]

    win_rate      = len(wins) / len(tdf) * 100
    profit_factor = (wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum())
                     if len(losses) > 0 else float("inf"))
    avg_win  = float(wins["pnl_usd"].mean())   if len(wins)   else 0
    avg_loss = float(losses["pnl_usd"].mean()) if len(losses) else 0

    balances  = [STARTING_BALANCE] + tdf["balance"].tolist()
    peaks     = pd.Series(balances).cummax()
    drawdowns = ((pd.Series(balances) - peaks) / peaks * 100)
    max_dd    = float(drawdowns.min())

    # Equity curve
    equity = [{"x": i, "y": round(b, 2)} for i, b in enumerate(balances)]

    # Weekly PnL
    tdf["week"] = pd.to_datetime(tdf["time"]).dt.to_period("W").astype(str)
    weekly_raw = tdf.groupby("week").agg(
        pnl=("pnl_usd", "sum"),
        trades=("pnl_usd", "count"),
        wins=("result", lambda x: (x == "WIN").sum()),
    ).reset_index()
    weekly = [
        {
            "week":   r["week"],
            "pnl":    round(r["pnl"], 2),
            "trades": int(r["trades"]),
            "wr":     round(r["wins"] / r["trades"] * 100, 1),
        }
        for _, r in weekly_raw.iterrows()
    ]

    # Per-trade list (all trades)
    trade_list = tdf.to_dict(orient="records")

    _cache.update({
        "trades":  trade_list,
        "weekly":  weekly,
        "equity":  equity,
        "metrics": {
            "starting_balance": STARTING_BALANCE,
            "final_balance":    round(final, 2),
            "net_pnl":          round(final - STARTING_BALANCE, 2),
            "pct_return":       round((final - STARTING_BALANCE) / STARTING_BALANCE * 100, 1),
            "total_trades":     len(tdf),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(win_rate, 1),
            "profit_factor":    round(profit_factor, 2),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "rr_ratio":         round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
            "max_drawdown":     round(max_dd, 2),
            "strategy":         f"EMA {EMA_FAST}/{EMA_SLOW} + RSI({RSI_LEN}) + ATR({ATR_LEN})",
            "risk_pct":         int(RISK_FRACTION * 100),
            "sl_mult":          SL_ATR_MULT,
            "tp_mult":          TP_ATR_MULT,
            "generated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
    })
    return _cache


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    data = load_results()
    return jsonify(data["metrics"])


@app.route("/api/equity")
def api_equity():
    data = load_results()
    return jsonify(data["equity"])


@app.route("/api/trades")
def api_trades():
    data = load_results()
    return jsonify(data["trades"])


@app.route("/api/weekly")
def api_weekly():
    data = load_results()
    return jsonify(data["weekly"])


if __name__ == "__main__":
    print("Loading backtest data...")
    load_results()
    print("Dashboard running at http://localhost:5000")
    app.run(debug=False, port=5000)
