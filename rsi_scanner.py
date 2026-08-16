"""
Alpaca (PAPER) — RSI Oversold Scanner & Auto-Buyer
====================================================
Designed to be run ONCE per invocation. GitHub Actions calls this on a
schedule (see .github/workflows/scan.yml) — that's what makes it a
"background bot": you don't run this yourself, GitHub's servers do,
on a timer, whether your computer is on or not.

WHY ALPACA INSTEAD OF TRADING 212
-----------------------------------
Trading 212's API Terms explicitly prohibit using their API for
algorithmic trading (Section 4.2a) — this kind of bot would violate that.
Alpaca is a US-regulated broker (FINRA/SIPC) built specifically for
API-based automated trading, with a free Paper Trading environment
equivalent to Trading 212's Demo. No terms conflict here.

WHAT IT DOES
------------
1. Downloads recent daily price history for each stock in WATCHLIST from
   Yahoo Finance (kept from the original version — works fine alongside
   Alpaca; Alpaca also has its own market data API if you'd rather
   consolidate onto one provider later).
2. Calculates 14-period RSI for each.
3. If RSI < 30 and you don't already hold a position, places a market buy
   for a fixed dollar amount (ORDER_VALUE_USD) using Alpaca's fractional
   "notional" order type — no manual share-quantity math needed.
4. Once that buy fills, places a limit sell 3% above the fill price.

WHAT IT DOESN'T DO
------------------
- No stop-loss. Only the 3% upside target is managed here.
- No memory between runs beyond what's already visible in your account.
- Not financial advice; this is a paper account for a reason.

API KEYS
--------
Read from environment variables ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.
In GitHub Actions these come from repository secrets — never written into
this file or committed to the repo.

Get them from: app.alpaca.markets → sign up (free) → make sure the toggle
in the top-left says "Paper Trading" (not "Live Trading") → API Keys → 
Generate New Key. Paper and live keys are completely separate pairs tied to
separate accounts, similar to how Trading 212 separates Demo and Live.
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rsi_scanner")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# SAFETY SWITCH — must be explicitly True or the script refuses to run.
I_CONFIRM_THIS_IS_PAPER = True

BASE_URL = "https://paper-api.alpaca.markets"   # PAPER environment only — do not change

API_KEY_ID = os.environ.get("ALPACA_API_KEY_ID")
API_SECRET_KEY = os.environ.get("ALPACA_API_SECRET_KEY")

if not API_KEY_ID or not API_SECRET_KEY:
    log.error("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set. Add them as GitHub Actions secrets.")
    sys.exit(1)

HEADERS = {
    "APCA-API-KEY-ID": API_KEY_ID,
    "APCA-API-SECRET-KEY": API_SECRET_KEY,
}

# Alpaca tickers are plain symbols, no exchange suffix — and the same symbol
# generally works for both Alpaca and Yahoo Finance, so no separate mapping
# is needed here (unlike the Trading 212 version).
WATCHLIST = ["AAPL", "MSFT", "TSLA"]

RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 30
PROFIT_TARGET_PCT = 0.03

# How much to spend per buy, in USD. Alpaca converts this into a fractional
# share quantity for you via the "notional" order field.
ORDER_VALUE_USD = 20

# ---------------------------------------------------------------------------
# SAFETY CHECK — run before anything else, every single time.
# ---------------------------------------------------------------------------
def verify_paper_environment():
    assert BASE_URL == "https://paper-api.alpaca.markets", (
        "SAFETY STOP: BASE_URL is not the paper trading endpoint. Refusing to continue."
    )
    if not I_CONFIRM_THIS_IS_PAPER:
        log.error("SAFETY STOP: I_CONFIRM_THIS_IS_PAPER is not True. Refusing to continue.")
        sys.exit(1)

    log.info(f"Connecting to: {BASE_URL}")
    r = requests.get(f"{BASE_URL}/v2/account", headers=HEADERS, timeout=10)
    if r.status_code in (401, 403):
        log.error(
            "Authentication failed. This usually means the keys don't match this "
            "environment — e.g. LIVE keys were used against the PAPER endpoint. "
            "Alpaca keys are tied to one environment. Refusing to continue."
        )
        sys.exit(1)
    r.raise_for_status()
    account = r.json()
    log.info(
        f"Authenticated OK — account id {account.get('id')}, "
        f"currency {account.get('currency')}, status {account.get('status')}. "
        f"Confirm this matches your PAPER account in the Alpaca dashboard "
        f"before trusting this run."
    )

# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------
def calculate_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def get_latest_rsi(symbol: str):
    data = yf.download(symbol, period="3mo", interval="1d", progress=False)
    if data.empty or len(data) < RSI_PERIOD + 1:
        log.warning(f"Not enough price data for {symbol}, skipping.")
        return None
    return calculate_rsi(data["Close"])

# ---------------------------------------------------------------------------
# ALPACA API HELPERS
# ---------------------------------------------------------------------------
def get_open_positions():
    r = requests.get(f"{BASE_URL}/v2/positions", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def already_holding(symbol, positions):
    return any(p.get("symbol") == symbol for p in positions)


def place_market_buy_notional(symbol, usd_amount):
    payload = {
        "symbol": symbol,
        "notional": str(usd_amount),   # dollar amount, not share count
        "side": "buy",
        "type": "market",
        "time_in_force": "day",        # required for notional/fractional orders
    }
    r = requests.post(f"{BASE_URL}/v2/orders", headers=HEADERS, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def place_limit_sell(symbol, qty, limit_price):
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "limit",
        "time_in_force": "gtc",
        "limit_price": str(round(limit_price, 2)),
    }
    r = requests.post(f"{BASE_URL}/v2/orders", headers=HEADERS, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def get_order(order_id):
    r = requests.get(f"{BASE_URL}/v2/orders/{order_id}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def wait_for_fill(order_id, timeout_seconds=60, poll_every=3):
    elapsed = 0
    while elapsed < timeout_seconds:
        order = get_order(order_id)
        status = order.get("status")
        if status == "filled":
            return order
        if status in ("canceled", "rejected", "expired"):
            log.warning(f"Order {order_id} ended with status {status}, no fill.")
            return None
        time.sleep(poll_every)
        elapsed += poll_every
    log.warning(f"Order {order_id} did not fill within {timeout_seconds}s.")
    return None

# ---------------------------------------------------------------------------
# MAIN SCAN — one pass through the watchlist, then exit.
# ---------------------------------------------------------------------------
def run_scan():
    verify_paper_environment()

    log.info("Fetching current positions...")
    positions = get_open_positions()

    for symbol in WATCHLIST:
        try:
            log.info(f"Checking {symbol}...")

            if already_holding(symbol, positions):
                log.info("  Already holding a position — skipping.")
                continue

            rsi = get_latest_rsi(symbol)
            if rsi is None:
                continue
            log.info(f"  RSI: {rsi:.2f}")

            if rsi >= RSI_BUY_THRESHOLD:
                log.info("  Not oversold — no action.")
                continue

            log.info(f"  RSI below {RSI_BUY_THRESHOLD} — buying ${ORDER_VALUE_USD} worth.")
            buy_order = place_market_buy_notional(symbol, ORDER_VALUE_USD)
            order_id = buy_order.get("id")
            if order_id is None:
                log.error(f"  Could not read order id: {buy_order}")
                continue

            filled = wait_for_fill(order_id)
            if not filled:
                continue

            fill_price = filled.get("filled_avg_price")
            filled_qty = filled.get("filled_qty")
            if not fill_price or not filled_qty:
                log.warning("  Filled, but couldn't read fill price/qty — check the dashboard.")
                continue

            fill_price = float(fill_price)
            target_price = fill_price * (1 + PROFIT_TARGET_PCT)
            log.info(f"  Filled {filled_qty} @ {fill_price}. Placing limit sell at {target_price:.2f}.")
            place_limit_sell(symbol, filled_qty, target_price)
            log.info("  Profit-target order placed.")

        except Exception as e:
            # One bad ticker should never take down the whole scan.
            log.exception(f"  Error processing {symbol}: {e}")
            continue


if __name__ == "__main__":
    run_scan()
    log.info("Scan complete.")
