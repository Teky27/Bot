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
1. Fetches the current S&P 500 ticker list (from Wikipedia) instead of a
   fixed hand-typed watchlist — so it scans ~500 stocks, not just a
   handful you picked.
2. Downloads recent daily price history for all of them in one batched
   request from Yahoo Finance, and calculates 14-period RSI for each.
3. For any stock at RSI < 30 that you don't already hold, places a market
   buy for a fixed dollar amount (ORDER_VALUE_USD) using Alpaca's
   fractional "notional" order type — up to MAX_NEW_BUYS_PER_RUN per run,
   so a broad market selloff can't trigger dozens of buys in one go.
4. On every run, checks your existing open positions: sells any that have
   gained 3% or more (take profit), and sells any that have dropped 5% or
   more (stop loss). Both use market orders, not limit orders placed right
   after buying — Alpaca rejects limit orders on fractional quantities, so
   this checks-and-exits on each run instead. See manage_open_positions.

WHAT IT DOESN'T DO
------------------
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
# generally works for both Alpaca and Yahoo Finance.
# The watchlist itself is now fetched dynamically at runtime (see
# get_sp500_tickers below) instead of being hand-typed here.

RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 30
PROFIT_TARGET_PCT = 0.03    # sell if a position is up 3% or more
STOP_LOSS_PCT = 0.05        # sell if a position is down 5% or more

# How much to spend per buy, in USD. Alpaca converts this into a fractional
# share quantity for you via the "notional" order field.
ORDER_VALUE_USD = 20

# Safety cap: with ~500 tickers being scanned, a broad market selloff could
# push many of them below RSI 30 at once. This caps how many NEW positions
# a single run is allowed to open, so one bad market day can't turn into
# dozens of simultaneous buys. Extra candidates are simply skipped that run
# and picked up again next run if still oversold.
MAX_NEW_BUYS_PER_RUN = 5

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


def market_is_open() -> bool:
    """Checks Alpaca's market clock. Catches weekends AND holidays, unlike
    a plain weekday cron schedule."""
    r = requests.get(f"{BASE_URL}/v2/clock", headers=HEADERS, timeout=10)
    r.raise_for_status()
    clock = r.json()
    return bool(clock.get("is_open"))

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


def get_sp500_tickers() -> list[str]:
    """Fetches the current S&P 500 constituent list from Wikipedia."""
    import io
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia returns 403 Forbidden to requests without a browser-like
    # User-Agent header — pandas.read_html doesn't set one by default, so
    # we fetch the page ourselves first and hand pandas the HTML directly.
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    tickers = tables[0]["Symbol"].tolist()
    # A handful of tickers use a dot for share class (e.g. BRK.B, BF.B).
    # Yahoo Finance and Alpaca don't always agree on how to format these,
    # so we skip the ~5 affected names rather than risk a mismatched trade.
    tickers = [t for t in tickers if "." not in t]
    return tickers


def get_price_history_bulk(tickers: list[str]) -> pd.DataFrame:
    """One batched download for all tickers, instead of one request per
    ticker — much friendlier to Yahoo Finance at this scale."""
    return yf.download(
        tickers, period="3mo", interval="1d",
        group_by="ticker", threads=True, progress=False,
    )

# ---------------------------------------------------------------------------
# ALPACA API HELPERS
# ---------------------------------------------------------------------------
def get_open_positions():
    r = requests.get(f"{BASE_URL}/v2/positions", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


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


def place_market_sell(symbol, qty):
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
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


def manage_open_positions(positions):
    """Checks every currently open position on each run and exits it with a
    MARKET order (fractional-share safe, unlike limit orders) if either:
      - unrealized gain has reached PROFIT_TARGET_PCT (take profit), or
      - unrealized loss has reached STOP_LOSS_PCT (stop loss).

    Because this only checks once per run (hourly), an actual exit price
    can land a bit past the exact target/stop level depending on how much
    the price moved between runs — an unavoidable trade-off of periodic
    checking vs. watching continuously.
    """
    if not positions:
        log.info("No open positions to check.")
        return

    sold_count = 0
    for p in positions:
        try:
            symbol = p.get("symbol")
            qty = p.get("qty")
            unrealized_plpc = float(p.get("unrealized_plpc", 0))

            if unrealized_plpc >= PROFIT_TARGET_PCT:
                log.info(
                    f"{symbol}: up {unrealized_plpc * 100:.2f}% — hit "
                    f"{PROFIT_TARGET_PCT * 100:.0f}% profit target, selling {qty} shares."
                )
                place_market_sell(symbol, qty)
                sold_count += 1
            elif unrealized_plpc <= -STOP_LOSS_PCT:
                log.info(
                    f"{symbol}: down {unrealized_plpc * 100:.2f}% — hit "
                    f"-{STOP_LOSS_PCT * 100:.0f}% stop loss, selling {qty} shares."
                )
                place_market_sell(symbol, qty)
                sold_count += 1
        except Exception as e:
            log.exception(f"Error checking/selling position {p.get('symbol')}: {e}")
            continue

    log.info(
        f"Checked {len(positions)} open position(s): {sold_count} exited "
        f"(profit target or stop loss), {len(positions) - sold_count} still open."
    )

# ---------------------------------------------------------------------------
# MAIN SCAN — one pass through the watchlist, then exit.
# ---------------------------------------------------------------------------
def run_scan():
    verify_paper_environment()

    if not market_is_open():
        log.info("Market is currently closed (weekend or holiday) — skipping this run.")
        return

    log.info("Fetching current positions...")
    positions = get_open_positions()
    held_symbols = {p.get("symbol") for p in positions}

    log.info("Checking open positions for profit-target / stop-loss hits...")
    manage_open_positions(positions)

    log.info("Fetching S&P 500 ticker list...")
    try:
        watchlist = get_sp500_tickers()
    except Exception as e:
        log.exception(f"Could not fetch S&P 500 list — aborting this run: {e}")
        return
    log.info(f"Scanning {len(watchlist)} tickers.")

    log.info("Downloading price history in bulk (this can take a minute or two)...")
    try:
        history = get_price_history_bulk(watchlist)
    except Exception as e:
        log.exception(f"Bulk price download failed — aborting this run: {e}")
        return

    oversold_count = 0
    buys_this_run = 0

    for symbol in watchlist:
        if buys_this_run >= MAX_NEW_BUYS_PER_RUN:
            log.info(f"Reached MAX_NEW_BUYS_PER_RUN ({MAX_NEW_BUYS_PER_RUN}) — stopping scan for this run.")
            break

        try:
            if symbol in held_symbols:
                continue

            try:
                closes = history[symbol]["Close"].dropna()
            except (KeyError, ValueError):
                continue  # Yahoo returned no data for this symbol this run

            if len(closes) < RSI_PERIOD + 1:
                continue

            rsi = calculate_rsi(closes)
            if rsi >= RSI_BUY_THRESHOLD:
                continue

            oversold_count += 1
            log.info(f"{symbol}: RSI {rsi:.2f} — oversold, buying ${ORDER_VALUE_USD} worth.")

            buy_order = place_market_buy_notional(symbol, ORDER_VALUE_USD)
            order_id = buy_order.get("id")
            if order_id is None:
                log.error(f"  Could not read order id: {buy_order}")
                continue

            filled = wait_for_fill(order_id)
            if filled:
                log.info(
                    f"  Filled {filled.get('filled_qty')} @ {filled.get('filled_avg_price')}. "
                    f"Profit target will be checked on future runs."
                )
            buys_this_run += 1

        except Exception as e:
            # One bad ticker should never take down the whole scan.
            log.exception(f"Error processing {symbol}: {e}")
            continue

    log.info(f"Scan finished. {oversold_count} oversold this run, {buys_this_run} new position(s) opened.")


if __name__ == "__main__":
    run_scan()
    log.info("Scan complete.")
