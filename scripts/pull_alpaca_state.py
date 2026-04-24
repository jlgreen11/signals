"""Pull paper-trading state from all three Alpaca accounts and dump JSON.

Used by LIVE_RECORD.md to capture forward-evidence snapshots. Each run
writes a timestamped JSON under scripts/data/paper_trading_<DATE>.json.

Usage:
    python scripts/pull_alpaca_state.py                      # stdout
    python scripts/pull_alpaca_state.py --output path.json   # to file

Requires .env with:
    ALPACA_API_KEY / ALPACA_SECRET_KEY               (momentum account)
    ALPACA_MULTIFACTOR_KEY / ALPACA_MULTIFACTOR_SECRET
    ALPACA_BASELINE_KEY / ALPACA_BASELINE_SECRET
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest  # noqa: E402

ACCOUNTS: dict[str, tuple[str, str]] = {
    "momentum":    ("ALPACA_API_KEY",         "ALPACA_SECRET_KEY"),
    "multifactor": ("ALPACA_MULTIFACTOR_KEY", "ALPACA_MULTIFACTOR_SECRET"),
    "baseline":    ("ALPACA_BASELINE_KEY",    "ALPACA_BASELINE_SECRET"),
}

BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def fetch_account(name: str, key_var: str, secret_var: str) -> dict:
    key = os.environ.get(key_var)
    secret = os.environ.get(secret_var)
    if not key or not secret:
        return {"name": name, "error": f"missing env: {key_var}/{secret_var}"}

    out: dict = {"name": name}
    try:
        tc = TradingClient(key, secret, paper=True)

        acct = tc.get_account()
        out["account"] = {
            "equity":             float(acct.equity),
            "last_equity":        float(acct.last_equity),
            "cash":               float(acct.cash),
            "buying_power":       float(acct.buying_power),
            "long_market_value":  float(acct.long_market_value),
            "portfolio_value":    float(acct.portfolio_value),
            "status":             str(acct.status),
            "created_at":         acct.created_at.isoformat() if acct.created_at else None,
            "currency":           acct.currency,
            "pattern_day_trader": bool(acct.pattern_day_trader),
            "account_number":     str(acct.account_number)[-4:],
        }

        positions = tc.get_all_positions()
        out["positions"] = [
            {
                "symbol":           p.symbol,
                "qty":              float(p.qty),
                "avg_entry_price":  float(p.avg_entry_price),
                "current_price":    float(p.current_price) if p.current_price else None,
                "market_value":     float(p.market_value),
                "cost_basis":       float(p.cost_basis),
                "unrealized_pl":    float(p.unrealized_pl),
                "unrealized_plpc":  float(p.unrealized_plpc),
                "side":             str(p.side),
            }
            for p in positions
        ]

        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500)
        orders = tc.get_orders(filter=req)
        out["orders"] = [
            {
                "symbol":           o.symbol,
                "side":             str(o.side),
                "qty":              float(o.qty) if o.qty else None,
                "filled_qty":       float(o.filled_qty) if o.filled_qty else None,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "submitted_at":     o.submitted_at.isoformat() if o.submitted_at else None,
                "filled_at":        o.filled_at.isoformat() if o.filled_at else None,
                "status":           str(o.status),
                "order_type":       str(o.order_type),
                "time_in_force":    str(o.time_in_force),
            }
            for o in orders
        ]

        # Portfolio history — alpaca-py 0.43 doesn't expose this cleanly, hit REST.
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        for period, tf in [("all", "1D"), ("1M", "1D"), ("7D", "1H")]:
            try:
                r = requests.get(
                    f"{BASE_URL}/v2/account/portfolio/history",
                    headers=headers,
                    params={"period": period, "timeframe": tf, "extended_hours": "true"},
                    timeout=15,
                )
                if r.status_code == 200:
                    out[f"portfolio_history_{period}_{tf}"] = r.json()
                    if period == "all":
                        break
                else:
                    out[f"portfolio_history_{period}_{tf}_error"] = (
                        f"{r.status_code}: {r.text[:200]}"
                    )
            except Exception as e:
                out[f"portfolio_history_{period}_{tf}_error"] = str(e)

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", "-o", help="Output path; omit for stdout")
    args = ap.parse_args()

    result = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "base_url":  BASE_URL,
        "accounts":  {name: fetch_account(name, k, s) for name, (k, s) in ACCOUNTS.items()},
    }
    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(payload + "\n")
        errs = [a.get("error") for a in result["accounts"].values() if a.get("error")]
        print(
            f"Wrote {args.output} ({sum(1 for a in result['accounts'].values() if 'error' not in a)}/"
            f"{len(result['accounts'])} accounts ok)",
            file=sys.stderr,
        )
        if errs:
            for e in errs:
                print(f"  error: {e}", file=sys.stderr)
            return 1
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
