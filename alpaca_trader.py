from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass

# Auto-load .env so keys work without manually exporting them
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    for _candidate in [_here, os.path.dirname(_here)]:
        _env_path = os.path.join(_candidate, ".env")
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            break
except ImportError:
    pass

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.models import Position as AlpacaPosition
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


@dataclass
class AlpacaConfig:
    api_key:    str
    secret_key: str
    paper:      bool = True  # always default to paper trading for safety

    @classmethod
    def from_env(cls) -> "AlpacaConfig | None":
        api_key    = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            missing = []
            if not api_key:    missing.append("ALPACA_API_KEY")
            if not secret_key: missing.append("ALPACA_SECRET_KEY")
            print(f"{YELLOW}[ALPACA] Missing env vars: {', '.join(missing)}{RESET}")
            print(f"{YELLOW}[ALPACA] Add them to your .env file. Trading disabled.{RESET}")
            return None

        # ALPACA_PAPER=false to switch to live — requires explicit opt-in
        paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        return cls(api_key=api_key, secret_key=secret_key, paper=paper)


class AlpacaTrader:
    # Wraps the Alpaca API for order submission and account queries.
    # All position sizing logic lives here so monitor.py stays clean.
    #
    # Risk management defaults:
    #   risk_pct         — fraction of account equity risked per trade (1%)
    #   max_position_pct — max fraction of equity in any single position (20%)
    #
    # Position size = floor(equity * risk_pct / (price - stop_price))
    # Capped at floor(equity * max_position_pct / price) to prevent
    # a very tight stop from generating an oversized position.

    def __init__(
        self,
        config:          AlpacaConfig,
        risk_pct:        float = 0.01,
        max_position_pct: float = 0.20,
    ):
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )

        self.cfg             = config
        self.risk_pct        = risk_pct
        self.max_position_pct = max_position_pct
        self.paper           = config.paper

        self.client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )

        mode = f"{YELLOW}PAPER{RESET}" if config.paper else f"{RED}{BOLD}LIVE{RESET}"
        print(f"  {GREEN}[ALPACA] Connected — {mode} trading{RESET}")
        acct = self.client.get_account()
        print(f"  [ALPACA] Account equity : ${float(acct.equity):,.2f}")
        print(f"  [ALPACA] Buying power   : ${float(acct.buying_power):,.2f}")

    def _equity(self) -> float:
        return float(self.client.get_account().equity)

    def _buying_power(self) -> float:
        return float(self.client.get_account().buying_power)

    def _calc_shares(self, price: float, stop_price: float) -> int:
        equity     = self._equity()
        stop_dist  = price - stop_price

        if stop_dist <= 0:
            print(f"  {YELLOW}[ALPACA] Invalid stop distance ({stop_dist:.4f}) — order skipped.{RESET}")
            return 0

        # Risk-based sizing: risk exactly risk_pct of equity on this trade
        risk_shares = math.floor((equity * self.risk_pct) / stop_dist)

        # Size cap: never put more than max_position_pct of equity into one stock
        cap_shares  = math.floor((equity * self.max_position_pct) / price)

        shares = min(risk_shares, cap_shares)

        # Make sure we can actually afford the position
        max_affordable = math.floor(self._buying_power() / price)
        shares = min(shares, max_affordable)

        return max(shares, 0)

    def get_position(self, ticker: str) -> "AlpacaPosition | None":
        try:
            return self.client.get_open_position(ticker)
        except Exception:
            return None

    def has_position(self, ticker: str) -> bool:
        return self.get_position(ticker) is not None

    def submit_buy(
        self,
        ticker:     str,
        price:      float,
        stop_price: float,
    ) -> bool:
        # Don't double-enter if we already hold this ticker
        if self.has_position(ticker):
            print(f"  {YELLOW}[ALPACA] Already in {ticker} — skipping buy.{RESET}")
            return False

        # Cancel any lingering open orders (e.g. orphaned stop loss from a
        # previous trade) before placing a new buy — Alpaca rejects the order
        # as a wash trade if an opposite-side order already exists.
        try:
            self.client.cancel_orders_for_symbol(ticker)
        except Exception:
            pass

        shares = self._calc_shares(price, stop_price)
        if shares == 0:
            print(f"  {YELLOW}[ALPACA] Position size computed to 0 shares — skipping.{RESET}")
            return False

        try:
            # Step 1: market buy
            buy_req = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(buy_req)
            print(f"  {GREEN}[ALPACA] BUY submitted — {shares} shares of {ticker} @ ~${price:.2f}{RESET}")
            print(f"  [ALPACA] Order ID : {order.id}")

            # Step 2: stop loss order (separate so it doesn't block the fill)
            stop_req = StopOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,  # GTC so it persists overnight
                stop_price=round(stop_price, 2),
            )
            stop_order = self.client.submit_order(stop_req)
            print(f"  [ALPACA] Stop loss set at ${stop_price:.2f} (order {stop_order.id})")

            dollar_risk = shares * (price - stop_price)
            print(f"  [ALPACA] Risk on trade : ${dollar_risk:.2f}  ({self.risk_pct*100:.1f}% of equity)")
            return True

        except Exception as e:
            print(f"  {RED}[ALPACA] Buy order failed: {e}{RESET}")
            return False

    def submit_sell(self, ticker: str, reason: str = "signal") -> bool:
        if not self.has_position(ticker):
            print(f"  {YELLOW}[ALPACA] No position in {ticker} — nothing to sell.{RESET}")
            return False

        try:
            # Cancel any open orders (e.g. the stop loss) before closing
            self.client.cancel_orders_for_symbol(ticker)

            # Close the full position at market
            self.client.close_position(ticker)

            pos = self.get_position(ticker)
            if pos:
                pnl    = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                sign   = "+" if pnl >= 0 else ""
                colour = GREEN if pnl >= 0 else RED
                print(
                    f"  {colour}[ALPACA] SELL submitted — {ticker}  "
                    f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)  "
                    f"Reason: {reason}{RESET}"
                )
            else:
                print(f"  {RED}[ALPACA] SELL submitted — {ticker}  (Reason: {reason}){RESET}")

            return True

        except Exception as e:
            print(f"  {RED}[ALPACA] Sell order failed: {e}{RESET}")
            return False

    def account_summary(self) -> dict:
        acct = self.client.get_account()
        return {
            "equity":       float(acct.equity),
            "cash":         float(acct.cash),
            "buying_power": float(acct.buying_power),
            "pnl_today":    float(acct.equity) - float(acct.last_equity),
        }

    def print_account_summary(self):
        s    = self.account_summary()
        sign = "+" if s["pnl_today"] >= 0 else ""
        col  = GREEN if s["pnl_today"] >= 0 else RED
        print(f"\n  [ALPACA] Account Summary")
        print(f"    Equity      : ${s['equity']:,.2f}")
        print(f"    Cash        : ${s['cash']:,.2f}")
        print(f"    Buying Power: ${s['buying_power']:,.2f}")
        print(f"    Today P&L   : {col}{sign}${s['pnl_today']:.2f}{RESET}")
