# =============================================================================
# PROGRAM 12: IBKR Live Execution Bridge (Capstone)
# =============================================================================
# Description:
#   Connects to Interactive Brokers TWS or IB Gateway via ib_insync.
#   Provides a full signal-to-order pipeline: connect → validate signal →
#   run risk checks → create order → submit → monitor.
#   Includes a SIMULATION MODE that runs the entire flow without IBKR,
#   so this program works in Colab for learning/demo purposes.
#
# IBKR Setup (required for live/paper trading):
#   1. Download IBKR Trader Workstation (TWS) or IB Gateway:
#      https://www.interactivebrokers.com/en/trading/tws.php
#   2. Enable API: TWS → Edit → Global Configuration → API → Settings
#      - Enable ActiveX and Socket Clients: YES
#      - Socket port: 7497 (paper), 7496 (live)
#      - Trusted IP: 127.0.0.1
#   3. Install ib_insync: pip install ib_insync
#   4. Set SIMULATION_MODE = False in config below
#
# What it produces:
#   - Account summary printout
#   - Live positions with real-time P&L
#   - Options chain snapshot for a target ticker
#   - Signal ingestion and risk-checked order submission
#   - Order status monitoring loop
#   - Dashboard: positions, P&L bar, order log, risk meter
#
# Platform: Google Colab  |  Runtime: depends on TWS connection
#
# Install (run first in Colab):
#   !pip install ib_insync yfinance numpy matplotlib pandas
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import time
import random
import threading
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIGURATION
# =============================================================================

SIMULATION_MODE = True      # ← Set False when TWS/Gateway is running
IBKR_HOST       = "127.0.0.1"
IBKR_PORT       = 7497      # 7497 = paper trading; 7496 = live
IBKR_CLIENT_ID  = 1         # unique client ID (1-32)

# Risk limits — these are hard stops checked before every order
RISK_LIMITS = {
    "max_position_value":   50_000,    # max $ value per single position
    "max_portfolio_delta":  200,       # max absolute net delta (shares equivalent)
    "max_daily_loss":       2_000,     # stop trading if daily P&L < -$2,000
    "max_vega_exposure":    5_000,     # max portfolio vega $ per 1% vol move
    "max_open_orders":      10,        # refuse new orders if too many open
    "min_option_volume":    50,        # minimum daily option volume for order
    "max_contracts_single": 10,        # max contracts in a single order
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Signal:
    """
    A trading signal that drives the execution pipeline.
    This would typically come from Program 11 (screener) or your own model.
    """
    ticker:       str
    signal_type:  str            # 'sell_vol', 'buy_vol', 'delta_hedge'
    strategy:     str            # 'short_put', 'iron_condor', 'long_call', etc.
    direction:    str            # 'BUY' or 'SELL'
    option_type:  str            # 'call', 'put', 'combo'
    strike:       float
    expiry:       str            # 'YYYY-MM-DD'
    quantity:     int            # contracts
    limit_price:  Optional[float] = None  # None = MID price
    notes:        str            = ""
    timestamp:    str            = field(default_factory=lambda: str(datetime.now()))


@dataclass
class OrderResult:
    """Tracks the result of a submitted order."""
    order_id:     int
    ticker:       str
    status:       str            # 'PENDING', 'FILLED', 'CANCELLED', 'ERROR'
    filled_price: Optional[float] = None
    filled_qty:   int            = 0
    timestamp:    str            = field(default_factory=lambda: str(datetime.now()))
    message:      str            = ""


@dataclass
class Position:
    """Represents a single open position."""
    ticker:        str
    description:   str
    quantity:      int
    avg_cost:      float
    market_value:  float
    unrealized_pnl: float
    realized_pnl:  float
    delta:         float
    vega:          float


@dataclass
class AccountSummary:
    """Key account metrics from IBKR."""
    net_liquidation:    float
    available_funds:    float
    buying_power:       float
    gross_position_val: float
    daily_pnl:          float
    total_pnl:          float

# =============================================================================
# SIMULATION ENGINE (runs when IBKR not connected)
# =============================================================================

class SimulationBroker:
    """
    Simulates IBKR API responses for testing in Colab.
    Generates realistic account data, positions, and order fills.
    """

    def __init__(self):
        self.connected  = False
        self.orders_log: List[OrderResult] = []
        self._order_counter = 1000
        self._sim_positions = self._generate_positions()
        self._daily_pnl = random.uniform(-500, 1200)
        print("  [SIM] Simulation broker initialized.")

    def _generate_positions(self) -> List[Position]:
        """Create realistic simulated options positions."""
        return [
            Position("SPY",   "SPY Jun20'26 560 Call",   1,  8.50, 11.20,  270.0,  0.0, 0.65, 0.18),
            Position("SPY",   "SPY Jun20'26 530 Put",   -1,  6.20,  4.80,  140.0,  0.0, 0.22, 0.15),
            Position("QQQ",   "QQQ May16'26 480 Call",   2,  5.10,  7.30,  440.0,  0.0, 0.55, 0.28),
            Position("AAPL",  "AAPL Apr17'26 210 Put",  -1,  3.80,  2.90,   90.0,  0.0, 0.28, 0.12),
            Position("NVDA",  "NVDA Apr17'26 900 Call",  1, 15.40, 21.80,  640.0,  0.0, 0.60, 0.32),
        ]

    def connect(self):
        self.connected = True
        print(f"  [SIM] Connected to simulation broker (paper port {IBKR_PORT})")
        return True

    def disconnect(self):
        self.connected = False
        print("  [SIM] Disconnected from simulation broker.")

    def get_account_summary(self) -> AccountSummary:
        total_mkt = sum(abs(p.market_value) * 100 for p in self._sim_positions)
        return AccountSummary(
            net_liquidation    = 85_420.50,
            available_funds    = 42_800.00,
            buying_power       = 128_400.00,
            gross_position_val = total_mkt,
            daily_pnl          = self._daily_pnl,
            total_pnl          = 6_830.40,
        )

    def get_positions(self) -> List[Position]:
        # Add small random drift to simulate live prices
        for p in self._sim_positions:
            drift = p.market_value * random.uniform(-0.008, 0.012)
            p.market_value    += drift
            p.unrealized_pnl   = (p.market_value - p.avg_cost) * abs(p.quantity) * 100
        return self._sim_positions

    def get_options_chain(self, ticker: str, expiry: str):
        """Return a simulated options chain for the nearest expiry."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            exps = t.options
            if exps:
                today = date.today()
                exp = min(exps, key=lambda e: abs(
                    (datetime.strptime(e, '%Y-%m-%d').date() - today).days - 30))
                chain = t.option_chain(exp)
                return chain.calls, chain.puts, exp
        except Exception:
            pass

        # Pure simulation fallback
        spot = 540.0
        strikes = np.arange(spot * 0.90, spot * 1.10, 5)
        calls = pd.DataFrame({
            'strike':            strikes,
            'lastPrice':         np.maximum(spot - strikes + 10, 0.5),
            'bid':               np.maximum(spot - strikes + 9.5, 0.4),
            'ask':               np.maximum(spot - strikes + 10.5, 0.6),
            'volume':            np.random.randint(100, 5000, len(strikes)),
            'openInterest':      np.random.randint(500, 20000, len(strikes)),
            'impliedVolatility': np.random.uniform(0.18, 0.35, len(strikes)),
        })
        puts = calls.copy()
        puts['lastPrice'] = np.maximum(strikes - spot + 10, 0.5)
        return calls, puts, "2026-05-16"

    def submit_order(self, signal: Signal) -> OrderResult:
        """Simulate order submission with realistic fills."""
        self._order_counter += 1
        oid = self._order_counter

        # Simulate fill delay (in reality this would be async)
        time.sleep(0.3)

        # 95% fill rate, 5% rejects for simulation
        if random.random() < 0.95:
            mid_price = signal.limit_price or random.uniform(2.0, 15.0)
            slippage  = random.uniform(-0.03, 0.05)
            filled_px = round(mid_price * (1 + slippage), 2)
            result = OrderResult(
                order_id    = oid,
                ticker      = signal.ticker,
                status      = "FILLED",
                filled_price = filled_px,
                filled_qty  = signal.quantity,
                message     = f"Sim fill at ${filled_px:.2f}"
            )
        else:
            result = OrderResult(
                order_id = oid,
                ticker   = signal.ticker,
                status   = "CANCELLED",
                message  = "Sim: rejected — no fill available"
            )

        self.orders_log.append(result)
        return result

    def cancel_order(self, order_id: int) -> bool:
        for o in self.orders_log:
            if o.order_id == order_id and o.status == "PENDING":
                o.status = "CANCELLED"
                return True
        return False

    def get_net_greeks(self, positions: List[Position]) -> Dict[str, float]:
        """Aggregate portfolio greeks (simulated)."""
        net_delta = sum(p.delta * p.quantity * 100 for p in positions)
        net_vega  = sum(p.vega * abs(p.quantity) * 100 for p in positions)
        return {"net_delta": net_delta, "net_vega": net_vega}


# =============================================================================
# IBKR LIVE BROKER (requires ib_insync + running TWS)
# =============================================================================

class IBKRBroker:
    """
    Production IBKR broker using ib_insync.
    Only instantiated when SIMULATION_MODE = False.
    """

    def __init__(self):
        self.ib = None
        self.connected = False
        self.orders_log: List[OrderResult] = []

    def connect(self):
        try:
            from ib_insync import IB, util
            self.ib = IB()
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
            self.connected = self.ib.isConnected()
            if self.connected:
                print(f"  [IBKR] Connected to TWS at {IBKR_HOST}:{IBKR_PORT}")
            else:
                print("  [IBKR] Connection failed — check TWS is running.")
            return self.connected
        except ImportError:
            print("  [IBKR] ib_insync not installed. Run: !pip install ib_insync")
            return False
        except Exception as e:
            print(f"  [IBKR] Connection error: {e}")
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("  [IBKR] Disconnected.")
        self.connected = False

    def get_account_summary(self) -> AccountSummary:
        from ib_insync import IB
        summary = {v.tag: float(v.value) for v in self.ib.accountSummary()
                   if v.currency == 'USD'}
        return AccountSummary(
            net_liquidation    = summary.get('NetLiquidation', 0),
            available_funds    = summary.get('AvailableFunds', 0),
            buying_power       = summary.get('BuyingPower', 0),
            gross_position_val = summary.get('GrossPositionValue', 0),
            daily_pnl          = summary.get('DailyPnL', 0),
            total_pnl          = summary.get('UnrealizedPnL', 0),
        )

    def get_positions(self) -> List[Position]:
        from ib_insync import IB
        positions = []
        for p in self.ib.positions():
            contract = p.contract
            pos_obj = Position(
                ticker        = contract.symbol,
                description   = str(contract),
                quantity      = int(p.position),
                avg_cost      = float(p.avgCost),
                market_value  = float(p.position * p.avgCost),
                unrealized_pnl = 0.0,
                realized_pnl   = 0.0,
                delta          = 0.0,
                vega           = 0.0,
            )
            positions.append(pos_obj)
        return positions

    def get_options_chain(self, ticker: str, expiry: str):
        """Pull live options chain via IBKR market data."""
        # NOTE: Requires market data subscription for full chain
        # Simplified version using yfinance as fallback
        import yfinance as yf
        t = yf.Ticker(ticker)
        exps = t.options
        today = date.today()
        best_exp = min(exps, key=lambda e: abs(
            (datetime.strptime(e, '%Y-%m-%d').date() - today).days - 30))
        chain = t.option_chain(best_exp)
        return chain.calls, chain.puts, best_exp

    def submit_order(self, signal: Signal) -> OrderResult:
        from ib_insync import IB, Option, LimitOrder, MarketOrder, contract
        try:
            expiry_fmt = signal.expiry.replace('-', '')[:6]  # YYYYMM
            opt = Option(
                symbol      = signal.ticker,
                lastTradeDateOrContractMonth = expiry_fmt,
                strike      = signal.strike,
                right       = 'C' if signal.option_type == 'call' else 'P',
                exchange    = 'SMART',
                currency    = 'USD',
                multiplier  = '100',
            )
            self.ib.qualifyContracts(opt)

            if signal.limit_price:
                order = LimitOrder(signal.direction, signal.quantity, signal.limit_price)
            else:
                order = MarketOrder(signal.direction, signal.quantity)

            trade = self.ib.placeOrder(opt, order)
            self.ib.sleep(1)

            result = OrderResult(
                order_id     = trade.order.orderId,
                ticker       = signal.ticker,
                status       = trade.orderStatus.status,
                filled_price = trade.orderStatus.avgFillPrice or None,
                filled_qty   = int(trade.orderStatus.filled),
                message      = f"IBKR order {trade.order.orderId}"
            )
            self.orders_log.append(result)
            return result

        except Exception as e:
            result = OrderResult(
                order_id = -1, ticker = signal.ticker, status = "ERROR",
                message  = str(e)
            )
            self.orders_log.append(result)
            return result

    def get_net_greeks(self, positions: List[Position]) -> Dict[str, float]:
        # In live mode, you'd pull this from account portfolio data
        return {"net_delta": 0.0, "net_vega": 0.0}

# =============================================================================
# RISK ENGINE
# =============================================================================

class RiskEngine:
    """
    Pre-trade risk check framework.
    Every signal must pass ALL checks before an order is submitted.
    """

    def __init__(self, limits: dict = RISK_LIMITS):
        self.limits  = limits
        self.blocked = False   # set True to halt all new orders

    def check_signal(self, signal: Signal, account: AccountSummary,
                     positions: List[Position], net_greeks: Dict,
                     open_order_count: int) -> (bool, List[str]):
        """
        Run all risk checks on a signal.
        Returns (approved: bool, list of failure reasons).
        """
        failures = []

        # 1. Daily loss breaker
        if account.daily_pnl < -abs(self.limits['max_daily_loss']):
            failures.append(
                f"Daily loss breaker: P&L ${account.daily_pnl:,.0f} < "
                f"-${self.limits['max_daily_loss']:,}"
            )

        # 2. Max contracts
        if signal.quantity > self.limits['max_contracts_single']:
            failures.append(
                f"Contract limit: {signal.quantity} > max "
                f"{self.limits['max_contracts_single']}"
            )

        # 3. Available funds check (rough: option premium * 100 * qty * 2x buffer)
        if signal.limit_price:
            est_cost = signal.limit_price * 100 * signal.quantity * 1.5
            if est_cost > account.available_funds:
                failures.append(
                    f"Insufficient funds: need ~${est_cost:,.0f}, "
                    f"have ${account.available_funds:,.0f}"
                )

        # 4. Portfolio delta limit
        signal_delta = 0.5 * signal.quantity * 100   # rough delta estimate
        if signal.direction == 'SELL':
            signal_delta = -signal_delta
        projected_delta = abs(net_greeks.get('net_delta', 0) + signal_delta)
        if projected_delta > self.limits['max_portfolio_delta']:
            failures.append(
                f"Delta limit: projected |delta|={projected_delta:.0f} > "
                f"max {self.limits['max_portfolio_delta']}"
            )

        # 5. Open orders limit
        if open_order_count >= self.limits['max_open_orders']:
            failures.append(
                f"Too many open orders: {open_order_count} >= "
                f"{self.limits['max_open_orders']}"
            )

        # 6. Global halt
        if self.blocked:
            failures.append("Risk engine is HALTED — all orders blocked.")

        approved = len(failures) == 0
        return approved, failures

    def emergency_halt(self, reason: str = "Manual halt"):
        """Stop all new orders immediately."""
        self.blocked = True
        print(f"\n  *** RISK ENGINE HALTED: {reason} ***")

    def resume(self):
        self.blocked = False
        print("  Risk engine resumed.")

# =============================================================================
# EXECUTION PIPELINE
# =============================================================================

class ExecutionPipeline:
    """
    Orchestrates the full signal → risk check → order → monitor flow.
    """

    def __init__(self, broker, risk_engine: RiskEngine):
        self.broker      = broker
        self.risk        = risk_engine
        self.order_log   = []

    def process_signal(self, signal: Signal) -> Optional[OrderResult]:
        """
        Full pipeline:
        1. Fetch current account & positions
        2. Run risk checks
        3. If approved: submit order
        4. Return OrderResult
        """
        print(f"\n  Pipeline: Processing signal → "
              f"{signal.direction} {signal.quantity}x "
              f"{signal.ticker} {signal.option_type.upper()} "
              f"K={signal.strike} exp={signal.expiry}")

        # Step 1: Fetch context
        account   = self.broker.get_account_summary()
        positions = self.broker.get_positions()
        greeks    = self.broker.get_net_greeks(positions)
        open_cnt  = len([o for o in self.broker.orders_log
                         if o.status == 'PENDING'])

        # Step 2: Risk check
        approved, failures = self.risk.check_signal(
            signal, account, positions, greeks, open_cnt)

        if not approved:
            print("  RISK CHECK FAILED:")
            for f in failures:
                print(f"    - {f}")
            result = OrderResult(
                order_id = -1, ticker = signal.ticker,
                status   = "BLOCKED",
                message  = " | ".join(failures)
            )
            self.order_log.append(result)
            return result

        print("  Risk check PASSED.")

        # Step 3: Submit order
        print("  Submitting order to broker...")
        result = self.broker.submit_order(signal)
        self.order_log.append(result)

        print(f"  Order result: {result.status} | "
              f"Filled={result.filled_qty}@${result.filled_price or 0:.2f} | "
              f"{result.message}")

        return result

    def monitor_positions(self, duration_seconds: int = 5):
        """Real-time position monitor loop."""
        print(f"\n  Monitoring positions for {duration_seconds}s "
              f"(Ctrl+C to stop)...")
        start = time.time()
        try:
            while time.time() - start < duration_seconds:
                positions = self.broker.get_positions()
                account   = self.broker.get_account_summary()
                total_pnl = sum(p.unrealized_pnl for p in positions)
                print(f"\r  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Net P&L: ${total_pnl:+,.0f}  "
                      f"Daily P&L: ${account.daily_pnl:+,.0f}  "
                      f"Positions: {len(positions)}",
                      end='', flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        print()

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_dashboard(account: AccountSummary, positions: List[Position],
                   order_log: List[OrderResult], net_greeks: Dict):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor='#0d1117')
    mode_label = "SIMULATION MODE" if SIMULATION_MODE else "LIVE MODE"
    fig.suptitle(f"IBKR Execution Bridge — {mode_label}",
                 fontsize=15, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           top=0.94, bottom=0.06, left=0.06, right=0.97)
    PLOT_BG = '#151b27'

    # ── Panel 1: Account Summary ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PLOT_BG)
    ax1.axis('off')
    ax1.set_title("Account Summary", color='white', fontsize=11,
                  fontweight='bold', loc='left', pad=8)

    acct_rows = [
        ("Net Liquidation",   f"${account.net_liquidation:,.2f}"),
        ("Available Funds",   f"${account.available_funds:,.2f}"),
        ("Buying Power",      f"${account.buying_power:,.2f}"),
        ("Gross Position Val",f"${account.gross_position_val:,.2f}"),
        ("Daily P&L",         f"${account.daily_pnl:+,.2f}"),
        ("Total Unr. P&L",    f"${account.total_pnl:+,.2f}"),
        ("Net Delta (shares)",f"{net_greeks.get('net_delta',0):+,.0f}"),
        ("Net Vega ($)",      f"${net_greeks.get('net_vega',0):+,.0f}"),
    ]
    for j, (label, value) in enumerate(acct_rows):
        y = 0.92 - j * 0.115
        try:
            num_str = value.replace('$', '').replace('%', '').replace(',', '')
            num = float(num_str.replace('+', ''))
            color = '#2ecc71' if num > 0 else '#e74c3c' if num < 0 else 'white'
        except Exception:
            color = 'white'
        if j < 4:
            color = '#ccc'
        ax1.text(0.04, y, label, transform=ax1.transAxes, color='#888', fontsize=9)
        ax1.text(0.96, y, value, transform=ax1.transAxes, ha='right',
                 color=color, fontsize=9, fontweight='bold')
        ax1.axhline(y=y - 0.045, xmin=0.03, xmax=0.97, color='#333',
                    linewidth=0.5, transform=ax1.transAxes)

    # ── Panel 2: Positions Table ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor(PLOT_BG)
    ax2.axis('off')
    ax2.set_title("Open Positions", color='white', fontsize=11,
                  fontweight='bold', loc='left', pad=8)

    pos_cols = ["Ticker", "Description", "Qty", "Avg Cost", "Mkt Val", "Unr P&L", "Delta", "Vega"]
    pos_rows = [[
        p.ticker,
        p.description[:28],
        f"{p.quantity:+d}",
        f"${p.avg_cost:.2f}",
        f"${p.market_value:.2f}",
        f"${p.unrealized_pnl:+,.0f}",
        f"{p.delta:.2f}",
        f"{p.vega:.2f}",
    ] for p in positions]

    if pos_rows:
        cell_colors = []
        for row in pos_rows:
            pnl_val = float(row[5].replace('$', '').replace(',', '').replace('+', ''))
            row_c = ['#1a1f2e'] * len(pos_cols)
            row_c[5] = '#1a3a1a' if pnl_val >= 0 else '#3a1a1a'
            cell_colors.append(row_c)

        tbl = ax2.table(cellText=pos_rows, colLabels=pos_cols,
                        cellLoc='center', loc='center',
                        cellColours=cell_colors)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)
        for (ri, ci), cell in tbl.get_celld().items():
            cell.set_text_props(color='white')
            cell.set_edgecolor('#333')
            if ri == 0:
                cell.set_facecolor('#1e3a5f')
                cell.set_text_props(color='#7fb3ff', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No open positions", transform=ax2.transAxes,
                 ha='center', color='#555', fontsize=12)

    # ── Panel 3: P&L Bar Chart ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PLOT_BG)

    if positions:
        tickers = [p.ticker + "\n" + p.description[:12] for p in positions]
        pnls    = [p.unrealized_pnl for p in positions]
        colors  = ['#2ecc71' if v >= 0 else '#e74c3c' for v in pnls]
        bars    = ax3.bar(range(len(tickers)), pnls, color=colors, edgecolor='#333', width=0.6)
        ax3.axhline(0, color='white', linewidth=0.8, alpha=0.5)
        ax3.set_xticks(range(len(tickers)))
        ax3.set_xticklabels(tickers, fontsize=7, color='#ccc')
        ax3.set_title("Unrealized P&L by Position", color='white',
                      fontsize=10, fontweight='bold', pad=8)
        ax3.set_ylabel("P&L ($)", color='#aaa', fontsize=8)
        ax3.tick_params(colors='#aaa', labelsize=7)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#333')
    else:
        ax3.text(0.5, 0.5, "No positions", ha='center', transform=ax3.transAxes,
                 color='#555', fontsize=11)
        ax3.set_facecolor(PLOT_BG)

    # ── Panel 4: Order Log ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor(PLOT_BG)
    ax4.axis('off')
    ax4.set_title("Order Log", color='white', fontsize=11,
                  fontweight='bold', loc='left', pad=8)

    if order_log:
        ord_cols = ["Order ID", "Ticker", "Status", "Filled Qty", "Fill Price", "Message"]
        ord_rows = [[
            str(o.order_id),
            o.ticker,
            o.status,
            str(o.filled_qty),
            f"${o.filled_price:.2f}" if o.filled_price else "—",
            o.message[:35],
        ] for o in order_log[-8:]]  # last 8 orders

        ord_cell_colors = []
        status_color_map = {
            "FILLED":    '#1a3a1a',
            "CANCELLED": '#3a2a1a',
            "ERROR":     '#3a1a1a',
            "BLOCKED":   '#3a1a1a',
            "PENDING":   '#1a2a3a',
        }
        for row in ord_rows:
            rc = ['#1a1f2e'] * len(ord_cols)
            rc[2] = status_color_map.get(row[2], '#1a1f2e')
            ord_cell_colors.append(rc)

        tbl2 = ax4.table(cellText=ord_rows, colLabels=ord_cols,
                         cellLoc='center', loc='center',
                         cellColours=ord_cell_colors)
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8.5)
        tbl2.scale(1, 1.65)
        for (ri, ci), cell in tbl2.get_celld().items():
            cell.set_text_props(color='white')
            cell.set_edgecolor('#333')
            if ri == 0:
                cell.set_facecolor('#1e3a5f')
                cell.set_text_props(color='#7fb3ff', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "No orders submitted yet",
                 transform=ax4.transAxes, ha='center', color='#555', fontsize=11)

    plt.savefig("ibkr_bridge.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\nDashboard saved to ibkr_bridge.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 12: IBKR Live Execution Bridge")
    print(f"  Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE (IBKR TWS)'}")
    print("=" * 65)

    # ── Step 1: Connect to broker ─────────────────────────────────────────────
    print("\n[1/6] Connecting to broker...")
    if SIMULATION_MODE:
        broker = SimulationBroker()
    else:
        broker = IBKRBroker()

    connected = broker.connect()
    if not connected and not SIMULATION_MODE:
        print("  Could not connect to IBKR. Switching to simulation mode.")
        broker = SimulationBroker()
        broker.connect()

    # ── Step 2: Account summary ───────────────────────────────────────────────
    print("\n[2/6] Fetching account summary...")
    account = broker.get_account_summary()
    print(f"  Net Liquidation : ${account.net_liquidation:,.2f}")
    print(f"  Available Funds : ${account.available_funds:,.2f}")
    print(f"  Buying Power    : ${account.buying_power:,.2f}")
    print(f"  Daily P&L       : ${account.daily_pnl:+,.2f}")

    # ── Step 3: Current positions ─────────────────────────────────────────────
    print("\n[3/6] Fetching open positions...")
    positions = broker.get_positions()
    net_greeks = broker.get_net_greeks(positions)
    print(f"  {len(positions)} open positions")
    print(f"  Net Delta: {net_greeks.get('net_delta',0):+.0f} shares equivalent")
    print(f"  Net Vega:  ${net_greeks.get('net_vega',0):+.0f}")
    for p in positions:
        print(f"  {p.ticker:6s} {p.quantity:+3d}x  {p.description[:30]:<30}  "
              f"UnrPnL=${p.unrealized_pnl:+,.0f}")

    # ── Step 4: Get options chain (example) ───────────────────────────────────
    print("\n[4/6] Pulling SPY options chain (example)...")
    calls, puts, chain_exp = broker.get_options_chain("SPY", "2026-05-16")
    if calls is not None and len(calls) > 0:
        spot_approx = calls['strike'].median()
        atm_calls = calls.iloc[(calls['strike'] - spot_approx).abs().argsort()[:3]]
        print(f"  Expiry: {chain_exp}  |  {len(calls)} call strikes, {len(puts)} put strikes")
        print(f"  ATM Call strikes:")
        for _, row in atm_calls.iterrows():
            iv_val = row.get('impliedVolatility', np.nan)
            print(f"    K={row['strike']:.0f}  Bid={row.get('bid',0):.2f}  "
                  f"Ask={row.get('ask',0):.2f}  "
                  f"IV={iv_val*100:.1f}%" if not np.isnan(iv_val) else
                  f"    K={row['strike']:.0f}  Bid={row.get('bid',0):.2f}  "
                  f"Ask={row.get('ask',0):.2f}")

    # ── Step 5: Signal-to-order pipeline ─────────────────────────────────────
    print("\n[5/6] Running signal-to-order pipeline (demo signals)...")
    risk_engine = RiskEngine(RISK_LIMITS)
    pipeline    = ExecutionPipeline(broker, risk_engine)

    demo_signals = [
        Signal(
            ticker="SPY", signal_type="sell_vol", strategy="cash_secured_put",
            direction="SELL", option_type="put", strike=520.0,
            expiry="2026-05-16", quantity=1, limit_price=5.20,
            notes="IVR=72, IV-RV gap=8.3% — screener top pick"
        ),
        Signal(
            ticker="SPY", signal_type="sell_vol", strategy="covered_call",
            direction="SELL", option_type="call", strike=575.0,
            expiry="2026-05-16", quantity=1, limit_price=4.80,
            notes="Hedging long delta from existing position"
        ),
        Signal(
            ticker="QQQ", signal_type="sell_vol", strategy="iron_condor",
            direction="SELL", option_type="call", strike=500.0,
            expiry="2026-05-16", quantity=20,  # This will trip the risk limit
            limit_price=3.50,
            notes="INTENTIONALLY OVERSIZED — should trip max contracts limit"
        ),
    ]

    for signal in demo_signals:
        result = pipeline.process_signal(signal)

    # ── Step 6: Monitor positions ──────────────────────────────────────────────
    print("\n[6/6] Position monitoring loop (5 seconds)...")
    pipeline.monitor_positions(duration_seconds=5)

    # ── Dashboard ─────────────────────────────────────────────────────────────
    print("\nRendering dashboard...")
    plot_dashboard(account, broker.get_positions(),
                   pipeline.order_log, net_greeks)

    # ── Disconnect ────────────────────────────────────────────────────────────
    broker.disconnect()
    print("\nDone.")

    print("\n" + "─" * 65)
    print("  CAPSTONE COMPLETE")
    print("  To use with real IBKR:")
    print("  1. Start TWS or IB Gateway (paper port 7497)")
    print("  2. Enable API in TWS settings")
    print("  3. Set SIMULATION_MODE = False at the top of this file")
    print("  4. Run again — all the same logic applies to live orders")
    print("─" * 65)


if __name__ == "__main__":
    main()
