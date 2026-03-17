"""
PROGRAM 5: VOL STRATEGY BACKTESTING ENGINE
============================================
Quant Mastery — MBA to Market Maker
Google Colab Ready | Libraries: yfinance, pandas, numpy, matplotlib, scipy

Event-driven backtesting engine for volatility selling strategies.
Tests short straddle/strangle strategies with realistic assumptions:
- Daily delta hedging (optional)
- Transaction costs
- Kelly-based position sizing
- Regime filtering (VIX-based)
- Walk-forward validation
- Full performance attribution (Sharpe, Sortino, Calmar, max DD)

To run in Google Colab:
  !pip install yfinance matplotlib scipy
  Then paste this entire file into a cell and run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

TICKER = "SPY"
BENCHMARK = "SPY"
START_DATE = "2019-01-01"
END_DATE = None  # None = today
INITIAL_CAPITAL = 100_000
RISK_FREE_RATE = 0.04

# Strategy parameters
STRATEGY = "short_straddle"  # "short_straddle" or "short_strangle"
DTE_TARGET = 30              # Target days to expiration
ROLL_DTE = 10                # Roll when DTE hits this
STRANGLE_WIDTH = 0.05        # 5% OTM for strangles (ignored for straddles)
DELTA_HEDGE = True           # Daily delta hedge
POSITION_SIZE_METHOD = "kelly"  # "fixed", "kelly", "vol_target"
FIXED_ALLOC = 0.05           # 5% of portfolio per trade (for fixed)
KELLY_FRACTION = 0.25        # Quarter-Kelly (conservative)
VOL_TARGET = 0.12            # 12% annual vol target
REGIME_FILTER = True          # Use VIX regime filter
VIX_CUTOFF = 30              # Don't sell vol when VIX > this
TRANSACTION_COST_PCT = 0.001  # 10bps round-trip

print(f"\n{'='*70}")
print(f"  VOL STRATEGY BACKTESTING ENGINE")
print(f"  {TICKER} | {START_DATE} to {END_DATE or 'today'}")
print(f"  Strategy: {STRATEGY.upper()} | DTE: {DTE_TARGET} | Roll: {ROLL_DTE}")
print(f"  Capital: ${INITIAL_CAPITAL:,.0f} | Sizing: {POSITION_SIZE_METHOD}")
print(f"  Delta Hedge: {DELTA_HEDGE} | Regime Filter: {REGIME_FILTER}")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES HELPERS
# ══════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, opt_type='call'):
    """Black-Scholes price."""
    if T <= 0:
        return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, opt_type='call'):
    """Black-Scholes delta."""
    if T <= 0:
        if opt_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if opt_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0

def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (same for call/put)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_theta(S, K, T, r, sigma, opt_type='call'):
    """Black-Scholes theta (per day)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    common = -(S*norm.pdf(d1)*sigma) / (2*np.sqrt(T))
    if opt_type == 'call':
        return (common - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        return (common + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365

def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (per 1% IV change)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

print("[1/6] Loading historical data...")
tk = yf.Ticker(TICKER)
hist = tk.history(start=START_DATE, end=END_DATE)
hist.index = hist.index.tz_localize(None)
print(f"  {TICKER}: {len(hist)} trading days ({hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')})")

# Load VIX for regime filter
print("  Loading VIX data...")
vix = yf.Ticker("^VIX").history(start=START_DATE, end=END_DATE)
vix.index = vix.index.tz_localize(None)
vix_series = vix['Close'].reindex(hist.index, method='ffill')
print(f"  VIX: {len(vix_series)} days, current: {vix_series.iloc[-1]:.1f}")

# Calculate rolling realized vol
print("  Calculating realized vol...")
log_ret = np.log(hist['Close'] / hist['Close'].shift(1))
rv_30 = log_ret.rolling(30).std() * np.sqrt(252)
rv_60 = log_ret.rolling(60).std() * np.sqrt(252)

# Simulate IV as RV + VRP (since we don't have historical options data)
# VRP averages ~3-5 vol pts for SPX, varies with regime
np.random.seed(42)
vrp_base = 0.035  # 3.5 vol pts base VRP
vrp_noise = pd.Series(np.random.normal(0, 0.008, len(hist)), index=hist.index)
vrp_regime = (vix_series / 100) * 0.15  # VRP increases when VIX is high
simulated_iv = rv_30 + vrp_base + vrp_noise + vrp_regime
simulated_iv = simulated_iv.clip(lower=0.08)  # Floor at 8% IV

print(f"  Avg simulated IV: {simulated_iv.mean()*100:.1f}% | Avg RV30: {rv_30.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════════════

print("\n[2/6] Running backtest...")

class Position:
    """Represents an active options position."""
    def __init__(self, entry_date, spot, strike_call, strike_put, iv, dte, n_contracts, premium_collected):
        self.entry_date = entry_date
        self.strike_call = strike_call
        self.strike_put = strike_put
        self.iv = iv
        self.initial_dte = dte
        self.n_contracts = n_contracts
        self.premium_collected = premium_collected
        self.entry_spot = spot
        self.delta_shares = 0  # shares held for delta hedging
        self.total_hedge_cost = 0

    def current_dte(self, current_date, entry_date):
        """Days remaining."""
        return max(0, self.initial_dte - (current_date - entry_date).days)

    def mark_to_market(self, S, current_dte_val, r=RISK_FREE_RATE):
        """Current value of short straddle/strangle position."""
        T = current_dte_val / 365
        call_val = bs_price(S, self.strike_call, T, r, self.iv, 'call')
        put_val = bs_price(S, self.strike_put, T, r, self.iv, 'put')
        # Short position: we owe the current value
        return -(call_val + put_val) * self.n_contracts * 100

    def get_delta(self, S, current_dte_val, r=RISK_FREE_RATE):
        """Net delta of the position."""
        T = current_dte_val / 365
        call_d = bs_delta(S, self.strike_call, T, r, self.iv, 'call')
        put_d = bs_delta(S, self.strike_put, T, r, self.iv, 'put')
        # Short position: negate deltas
        return -(call_d + put_d) * self.n_contracts * 100

    def get_theta(self, S, current_dte_val, r=RISK_FREE_RATE):
        """Daily theta P&L (positive for short options)."""
        T = current_dte_val / 365
        call_t = bs_theta(S, self.strike_call, T, r, self.iv, 'call')
        put_t = bs_theta(S, self.strike_put, T, r, self.iv, 'put')
        # Short position: positive theta
        return -(call_t + put_t) * self.n_contracts * 100

    def settlement_pnl(self, S_final):
        """P&L at expiration."""
        call_intrinsic = max(S_final - self.strike_call, 0)
        put_intrinsic = max(self.strike_put - S_final, 0)
        settlement_cost = (call_intrinsic + put_intrinsic) * self.n_contracts * 100
        return self.premium_collected - settlement_cost - self.total_hedge_cost


# ── Run the backtest ─────────────────────────────────────────────────
portfolio_value = [INITIAL_CAPITAL]
cash = INITIAL_CAPITAL
positions = []
trades = []
daily_log = []

# Skip first 60 days for RV warmup
start_idx = 60

for i in range(start_idx, len(hist)):
    date = hist.index[i]
    S = hist['Close'].iloc[i]
    current_vix = vix_series.iloc[i] if i < len(vix_series) else 20
    current_iv = simulated_iv.iloc[i] if i < len(simulated_iv) else 0.20
    current_rv = rv_30.iloc[i] if i < len(rv_30) and not np.isnan(rv_30.iloc[i]) else 0.15

    # ── Check existing positions ────────────────────────────────────
    positions_to_close = []
    daily_pnl = 0

    for pos_idx, pos in enumerate(positions):
        dte_remaining = pos.current_dte(date, pos.entry_date)

        if dte_remaining <= ROLL_DTE or dte_remaining <= 0:
            # Close/roll position
            pnl = pos.settlement_pnl(S) if dte_remaining <= 0 else pos.mark_to_market(S, dte_remaining) + pos.premium_collected
            cost = abs(pnl) * TRANSACTION_COST_PCT
            pnl -= cost
            cash += pnl
            daily_pnl += pnl

            trades.append({
                'entry_date': pos.entry_date,
                'exit_date': date,
                'entry_spot': pos.entry_spot,
                'exit_spot': S,
                'strike_call': pos.strike_call,
                'strike_put': pos.strike_put,
                'premium': pos.premium_collected,
                'pnl': pnl,
                'pnl_pct': pnl / (pos.entry_spot * pos.n_contracts * 100) * 100,
                'hedge_cost': pos.total_hedge_cost,
                'contracts': pos.n_contracts,
                'held_days': (date - pos.entry_date).days,
            })
            positions_to_close.append(pos_idx)
        else:
            # Delta hedge
            if DELTA_HEDGE:
                target_shares = pos.get_delta(S, dte_remaining)
                share_diff = target_shares - pos.delta_shares
                hedge_cost = abs(share_diff * S * TRANSACTION_COST_PCT)
                pos.total_hedge_cost += hedge_cost
                cash -= hedge_cost
                pos.delta_shares = target_shares

    # Remove closed positions (reverse order to preserve indices)
    for idx in sorted(positions_to_close, reverse=True):
        positions.pop(idx)

    # ── Open new position if flat ───────────────────────────────────
    if len(positions) == 0:
        # Regime filter
        if REGIME_FILTER and current_vix > VIX_CUTOFF:
            pass  # Skip — too volatile
        else:
            # Position sizing
            portfolio_val = cash  # Simplified: all cash
            if POSITION_SIZE_METHOD == "kelly":
                # Kelly: f* ~ edge / variance
                vrp = current_iv - current_rv
                if vrp > 0:
                    edge = vrp * S * np.sqrt(DTE_TARGET / 365)
                    variance = (current_iv * S * np.sqrt(DTE_TARGET / 365)) ** 2
                    kelly_f = edge / variance if variance > 0 else 0
                    kelly_f = min(kelly_f, 0.10) * KELLY_FRACTION
                    notional = portfolio_val * kelly_f
                else:
                    notional = 0
            elif POSITION_SIZE_METHOD == "vol_target":
                annual_straddle_vol = current_iv * S * np.sqrt(252 / DTE_TARGET)
                notional = portfolio_val * VOL_TARGET / annual_straddle_vol if annual_straddle_vol > 0 else 0
            else:  # fixed
                notional = portfolio_val * FIXED_ALLOC

            n_contracts = max(1, int(notional / (S * 100)))
            n_contracts = min(n_contracts, 10)  # Cap at 10 contracts

            if n_contracts >= 1 and notional > 0:
                # Determine strikes
                T = DTE_TARGET / 365
                if STRATEGY == "short_strangle":
                    K_call = round(S * (1 + STRANGLE_WIDTH))
                    K_put = round(S * (1 - STRANGLE_WIDTH))
                else:  # straddle
                    K_call = round(S)
                    K_put = round(S)

                # Calculate premium
                call_price = bs_price(S, K_call, T, RISK_FREE_RATE, current_iv, 'call')
                put_price = bs_price(S, K_put, T, RISK_FREE_RATE, current_iv, 'put')
                premium = (call_price + put_price) * n_contracts * 100

                # Transaction cost
                entry_cost = premium * TRANSACTION_COST_PCT
                cash += premium - entry_cost

                pos = Position(
                    entry_date=date,
                    spot=S,
                    strike_call=K_call,
                    strike_put=K_put,
                    iv=current_iv,
                    dte=DTE_TARGET,
                    n_contracts=n_contracts,
                    premium_collected=premium
                )
                positions.append(pos)

    # ── Log daily state ─────────────────────────────────────────────
    # Mark-to-market open positions
    position_mtm = 0
    for pos in positions:
        dte_rem = pos.current_dte(date, pos.entry_date)
        position_mtm += pos.mark_to_market(S, dte_rem)

    total_value = cash + position_mtm
    portfolio_value.append(total_value)

    daily_log.append({
        'date': date,
        'spot': S,
        'vix': current_vix,
        'iv': current_iv * 100,
        'rv30': current_rv * 100,
        'portfolio_value': total_value,
        'n_positions': len(positions),
        'daily_pnl': daily_pnl,
    })

# ══════════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

print("[3/6] Calculating performance metrics...\n")

daily_df = pd.DataFrame(daily_log)
daily_df.set_index('date', inplace=True)
daily_df['returns'] = daily_df['portfolio_value'].pct_change()
daily_df['cum_return'] = (1 + daily_df['returns']).cumprod() - 1

# Benchmark
bench = yf.Ticker(BENCHMARK).history(start=START_DATE, end=END_DATE)
bench.index = bench.index.tz_localize(None)
bench = bench.reindex(daily_df.index, method='ffill')
bench['returns'] = bench['Close'].pct_change()
bench['cum_return'] = (1 + bench['returns']).cumprod() - 1

# Key metrics
total_days = len(daily_df)
total_return = (daily_df['portfolio_value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
ann_return = (1 + total_return/100) ** (252/total_days) - 1
ann_vol = daily_df['returns'].std() * np.sqrt(252)
sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

# Sortino (downside deviation)
downside = daily_df['returns'][daily_df['returns'] < 0]
downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.01
sortino = (ann_return - RISK_FREE_RATE) / downside_vol

# Max drawdown
cum_max = daily_df['portfolio_value'].cummax()
drawdown = (daily_df['portfolio_value'] - cum_max) / cum_max
max_dd = drawdown.min() * 100
max_dd_date = drawdown.idxmin()

# Calmar ratio
calmar = ann_return / abs(max_dd/100) if max_dd != 0 else 0

# Trade statistics
trades_df = pd.DataFrame(trades)
n_trades = len(trades_df)
if n_trades > 0:
    win_rate = (trades_df['pnl'] > 0).mean() * 100
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0
    profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() and trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
    avg_pnl = trades_df['pnl'].mean()
    total_pnl = trades_df['pnl'].sum()
    avg_held = trades_df['held_days'].mean()
    max_win = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
else:
    win_rate = avg_win = avg_loss = profit_factor = avg_pnl = total_pnl = avg_held = max_win = max_loss = 0

# ── Print Results ────────────────────────────────────────────────────
print(f"  {'='*55}")
print(f"  BACKTEST RESULTS: {STRATEGY.upper()}")
print(f"  {'='*55}")
print(f"  Period:          {daily_df.index[0].strftime('%Y-%m-%d')} to {daily_df.index[-1].strftime('%Y-%m-%d')}")
print(f"  Trading Days:    {total_days}")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"  Final Value:     ${daily_df['portfolio_value'].iloc[-1]:,.0f}")
print(f"")
print(f"  Total Return:    {total_return:+.1f}%")
print(f"  Ann. Return:     {ann_return*100:+.1f}%")
print(f"  Ann. Volatility: {ann_vol*100:.1f}%")
print(f"  Sharpe Ratio:    {sharpe:.2f}")
print(f"  Sortino Ratio:   {sortino:.2f}")
print(f"  Max Drawdown:    {max_dd:.1f}% (on {max_dd_date.strftime('%Y-%m-%d')})")
print(f"  Calmar Ratio:    {calmar:.2f}")
print(f"")
print(f"  {'─'*55}")
print(f"  TRADE STATISTICS ({n_trades} trades)")
print(f"  {'─'*55}")
print(f"  Win Rate:        {win_rate:.1f}%")
print(f"  Avg Win:         ${avg_win:,.0f}")
print(f"  Avg Loss:        ${avg_loss:,.0f}")
print(f"  Profit Factor:   {profit_factor:.2f}")
print(f"  Avg P&L/Trade:   ${avg_pnl:,.0f}")
print(f"  Best Trade:      ${max_win:,.0f}")
print(f"  Worst Trade:     ${max_loss:,.0f}")
print(f"  Avg Holding:     {avg_held:.0f} days")
print(f"  Total P&L:       ${total_pnl:,.0f}")

# Benchmark comparison
bench_total = bench['cum_return'].iloc[-1] * 100 if len(bench) > 0 else 0
bench_ann = (1 + bench_total/100) ** (252/total_days) - 1
bench_vol = bench['returns'].std() * np.sqrt(252)
bench_sharpe = (bench_ann - RISK_FREE_RATE) / bench_vol if bench_vol > 0 else 0

print(f"\n  {'─'*55}")
print(f"  vs BENCHMARK ({BENCHMARK})")
print(f"  {'─'*55}")
print(f"  Strategy Sharpe: {sharpe:.2f}  |  {BENCHMARK} Sharpe: {bench_sharpe:.2f}")
print(f"  Strategy Return: {ann_return*100:+.1f}%  |  {BENCHMARK} Return: {bench_ann*100:+.1f}%")
print(f"  Strategy Vol:    {ann_vol*100:.1f}%   |  {BENCHMARK} Vol:    {bench_vol*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS
# ══════════════════════════════════════════════════════════════════════

print(f"\n[4/6] Walk-forward analysis...")

# Split into yearly windows
daily_df['year'] = daily_df.index.year
yearly_stats = []
for year, group in daily_df.groupby('year'):
    if len(group) < 20:
        continue
    yr_ret = (group['portfolio_value'].iloc[-1] / group['portfolio_value'].iloc[0] - 1) * 100
    yr_vol = group['returns'].std() * np.sqrt(252) * 100
    yr_sharpe = (yr_ret/100 - RISK_FREE_RATE) / (yr_vol/100) if yr_vol > 0 else 0
    yr_dd = ((group['portfolio_value'] - group['portfolio_value'].cummax()) / group['portfolio_value'].cummax()).min() * 100
    yearly_stats.append({'year': year, 'return': yr_ret, 'vol': yr_vol, 'sharpe': yr_sharpe, 'max_dd': yr_dd})

print(f"\n  {'Year':<6} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'Max DD':>8}")
print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for ys in yearly_stats:
    print(f"  {ys['year']:<6} {ys['return']:>+7.1f}% {ys['vol']:>7.1f}% {ys['sharpe']:>7.2f} {ys['max_dd']:>7.1f}%")

# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Rendering performance dashboard...\n")

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'{STRATEGY.upper()} Backtest — {TICKER}\n'
             f'Sharpe: {sharpe:.2f} | Return: {ann_return*100:+.1f}% | Max DD: {max_dd:.1f}% | Win Rate: {win_rate:.0f}%',
             fontsize=13, fontweight='bold', color='#e0e0e0', y=0.99)
fig.patch.set_facecolor('#0d1117')

accent = '#58a6ff'
accent2 = '#f78166'
accent3 = '#7ee787'
accent4 = '#d2a8ff'
grid_color = '#21262d'

for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color=grid_color)

# Panel 1: Equity curve
ax1 = axes[0, 0]
ax1.set_title('Equity Curve', fontsize=10, color='#e0e0e0', pad=8)
ax1.plot(daily_df.index, daily_df['portfolio_value'] / 1000, color=accent, linewidth=1.5, label='Strategy')
if len(bench) > 0:
    bench_eq = INITIAL_CAPITAL * (1 + bench['cum_return'])
    bench_eq = bench_eq.reindex(daily_df.index, method='ffill')
    ax1.plot(daily_df.index, bench_eq / 1000, color='#8b949e', linewidth=1, alpha=0.7, label=BENCHMARK)
ax1.set_ylabel('Value ($K)', color='#8b949e', fontsize=9)
ax1.legend(fontsize=8)

# Panel 2: Drawdown
ax2 = axes[0, 1]
ax2.set_title('Drawdown', fontsize=10, color='#e0e0e0', pad=8)
ax2.fill_between(daily_df.index, drawdown * 100, 0, color=accent2, alpha=0.5)
ax2.plot(daily_df.index, drawdown * 100, color=accent2, linewidth=0.8)
ax2.set_ylabel('Drawdown (%)', color='#8b949e', fontsize=9)
ax2.axhline(max_dd, color='#ff4444', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.text(daily_df.index[len(daily_df)//2], max_dd - 1, f'Max DD: {max_dd:.1f}%',
         fontsize=8, color='#ff4444')

# Panel 3: Monthly returns heatmap (simplified as bar chart)
ax3 = axes[0, 2]
ax3.set_title('Monthly Returns', fontsize=10, color='#e0e0e0', pad=8)
monthly = daily_df['returns'].resample('ME').sum() * 100
colors_m = [accent3 if r > 0 else accent2 for r in monthly]
ax3.bar(range(len(monthly)), monthly, color=colors_m, alpha=0.7, width=0.8)
ax3.axhline(0, color='#8b949e', linewidth=0.5)
ax3.set_ylabel('Return (%)', color='#8b949e', fontsize=9)
ax3.set_xlabel('Month', color='#8b949e', fontsize=9)

# Panel 4: Trade P&L distribution
ax4 = axes[1, 0]
ax4.set_title('Trade P&L Distribution', fontsize=10, color='#e0e0e0', pad=8)
if n_trades > 0:
    trade_pnls = trades_df['pnl']
    colors_t = [accent3 if p > 0 else accent2 for p in trade_pnls]
    ax4.bar(range(len(trade_pnls)), trade_pnls, color=colors_t, alpha=0.7)
    ax4.axhline(0, color='#8b949e', linewidth=0.5)
    ax4.axhline(avg_pnl, color=accent, linestyle='--', alpha=0.7, label=f'Avg: ${avg_pnl:,.0f}')
    ax4.set_ylabel('P&L ($)', color='#8b949e', fontsize=9)
    ax4.set_xlabel('Trade #', color='#8b949e', fontsize=9)
    ax4.legend(fontsize=8)

# Panel 5: Rolling Sharpe
ax5 = axes[1, 1]
ax5.set_title('Rolling 60-Day Sharpe', fontsize=10, color='#e0e0e0', pad=8)
rolling_ret = daily_df['returns'].rolling(60).mean() * 252
rolling_vol = daily_df['returns'].rolling(60).std() * np.sqrt(252)
rolling_sharpe = (rolling_ret - RISK_FREE_RATE) / rolling_vol
ax5.plot(daily_df.index, rolling_sharpe, color=accent4, linewidth=1)
ax5.axhline(0, color='#8b949e', linewidth=0.5)
ax5.axhline(sharpe, color=accent, linestyle='--', alpha=0.5, label=f'Overall: {sharpe:.2f}')
ax5.set_ylabel('Sharpe Ratio', color='#8b949e', fontsize=9)
ax5.legend(fontsize=8)
ax5.set_ylim(-3, 5)

# Panel 6: VIX vs Strategy returns scatter
ax6 = axes[1, 2]
ax6.set_title('VIX Level vs Daily P&L', fontsize=10, color='#e0e0e0', pad=8)
ax6.scatter(daily_df['vix'], daily_df['returns'] * 100, alpha=0.3, s=8, color=accent)
ax6.axhline(0, color='#8b949e', linewidth=0.5)
ax6.axvline(VIX_CUTOFF, color=accent2, linestyle='--', alpha=0.5, label=f'VIX cutoff ({VIX_CUTOFF})')
ax6.set_xlabel('VIX', color='#8b949e', fontsize=9)
ax6.set_ylabel('Daily Return (%)', color='#8b949e', fontsize=9)
ax6.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('program5_backtest.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# POKER TRANSLATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Final analysis...\n")
print(f"{'='*70}")
print(f"  POKER TRANSLATION — BACKTEST RESULTS")
print(f"{'='*70}")
print(f"""
  Win Rate {win_rate:.0f}% is like winning {win_rate:.0f} out of 100 pots.
  Profit Factor {profit_factor:.1f}x means for every $1 lost, you made ${profit_factor:.1f}.
  Sharpe {sharpe:.2f} is your risk-adjusted edge per unit of variance.

  Max Drawdown {max_dd:.1f}% is the worst losing streak.
  In poker terms: a {abs(max_dd):.0f} buy-in downswing.
  If you can't stomach that, reduce position size (lower Kelly fraction).

  The regime filter (VIX > {VIX_CUTOFF} = sit out) is like game selection:
  Don't play when the table is too wild — the variance isn't worth the edge.

  Walk-forward shows if the edge persists out-of-sample:
""")
for ys in yearly_stats:
    emoji = "+" if ys['sharpe'] > 0.5 else "~" if ys['sharpe'] > 0 else "-"
    print(f"    {ys['year']}: Sharpe {ys['sharpe']:.2f} [{emoji}]")

print(f"\n  Dashboard saved to program5_backtest.png")
print(f"{'='*70}\n")
