"""
PROGRAM 2: BLACK-SCHOLES PRICER + IV vs RV EDGE DETECTOR
=========================================================
Quant Mastery — MBA to Market Maker
Google Colab Ready | Libraries: yfinance, pandas, numpy, matplotlib, scipy

Full Black-Scholes pricer for calls and puts with all 5 Greeks.
Pulls 90-day SPY history, calculates realized vol (30/60/90 day windows),
and charts IV vs RV gap — the structural edge that vol sellers exploit.

To run in Google Colab:
  !pip install yfinance matplotlib scipy
  Then paste this entire file into a cell and run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm

# ── Install yfinance if needed (Colab) ──────────────────────────────
try:
    import yfinance as yf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: BLACK-SCHOLES PRICER
# ══════════════════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes option pricing formula.

    Parameters:
        S     : Current stock price
        K     : Strike price
        T     : Time to expiration (years)
        r     : Risk-free rate (annualized)
        sigma : Volatility (annualized)
        option_type : 'call' or 'put'

    Returns:
        price : Option price
    """
    if T <= 0 or sigma <= 0:
        # At expiration: intrinsic value only
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all 5 Greeks for a European option.

    Returns dict with: delta, gamma, theta, vega, rho
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for calls and puts) — per 1% vol move
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,   # per day
        'vega': vega,      # per 1% IV change
        'rho': rho         # per 1% rate change
    }


def implied_volatility(market_price, S, K, T, r, option_type='call',
                        tol=1e-6, max_iter=100):
    """
    Newton-Raphson implied volatility solver.

    Given a market price, finds the sigma that makes BS price = market price.
    """
    sigma = 0.25  # initial guess
    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega for Newton step (not divided by 100 here)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)

        if vega_raw < 1e-12:
            break

        sigma = sigma - diff / vega_raw
        sigma = max(sigma, 0.001)  # floor

    return sigma


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: LIVE DATA + PRICING
# ══════════════════════════════════════════════════════════════════════

TICKER = "SPY"
RISK_FREE_RATE = 0.043  # ~4.3% (current fed funds approximate)

print(f"\n{'='*60}")
print(f"  BLACK-SCHOLES PRICER + IV vs RV EDGE DETECTOR")
print(f"  Ticker: {TICKER} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}\n")

# ── Fetch current price ─────────────────────────────────────────────
print("[1/7] Fetching current price...")
spy = yf.Ticker(TICKER)
hist_5d = spy.history(period="5d")
spot = hist_5d['Close'].iloc[-1]
print(f"  {TICKER} spot: ${spot:.2f}")

# ── Fetch options chain for ~30 DTE ─────────────────────────────────
print("[2/7] Fetching options chain...")
expirations = spy.options
today = datetime.now()
exp_dates = [(e, (datetime.strptime(e, "%Y-%m-%d") - today).days) for e in expirations]
exp_dates = [(e, d) for e, d in exp_dates if d > 0]

# Find ~30 DTE
target_exp = min(exp_dates, key=lambda x: abs(x[1] - 30))
exp_str, dte = target_exp
T = dte / 365.0

chain = spy.option_chain(exp_str)
calls = chain.calls
puts = chain.puts

# Find ATM call
atm_idx = (calls['strike'] - spot).abs().idxmin()
atm_call = calls.loc[atm_idx]
K = atm_call['strike']
market_price_call = atm_call['lastPrice']
market_iv = atm_call['impliedVolatility']

print(f"  Expiration: {exp_str} ({dte} DTE)")
print(f"  ATM Strike: ${K:.0f}")
print(f"  Market Call Price: ${market_price_call:.2f}")
print(f"  Market IV: {market_iv*100:.1f}%")

# ── Price with Black-Scholes ────────────────────────────────────────
print("\n[3/7] Pricing with Black-Scholes...")

bs_call = black_scholes(spot, K, T, RISK_FREE_RATE, market_iv, 'call')
bs_put = black_scholes(spot, K, T, RISK_FREE_RATE, market_iv, 'put')
call_greeks = greeks(spot, K, T, RISK_FREE_RATE, market_iv, 'call')
put_greeks = greeks(spot, K, T, RISK_FREE_RATE, market_iv, 'put')

# Solve IV from market price
solved_iv = implied_volatility(market_price_call, spot, K, T, RISK_FREE_RATE, 'call')

print(f"\n  {'─'*50}")
print(f"  BLACK-SCHOLES PRICING")
print(f"  {'─'*50}")
print(f"  Inputs:")
print(f"    S = ${spot:.2f}  |  K = ${K:.0f}  |  T = {T:.4f} yr ({dte}d)")
print(f"    r = {RISK_FREE_RATE*100:.1f}%  |  sigma = {market_iv*100:.1f}%")
print(f"\n  BS Call Price:   ${bs_call:.2f}  (Market: ${market_price_call:.2f})")
print(f"  BS Put Price:    ${bs_put:.2f}")
print(f"  Solved IV:       {solved_iv*100:.1f}%  (from market price)")

# ── Greeks Display ───────────────────────────────────────────────────
print(f"\n  {'─'*50}")
print(f"  THE 5 GREEKS — ${K:.0f} Call, {dte} DTE")
print(f"  {'─'*50}")
print(f"  Delta  = {call_greeks['delta']:+.4f}   (Direction: {call_greeks['delta']*100:.1f}% stock-equivalent)")
print(f"    -> For every $1 {TICKER} moves, this call moves ${call_greeks['delta']:.2f}")
print(f"  Gamma  = {call_greeks['gamma']:.4f}    (Acceleration: delta changes by {call_greeks['gamma']:.4f} per $1)")
print(f"    -> Gamma is highest ATM — your delta is most unstable here")
print(f"  Theta  = {call_greeks['theta']:.4f}   (Time decay: loses ${abs(call_greeks['theta']):.2f}/day)")
print(f"    -> Like poker blinds: you pay ${abs(call_greeks['theta']):.2f} per day to hold this option")
print(f"  Vega   = {call_greeks['vega']:.4f}    (Vol sensitivity: ${call_greeks['vega']:.2f} per 1% IV change)")
print(f"    -> If IV rises 1%, this call gains ${call_greeks['vega']:.2f}")
print(f"  Rho    = {call_greeks['rho']:.4f}    (Rate sensitivity: ${call_greeks['rho']:.2f} per 1% rate change)")

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: REALIZED VOLATILITY + IV vs RV EDGE
# ══════════════════════════════════════════════════════════════════════

print(f"\n[4/7] Fetching historical prices (1 year)...")
hist = spy.history(period="1y")
hist['log_return'] = np.log(hist['Close'] / hist['Close'].shift(1))
print(f"  Got {len(hist)} trading days")

# ── Rolling Realized Vol ─────────────────────────────────────────────
print("[5/7] Calculating realized volatility...")

windows = [30, 60, 90]
for w in windows:
    hist[f'rv_{w}d'] = hist['log_return'].rolling(w).std() * np.sqrt(252)

rv_30 = hist['rv_30d'].iloc[-1]
rv_60 = hist['rv_60d'].iloc[-1]
rv_90 = hist['rv_90d'].iloc[-1]

print(f"  RV (30d): {rv_30*100:.1f}%")
print(f"  RV (60d): {rv_60*100:.1f}%")
print(f"  RV (90d): {rv_90*100:.1f}%")

# ── IV vs RV Gap (The Edge) ─────────────────────────────────────────
print(f"\n[6/7] Calculating IV vs RV edge...")
vrp = (market_iv - rv_30) * 100  # in percentage points

print(f"\n  {'─'*50}")
print(f"  THE EDGE: IV vs RV")
print(f"  {'─'*50}")
print(f"  ATM IV (market's fear): {market_iv*100:.1f}%")
print(f"  RV 30d (actual moves):  {rv_30*100:.1f}%")
print(f"  RV 60d:                 {rv_60*100:.1f}%")
print(f"  RV 90d:                 {rv_90*100:.1f}%")
print(f"\n  Vol Risk Premium (IV - RV30): {vrp:+.1f} percentage points")

if vrp > 3:
    print(f"  Signal: SELL VOL — IV is {vrp:.1f}pp rich to realized. Premium is fat.")
    print(f"  Poker: You're getting 2:1 on a coin flip. Take the bet every time.")
elif vrp > 0:
    print(f"  Signal: SLIGHT EDGE SELLING — IV marginally above RV. Small edge.")
    print(f"  Poker: Pot odds are slightly in your favor. Grind it.")
elif vrp > -3:
    print(f"  Signal: NEUTRAL — IV roughly equals RV. No clear edge either way.")
    print(f"  Poker: Break-even spot. Wait for better cards.")
else:
    print(f"  Signal: BUY VOL — IV is cheap vs realized. Options are underpriced.")
    print(f"  Poker: You're getting a discount on a strong hand. Load up.")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: 4-PANEL CHART
# ══════════════════════════════════════════════════════════════════════

print(f"\n[7/7] Rendering charts...\n")

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'{TICKER} — Black-Scholes + IV vs RV Edge Detector\n'
             f'Spot: ${spot:.2f} | ATM IV: {market_iv*100:.1f}% | RV30: {rv_30*100:.1f}% | VRP: {vrp:+.1f}pp',
             fontsize=13, fontweight='bold', color='#e0e0e0', y=0.98)
fig.patch.set_facecolor('#0d1117')

accent = '#58a6ff'
accent2 = '#f78166'
accent3 = '#7ee787'
accent4 = '#d2a8ff'
grid_color = '#21262d'

for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color=grid_color)

# ── Panel 1: Option Value vs Stock Price ─────────────────────────────
ax1 = axes[0, 0]
ax1.set_title(f'Call Value vs Stock Price (K=${K:.0f}, {dte}d)', fontsize=11, color='#e0e0e0', pad=10)

S_range = np.linspace(spot * 0.85, spot * 1.15, 200)
call_values = [black_scholes(s, K, T, RISK_FREE_RATE, market_iv, 'call') for s in S_range]
intrinsic = [max(s - K, 0) for s in S_range]

ax1.plot(S_range, call_values, color=accent, linewidth=2.5, label='BS Call Value')
ax1.plot(S_range, intrinsic, color='#8b949e', linewidth=1.5, linestyle='--', label='Intrinsic Value')
ax1.fill_between(S_range, intrinsic, call_values, alpha=0.15, color=accent, label='Time Value')
ax1.axvline(spot, color=accent3, linestyle='--', alpha=0.5, label=f'Spot ${spot:.0f}')
ax1.axhline(market_price_call, color=accent2, linestyle=':', alpha=0.5, label=f'Market ${market_price_call:.2f}')
ax1.set_xlabel(f'{TICKER} Price ($)', color='#8b949e', fontsize=9)
ax1.set_ylabel('Option Value ($)', color='#8b949e', fontsize=9)
ax1.legend(fontsize=8, loc='upper left')

# ── Panel 2: Greeks Sensitivity ──────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_title('Delta & Gamma vs Stock Price', fontsize=11, color='#e0e0e0', pad=10)

deltas = [greeks(s, K, T, RISK_FREE_RATE, market_iv, 'call')['delta'] for s in S_range]
gammas = [greeks(s, K, T, RISK_FREE_RATE, market_iv, 'call')['gamma'] for s in S_range]

ax2.plot(S_range, deltas, color=accent, linewidth=2, label='Delta')
ax2.axvline(spot, color=accent3, linestyle='--', alpha=0.5)
ax2.set_xlabel(f'{TICKER} Price ($)', color='#8b949e', fontsize=9)
ax2.set_ylabel('Delta', color='#8b949e', fontsize=9)

ax2b = ax2.twinx()
ax2b.plot(S_range, gammas, color=accent4, linewidth=2, label='Gamma')
ax2b.set_ylabel('Gamma', color=accent4, fontsize=9)
ax2b.tick_params(colors=accent4, labelsize=9)
ax2b.spines['right'].set_color(accent4)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

# ── Panel 3: IV vs RV Over Time ─────────────────────────────────────
ax3 = axes[1, 0]
ax3.set_title('IV vs Realized Volatility (Rolling)', fontsize=11, color='#e0e0e0', pad=10)

# Plot RV windows
plot_hist = hist.dropna(subset=['rv_30d']).copy()
ax3.plot(plot_hist.index, plot_hist['rv_30d'] * 100, color=accent, linewidth=1.5, label='RV 30d', alpha=0.9)
ax3.plot(plot_hist.index, plot_hist['rv_60d'] * 100, color=accent4, linewidth=1.5, label='RV 60d', alpha=0.7)
ax3.plot(plot_hist.index, plot_hist['rv_90d'] * 100, color='#8b949e', linewidth=1.5, label='RV 90d', alpha=0.6)

# Current IV line
ax3.axhline(market_iv * 100, color=accent2, linewidth=2, linestyle='--', label=f'Current IV {market_iv*100:.1f}%')

# Shade the VRP gap
ax3.fill_between(plot_hist.index, plot_hist['rv_30d'] * 100, market_iv * 100,
                 where=(plot_hist['rv_30d'] * 100 < market_iv * 100),
                 alpha=0.15, color=accent2, label='VRP (IV > RV)')

ax3.set_xlabel('Date', color='#8b949e', fontsize=9)
ax3.set_ylabel('Volatility (%)', color='#8b949e', fontsize=9)
ax3.legend(fontsize=7, loc='upper right')

# ── Panel 4: Daily Returns Distribution ──────────────────────────────
ax4 = axes[1, 1]
ax4.set_title('Daily Returns Distribution', fontsize=11, color='#e0e0e0', pad=10)

returns = hist['log_return'].dropna() * 100
ax4.hist(returns, bins=50, color=accent, alpha=0.7, edgecolor='none', density=True)

# Overlay normal distribution based on RV
x = np.linspace(returns.min(), returns.max(), 200)
mu = returns.mean()
std = returns.std()
normal_pdf = norm.pdf(x, mu, std)
ax4.plot(x, normal_pdf, color=accent2, linewidth=2, label=f'Normal (mu={mu:.2f}%, sig={std:.2f}%)')

# Show actual vs normal stats
skew = returns.skew()
kurt = returns.kurtosis()
ax4.text(0.02, 0.95, f'Skew: {skew:.2f}\nExcess Kurt: {kurt:.2f}',
         transform=ax4.transAxes, fontsize=9, color='#e0e0e0',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

ax4.set_xlabel('Daily Return (%)', color='#8b949e', fontsize=9)
ax4.set_ylabel('Density', color='#8b949e', fontsize=9)
ax4.legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('program2_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  EDGE SUMMARY")
print(f"{'='*60}")
print(f"  The market is pricing {market_iv*100:.1f}% annualized vol into {TICKER} options.")
print(f"  The stock has actually realized {rv_30*100:.1f}% vol over the past 30 days.")
print(f"  That's a {abs(vrp):.1f} percentage point {'premium' if vrp > 0 else 'discount'}.")
print(f"")
if vrp > 0:
    print(f"  Translation: Options are EXPENSIVE relative to actual movement.")
    print(f"  The market is overpaying for insurance by {vrp:.1f}pp.")
    print(f"  Edge: Sell volatility (short strangles, iron condors, covered calls).")
    print(f"  Poker: You're the house. The edge compounds over many hands.")
else:
    print(f"  Translation: Options are CHEAP relative to actual movement.")
    print(f"  The market is underpricing risk by {abs(vrp):.1f}pp.")
    print(f"  Edge: Buy volatility (long straddles, long calls/puts).")
    print(f"  Poker: You're getting implied odds better than true odds.")

print(f"\n  Dashboard saved to program2_dashboard.png")
print(f"{'='*60}\n")
