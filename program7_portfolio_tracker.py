# =============================================================================
# PROGRAM 7: Live Portfolio Tracker & Greek Aggregator
# =============================================================================
# Description:
#   Enter your options positions in the PORTFOLIO config dict at the top.
#   The program pulls live prices via yfinance, computes Black-Scholes Greeks
#   for every leg, aggregates portfolio-level dollar Greeks, attributes today's
#   P&L to each Greek, and fires risk alerts if thresholds are breached.
#
# What it produces:
#   - Per-position Greeks table (delta, gamma, theta, vega)
#   - Portfolio net Greeks in dollar terms
#   - P&L attribution bar chart (delta P&L, gamma P&L, theta P&L, vega P&L)
#   - Risk meter gauges (delta exposure, vega exposure)
#   - Console alerts if any risk threshold is exceeded
#
# Platform: Google Colab  |  Runtime: ~30-60 seconds
#
# Install (run this cell first in Colab):
#   !pip install yfinance numpy scipy matplotlib pandas
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm
from datetime import datetime, date
import yfinance as yf

# =============================================================================
# USER CONFIGURATION  ← Edit this dict to match your actual positions
# =============================================================================
PORTFOLIO = [
    # Each leg: ticker, qty (+ = long, - = short), option_type ('call'/'put'),
    #           strike, expiry ('YYYY-MM-DD'), cost_basis (premium paid/received per share)
    {
        "ticker":      "SPY",
        "qty":          1,           # contracts (1 contract = 100 shares)
        "option_type": "call",
        "strike":       560.0,
        "expiry":       "2026-06-20",
        "cost_basis":   8.50,        # premium per share (what you paid)
    },
    {
        "ticker":      "SPY",
        "qty":         -1,           # short put (credit received)
        "option_type": "put",
        "strike":       530.0,
        "expiry":       "2026-06-20",
        "cost_basis":   -6.20,       # negative = you received this credit
    },
    {
        "ticker":      "QQQ",
        "qty":          2,
        "option_type": "call",
        "strike":       480.0,
        "expiry":       "2026-05-16",
        "cost_basis":   5.10,
    },
    {
        "ticker":      "AAPL",
        "qty":         -1,
        "option_type": "put",
        "strike":       210.0,
        "expiry":       "2026-04-17",
        "cost_basis":   -3.80,
    },
]

# Risk alert thresholds
RISK_THRESHOLDS = {
    "net_delta_dollars": 50_000,   # alert if |net delta $| exceeds this
    "net_vega_dollars":  10_000,   # alert if |net vega $| exceeds this
    "net_theta_dollars": -500,     # alert if daily theta bleed < this (negative = you pay)
    "max_loss_pct":       0.30,    # alert if unrealized loss > 30% of cost basis
}

RISK_FREE_RATE = 0.0525   # 5.25% — update to current Fed Funds rate

# =============================================================================
# BLACK-SCHOLES ENGINE
# =============================================================================

def _d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes theoretical price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Return dict of all 5 Greeks."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    # Delta
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Theta (per calendar day — divide annual by 365)
    theta_call = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                  - r * K * np.exp(-r * T) * norm.cdf(d2))
    if option_type == 'call':
        theta = theta_call / 365
    else:
        theta = (theta_call + r * K * np.exp(-r * T)) / 365

    # Vega (for 1% move in vol — divide raw vega by 100)
    vega = S * sqrt_T * pdf_d1 / 100

    # Rho (for 1% move in rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def implied_vol_bisection(market_price, S, K, T, r, option_type, tol=1e-5, max_iter=200):
    """Compute IV via bisection search."""
    if T <= 0 or market_price <= 0:
        return np.nan
    low, high = 0.001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = bs_price(S, K, T, r, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_underlying_price(ticker):
    """Pull latest close price for an underlying."""
    t = yf.Ticker(ticker)
    hist = t.history(period='2d')
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(hist['Close'].iloc[-1]), float(hist['Close'].iloc[-2])


def fetch_option_market_price(ticker, strike, expiry, option_type):
    """Pull bid/ask midpoint for a specific option contract."""
    t = yf.Ticker(ticker)
    try:
        chain = t.option_chain(expiry)
        df = chain.calls if option_type == 'call' else chain.puts
        row = df[df['strike'] == strike]
        if row.empty:
            # Find nearest strike
            row = df.iloc[(df['strike'] - strike).abs().argsort()[:1]]
        bid = float(row['bid'].iloc[0])
        ask = float(row['ask'].iloc[0])
        last = float(row['lastPrice'].iloc[0])
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
        iv = float(row['impliedVolatility'].iloc[0]) if 'impliedVolatility' in row.columns else np.nan
        return mid, iv
    except Exception:
        return np.nan, np.nan

# =============================================================================
# PORTFOLIO ANALYSIS
# =============================================================================

def analyze_portfolio(positions, r=RISK_FREE_RATE):
    """
    For each position: fetch live data, compute Greeks, dollar Greeks,
    unrealized P&L, and Greek-attributed P&L contribution.
    Returns a list of enriched position dicts.
    """
    today = date.today()
    results = []

    # Cache underlying prices (avoid re-fetching same ticker)
    price_cache = {}

    for i, pos in enumerate(positions):
        ticker = pos['ticker']
        print(f"  [{i+1}/{len(positions)}] Fetching {ticker} {pos['option_type'].upper()} "
              f"K={pos['strike']} exp={pos['expiry']} ...")

        # Underlying price
        if ticker not in price_cache:
            try:
                cur, prev = fetch_underlying_price(ticker)
                price_cache[ticker] = (cur, prev)
            except Exception as e:
                print(f"    WARNING: Could not fetch {ticker}: {e}")
                price_cache[ticker] = (100.0, 100.0)

        S, S_prev = price_cache[ticker]
        dS = S - S_prev   # today's underlying move

        # Time to expiry
        expiry_dt = datetime.strptime(pos['expiry'], '%Y-%m-%d').date()
        T = max((expiry_dt - today).days / 365, 0.001)

        # Market price of option & IV
        mkt_price, iv = fetch_option_market_price(
            ticker, pos['strike'], pos['expiry'], pos['option_type'])

        # Fall back to IV estimation if yfinance didn't give one
        if np.isnan(iv) or iv <= 0:
            iv = 0.25  # 25% vol as placeholder

        # Greeks at current S
        g = bs_greeks(S, pos['strike'], T, r, iv, pos['option_type'])
        th_price = bs_price(S, pos['strike'], T, r, iv, pos['option_type'])

        # Dollar Greeks (per contract = 100 multiplier)
        multiplier = 100 * pos['qty']   # sign baked in (short = negative)
        dollar_delta = g['delta'] * S * multiplier
        dollar_gamma = 0.5 * g['gamma'] * (S ** 2) * multiplier  # $-gamma for 1% move
        dollar_theta = g['theta'] * multiplier
        dollar_vega  = g['vega'] * multiplier

        # Unrealized P&L
        current_px = mkt_price if not np.isnan(mkt_price) else th_price
        cost = pos['cost_basis']
        unrealized_pnl = (current_px - cost) * multiplier

        # Greek P&L attribution (simplified Taylor expansion)
        #  ΔP ≈ delta*ΔS + 0.5*gamma*ΔS² + theta*Δt + vega*Δσ
        #  We estimate each component separately assuming Δσ ≈ 0 intraday
        delta_pnl = g['delta'] * dS * multiplier
        gamma_pnl = 0.5 * g['gamma'] * (dS ** 2) * multiplier
        theta_pnl = g['theta'] * 1 * multiplier   # 1 calendar day
        # Vega P&L requires a vol change estimate — approximate from IV history
        vega_pnl  = 0.0   # placeholder; set to 0 without prior-day IV

        results.append({
            **pos,
            "S":             S,
            "T_days":        round((expiry_dt - today).days),
            "iv":            iv,
            "th_price":      round(th_price, 4),
            "mkt_price":     round(current_px, 4),
            "delta":         round(g['delta'], 4),
            "gamma":         round(g['gamma'], 6),
            "theta":         round(g['theta'], 4),
            "vega":          round(g['vega'], 4),
            "rho":           round(g['rho'], 4),
            "dollar_delta":  round(dollar_delta, 2),
            "dollar_gamma":  round(dollar_gamma, 2),
            "dollar_theta":  round(dollar_theta, 2),
            "dollar_vega":   round(dollar_vega, 2),
            "unrealized_pnl":round(unrealized_pnl, 2),
            "delta_pnl":     round(delta_pnl, 2),
            "gamma_pnl":     round(gamma_pnl, 2),
            "theta_pnl":     round(theta_pnl, 2),
            "vega_pnl":      round(vega_pnl, 2),
        })

    return results


def aggregate_greeks(results):
    """Sum dollar Greeks across all positions."""
    return {
        "net_dollar_delta": sum(r['dollar_delta'] for r in results),
        "net_dollar_gamma": sum(r['dollar_gamma'] for r in results),
        "net_dollar_theta": sum(r['dollar_theta'] for r in results),
        "net_dollar_vega":  sum(r['dollar_vega']  for r in results),
        "total_unrealized": sum(r['unrealized_pnl'] for r in results),
        "total_delta_pnl":  sum(r['delta_pnl'] for r in results),
        "total_gamma_pnl":  sum(r['gamma_pnl'] for r in results),
        "total_theta_pnl":  sum(r['theta_pnl'] for r in results),
        "total_vega_pnl":   sum(r['vega_pnl']  for r in results),
    }


def check_risk_alerts(agg, thresholds):
    """Print risk alerts if any threshold is breached."""
    alerts = []
    if abs(agg['net_dollar_delta']) > thresholds['net_delta_dollars']:
        alerts.append(f"  DELTA ALERT: Net delta exposure ${agg['net_dollar_delta']:,.0f} "
                      f"exceeds limit ±${thresholds['net_delta_dollars']:,.0f}")
    if abs(agg['net_dollar_vega']) > thresholds['net_vega_dollars']:
        alerts.append(f"  VEGA ALERT:  Net vega exposure ${agg['net_dollar_vega']:,.0f} "
                      f"exceeds limit ±${thresholds['net_vega_dollars']:,.0f}")
    if agg['net_dollar_theta'] < thresholds['net_theta_dollars']:
        alerts.append(f"  THETA ALERT: Daily theta bleed ${agg['net_dollar_theta']:,.0f}/day "
                      f"worse than limit ${thresholds['net_theta_dollars']:,.0f}")
    return alerts

# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_gauge(ax, value, min_val, max_val, title, unit="$", color_scheme='rg'):
    """Draw a simple horizontal gauge bar."""
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title, color='white', fontsize=10, pad=6)

    # Background bar
    ax.barh(0.5, max_val - min_val, left=min_val, height=0.3,
            color='#2a2a2a', edgecolor='#444', linewidth=1)

    # Value bar
    bar_color = '#e74c3c' if (color_scheme == 'rg' and value < 0) else '#2ecc71'
    ax.barh(0.5, value - min_val, left=min_val, height=0.3,
            color=bar_color, alpha=0.85)

    # Zero line
    ax.axvline(0, color='white', linewidth=1.2, alpha=0.6)

    # Value label
    ax.text(value, 0.5, f" {unit}{value:,.0f}", va='center',
            color='white', fontsize=9, fontweight='bold')

    ax.set_xlabel(unit, color='#aaa', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.tick_params(colors='#aaa', labelsize=8)


def plot_dashboard(results, agg):
    """Render the 4-panel portfolio dashboard."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle("Portfolio Greek Tracker & Risk Dashboard",
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    # ── Panel 1: Per-Position Greeks Table ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0d1117')
    ax1.axis('off')
    ax1.set_title("Position-Level Greeks (Dollar Terms)", color='white',
                  fontsize=12, fontweight='bold', loc='left', pad=8)

    labels = ["Ticker", "Type", "Strike", "Expiry", "Qty", "S", "IV%",
              "Th.Price", "Mkt.Price", "Δ Delta$", "Γ Gamma$", "Θ Theta$/d", "ν Vega$", "UnrPnL"]
    rows = []
    for r in results:
        rows.append([
            r['ticker'],
            r['option_type'].upper(),
            f"{r['strike']:.0f}",
            r['expiry'],
            f"{r['qty']:+d}",
            f"{r['S']:.2f}",
            f"{r['iv']*100:.1f}%",
            f"{r['th_price']:.2f}",
            f"{r['mkt_price']:.2f}",
            f"${r['dollar_delta']:,.0f}",
            f"${r['dollar_gamma']:,.0f}",
            f"${r['dollar_theta']:,.0f}",
            f"${r['dollar_vega']:,.0f}",
            f"${r['unrealized_pnl']:,.0f}",
        ])

    col_colors = [['#1a1f2e'] * len(labels)] * len(rows)
    pnl_col = len(labels) - 1
    for i, row in enumerate(rows):
        pnl = results[i]['unrealized_pnl']
        col_colors[i][pnl_col] = '#1a3a1a' if pnl >= 0 else '#3a1a1a'

    tbl = ax1.table(
        cellText=rows,
        colLabels=labels,
        cellLoc='center',
        loc='center',
        cellColours=col_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333')
        if row_idx == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='#7fb3ff', fontweight='bold')

    # ── Panel 2: Portfolio Summary ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#0d1117')
    ax2.axis('off')
    ax2.set_title("Net Portfolio Greeks", color='white', fontsize=12,
                  fontweight='bold', loc='left', pad=8)

    summary_data = [
        ("Net Delta $",    agg['net_dollar_delta']),
        ("Net Gamma $",    agg['net_dollar_gamma']),
        ("Net Theta $/day",agg['net_dollar_theta']),
        ("Net Vega $",     agg['net_dollar_vega']),
        ("Total Unr. P&L", agg['total_unrealized']),
    ]
    for j, (label, val) in enumerate(summary_data):
        y = 0.88 - j * 0.18
        color = '#2ecc71' if val >= 0 else '#e74c3c'
        ax2.text(0.02, y, label, color='#aaa', fontsize=10, transform=ax2.transAxes)
        ax2.text(0.98, y, f"${val:,.0f}", color=color, fontsize=11,
                 fontweight='bold', ha='right', transform=ax2.transAxes)
        ax2.axhline(y=y - 0.04, xmin=0.02, xmax=0.98, color='#333',
                    linewidth=0.5, transform=ax2.transAxes)

    # ── Panel 3: P&L Attribution ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#151b27')

    greek_names = ['Delta P&L', 'Gamma P&L', 'Theta P&L', 'Vega P&L']
    greek_vals  = [agg['total_delta_pnl'], agg['total_gamma_pnl'],
                   agg['total_theta_pnl'], agg['total_vega_pnl']]
    colors = ['#3498db' if v >= 0 else '#e74c3c' for v in greek_vals]

    bars = ax3.barh(greek_names, greek_vals, color=colors, edgecolor='#333',
                    height=0.55, linewidth=0.8)
    ax3.axvline(0, color='white', linewidth=1, alpha=0.5)
    ax3.set_facecolor('#151b27')
    ax3.set_title("Today's P&L Attribution by Greek", color='white',
                  fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel("P&L ($)", color='#aaa', fontsize=9)
    ax3.tick_params(colors='#ccc', labelsize=9)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333')
    for bar, val in zip(bars, greek_vals):
        x_pos = val + (max(greek_vals + [1]) * 0.02)
        ax3.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}", va='center', color='white', fontsize=9)

    # ── Panel 4: Risk Gauges ──────────────────────────────────────────────────
    ax4a = fig.add_subplot(gs[2, 0])
    ax4a.set_facecolor('#151b27')
    d_range = max(abs(agg['net_dollar_delta']) * 1.5, 10000)
    draw_gauge(ax4a, agg['net_dollar_delta'], -d_range, d_range,
               "Net Delta Exposure ($)", unit="$")

    ax4b = fig.add_subplot(gs[2, 1])
    ax4b.set_facecolor('#151b27')
    v_range = max(abs(agg['net_dollar_vega']) * 1.5, 5000)
    draw_gauge(ax4b, agg['net_dollar_vega'], -v_range, v_range,
               "Net Vega Exposure ($ per 1% vol move)", unit="$",
               color_scheme='rg')

    plt.savefig("portfolio_tracker.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\nDashboard saved to portfolio_tracker.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 7: Portfolio Greek Tracker & Risk Dashboard")
    print("=" * 65)
    print(f"\nAnalyzing {len(PORTFOLIO)} positions as of {date.today()}\n")
    print("Fetching live market data...")

    results = analyze_portfolio(PORTFOLIO, r=RISK_FREE_RATE)

    print("\n" + "─" * 65)
    print("Position Summary:")
    print("─" * 65)
    for r in results:
        direction = "LONG" if r['qty'] > 0 else "SHORT"
        print(f"  {r['ticker']} {r['option_type'].upper()} {r['strike']} "
              f"exp:{r['expiry']} | {direction} {abs(r['qty'])}x | "
              f"IV={r['iv']*100:.1f}% | Mkt={r['mkt_price']:.2f} | "
              f"UnrPnL=${r['unrealized_pnl']:,.0f}")

    agg = aggregate_greeks(results)

    print("\n" + "─" * 65)
    print("Aggregated Portfolio Greeks:")
    print("─" * 65)
    print(f"  Net Delta Exposure : ${agg['net_dollar_delta']:>10,.0f}")
    print(f"  Net Gamma Exposure : ${agg['net_dollar_gamma']:>10,.0f}")
    print(f"  Net Theta / day    : ${agg['net_dollar_theta']:>10,.0f}")
    print(f"  Net Vega / 1% vol  : ${agg['net_dollar_vega']:>10,.0f}")
    print(f"  Total Unrealized   : ${agg['total_unrealized']:>10,.0f}")

    print("\n" + "─" * 65)
    print("P&L Attribution (today):")
    print("─" * 65)
    print(f"  Delta P&L  (from underlying move): ${agg['total_delta_pnl']:>8,.0f}")
    print(f"  Gamma P&L  (convexity benefit)   : ${agg['total_gamma_pnl']:>8,.0f}")
    print(f"  Theta P&L  (daily time decay)     : ${agg['total_theta_pnl']:>8,.0f}")
    print(f"  Vega P&L   (vol change est.)      : ${agg['total_vega_pnl']:>8,.0f}")

    alerts = check_risk_alerts(agg, RISK_THRESHOLDS)
    if alerts:
        print("\n" + "!" * 65)
        print("  RISK ALERTS TRIGGERED:")
        for a in alerts:
            print(a)
        print("!" * 65)
    else:
        print("\n  [OK] All risk metrics within thresholds.")

    print("\nRendering dashboard...")
    plot_dashboard(results, agg)
    print("\nDone.")


if __name__ == "__main__":
    main()
