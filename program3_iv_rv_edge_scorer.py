"""
╔══════════════════════════════════════════════════════════════════╗
║  PROGRAM 3: Rolling IV vs RV Edge Scorer                        ║
║  Quant Mastery — MBA to Market Maker                            ║
║                                                                  ║
║  What it does:                                                   ║
║  • Pulls 1 year of price history for any ticker (default: SPY)  ║
║  • Calculates Realized Volatility (30/60/90 day rolling)        ║
║  • Pulls current options chain to get Implied Volatility        ║
║  • Computes IV Rank, IV Percentile, VRP signal                  ║
║  • Outputs a clear SELL VOL / HOLD / BUY VOL signal             ║
║  • 5-panel dark dashboard with rolling edge visualization       ║
║                                                                  ║
║  Paste into Google Colab and run. Free data via yfinance.       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── INSTALL (Colab already has these, but just in case) ────────
# !pip install yfinance matplotlib pandas numpy -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Change these to scan any ticker
# ═══════════════════════════════════════════════════════════════
TICKER = 'SPY'           # Any optionable stock/ETF
LOOKBACK_DAYS = 365      # 1 year of history for IV rank calc
RV_WINDOWS = [30, 60, 90]  # Rolling RV windows (trading days)

# Edge thresholds (in vol points)
STRONG_SELL_THRESHOLD = 5.0   # VRP > 5 = strong sell vol signal
SELL_THRESHOLD = 2.0          # VRP > 2 = sell vol
BUY_THRESHOLD = -2.0          # VRP < -2 = buy vol (rare)

print(f"{'='*60}")
print(f"  PROGRAM 3: IV vs RV Edge Scorer")
print(f"  Ticker: {TICKER}")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════
# STEP 1: Pull Historical Price Data
# ═══════════════════════════════════════════════════════════════
print("📊 Pulling price history...")
ticker = yf.Ticker(TICKER)
end_date = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_DAYS + 100)  # extra buffer
hist = ticker.history(start=start_date, end=end_date)

if hist.empty:
    raise ValueError(f"No data returned for {TICKER}. Check the ticker symbol.")

# Calculate log returns
hist['log_return'] = np.log(hist['Close'] / hist['Close'].shift(1))
hist = hist.dropna()
print(f"   ✓ {len(hist)} trading days loaded ({hist.index[0].strftime('%Y-%m-%d')} → {hist.index[-1].strftime('%Y-%m-%d')})")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Calculate Rolling Realized Volatility
# ═══════════════════════════════════════════════════════════════
print("📐 Calculating Realized Volatility...")
for window in RV_WINDOWS:
    col_name = f'RV_{window}d'
    hist[col_name] = hist['log_return'].rolling(window=window).std() * np.sqrt(252) * 100
    current_rv = hist[col_name].iloc[-1]
    print(f"   ✓ RV {window}-day: {current_rv:.1f}%")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Get Current Implied Volatility from Options Chain
# ═══════════════════════════════════════════════════════════════
print("\n📋 Pulling options chain for IV...")
try:
    # Get available expiration dates
    expirations = ticker.options
    if not expirations:
        raise ValueError("No options data available")

    # Pick nearest expiration 20-45 DTE (sweet spot for vol selling)
    today = datetime.now().date()
    target_dte = 30
    best_exp = None
    best_diff = float('inf')

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        if 14 <= dte <= 60:  # reasonable range
            diff = abs(dte - target_dte)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp_str

    if best_exp is None:
        best_exp = expirations[min(2, len(expirations)-1)]

    exp_date = datetime.strptime(best_exp, '%Y-%m-%d').date()
    dte = (exp_date - today).days
    print(f"   Using expiration: {best_exp} ({dte} DTE)")

    # Get the chain
    chain = ticker.option_chain(best_exp)
    calls = chain.calls
    puts = chain.puts

    # Current price
    current_price = hist['Close'].iloc[-1]
    print(f"   Current price: ${current_price:.2f}")

    # Find ATM options (closest to current price)
    calls['dist'] = abs(calls['strike'] - current_price)
    puts['dist'] = abs(puts['strike'] - current_price)
    atm_call = calls.loc[calls['dist'].idxmin()]
    atm_put = puts.loc[puts['dist'].idxmin()]

    # ATM IV = average of ATM call and put IV
    atm_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2 * 100
    print(f"   ATM Call IV: {atm_call['impliedVolatility']*100:.1f}%  (strike ${atm_call['strike']:.0f})")
    print(f"   ATM Put IV:  {atm_put['impliedVolatility']*100:.1f}%  (strike ${atm_put['strike']:.0f})")
    print(f"   ✓ ATM IV (avg): {atm_iv:.1f}%")

    # Also grab 25-delta puts for skew
    otm_put_strike = current_price * 0.95  # ~5% OTM
    puts['dist_otm'] = abs(puts['strike'] - otm_put_strike)
    otm_put = puts.loc[puts['dist_otm'].idxmin()]
    skew = otm_put['impliedVolatility'] * 100 - atm_iv
    print(f"   Vol skew (25d put - ATM): {skew:+.1f} pts")

except Exception as e:
    print(f"   ⚠ Options data issue: {e}")
    print(f"   Using VIX as IV proxy...")
    vix = yf.Ticker('^VIX')
    vix_hist = vix.history(period='5d')
    atm_iv = vix_hist['Close'].iloc[-1]
    skew = 0
    dte = 30
    print(f"   ✓ VIX (IV proxy): {atm_iv:.1f}%")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Calculate IV Rank & IV Percentile (1-year lookback)
# ═══════════════════════════════════════════════════════════════
print("\n📈 Computing IV metrics...")

# For IV Rank/Percentile, we'll use the 30-day RV as a proxy for
# historical IV levels (since free IV history isn't available).
# In production, you'd use actual historical IV data.
# VIX gives us a better proxy for SPY specifically.

try:
    vix = yf.Ticker('^VIX')
    vix_hist = vix.history(start=start_date, end=end_date)
    iv_series = vix_hist['Close']
    iv_source = 'VIX'
except:
    # Fallback: use RV_30d as rough proxy
    iv_series = hist['RV_30d'].dropna()
    iv_source = 'RV_30d proxy'

iv_1yr = iv_series.tail(252)  # last year
iv_high = iv_1yr.max()
iv_low = iv_1yr.min()
iv_current = atm_iv

# IV Rank = (current - 1yr low) / (1yr high - 1yr low)
iv_rank = (iv_current - iv_low) / (iv_high - iv_low) * 100 if iv_high != iv_low else 50

# IV Percentile = % of days in last year where IV was LOWER than current
iv_percentile = (iv_1yr < iv_current).sum() / len(iv_1yr) * 100

print(f"   IV source: {iv_source}")
print(f"   1-year IV range: {iv_low:.1f}% – {iv_high:.1f}%")
print(f"   ✓ IV Rank: {iv_rank:.0f}th (current vs 1yr range)")
print(f"   ✓ IV Percentile: {iv_percentile:.0f}th (% of days below current)")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Compute VRP & Edge Score
# ═══════════════════════════════════════════════════════════════
print("\n🎯 Computing Vol Risk Premium & Edge Score...")
rv_30 = hist['RV_30d'].iloc[-1]
rv_60 = hist['RV_60d'].iloc[-1]
rv_90 = hist['RV_90d'].iloc[-1]

# VRP = IV - RV (positive = IV overpricing = sell vol opportunity)
vrp_30 = atm_iv - rv_30
vrp_60 = atm_iv - rv_60
vrp_90 = atm_iv - rv_90

print(f"   VRP (IV - RV30): {vrp_30:+.1f} vol pts")
print(f"   VRP (IV - RV60): {vrp_60:+.1f} vol pts")
print(f"   VRP (IV - RV90): {vrp_90:+.1f} vol pts")

# Composite edge score (weighted average)
vrp_composite = 0.5 * vrp_30 + 0.3 * vrp_60 + 0.2 * vrp_90

# Edge score: combines VRP with IV rank
# High IV rank + positive VRP = strongest sell signal
edge_score = vrp_composite * (1 + (iv_rank - 50) / 100)

print(f"\n   Composite VRP: {vrp_composite:+.1f}")
print(f"   Edge Score: {edge_score:+.1f}")

# ═══════════════════════════════════════════════════════════════
# STEP 6: Generate Signal
# ═══════════════════════════════════════════════════════════════
if vrp_composite > STRONG_SELL_THRESHOLD and iv_rank > 50:
    signal = "🔥 STRONG SELL VOL"
    signal_detail = "IV is expensive AND overpricing realized moves. Premium selling is highly favorable."
    signal_color = '#00ff88'
elif vrp_composite > SELL_THRESHOLD:
    signal = "✅ SELL VOL"
    signal_detail = "Positive VRP — IV is overpricing. Standard vol-selling conditions."
    signal_color = '#88ff88'
elif vrp_composite < BUY_THRESHOLD:
    signal = "🛒 BUY VOL"
    signal_detail = "Rare: IV is UNDER-pricing realized moves. Consider long vol or sit out."
    signal_color = '#ff4444'
else:
    signal = "⏸ HOLD / NO EDGE"
    signal_detail = "VRP is too narrow. No clear edge — wait for better setup."
    signal_color = '#ffaa00'

print(f"\n{'='*60}")
print(f"  SIGNAL: {signal}")
print(f"  {signal_detail}")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
# STEP 7: 5-Panel Dashboard
# ═══════════════════════════════════════════════════════════════
print("\n🎨 Rendering dashboard...")

# Dark theme
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.25,
                       left=0.06, right=0.97, top=0.92, bottom=0.05)

# Title
fig.suptitle(f'{TICKER} — IV vs RV Edge Scanner  |  {datetime.now().strftime("%Y-%m-%d")}',
             fontsize=16, fontweight='bold', color='white', y=0.97)

# ── Panel 1: Price chart with vol regime shading ──
ax1 = fig.add_subplot(gs[0, 0])
price_data = hist['Close'].tail(180)
ax1.plot(price_data.index, price_data.values, color='white', linewidth=1.2, alpha=0.9)
ax1.fill_between(price_data.index, price_data.values, alpha=0.05, color='white')
ax1.set_title(f'{TICKER} Price (6M)', fontsize=11, fontweight='bold', color='white')
ax1.set_ylabel('Price ($)', fontsize=9, color='#888')
ax1.tick_params(colors='#888', labelsize=8)
ax1.grid(alpha=0.1)
for spine in ax1.spines.values(): spine.set_color('#333')

# ── Panel 2: Rolling RV comparison ──
ax2 = fig.add_subplot(gs[0, 1])
rv_data = hist[['RV_30d', 'RV_60d', 'RV_90d']].tail(180)
ax2.plot(rv_data.index, rv_data['RV_30d'], color='#ffffff', linewidth=1.5, label='RV 30d', alpha=0.9)
ax2.plot(rv_data.index, rv_data['RV_60d'], color='#aaaaaa', linewidth=1.2, label='RV 60d', alpha=0.7)
ax2.plot(rv_data.index, rv_data['RV_90d'], color='#666666', linewidth=1.0, label='RV 90d', alpha=0.6)
ax2.axhline(y=atm_iv, color='#00ff88', linestyle='--', linewidth=1.5, label=f'Current IV ({atm_iv:.1f}%)', alpha=0.8)
ax2.set_title('Realized Vol vs Implied Vol', fontsize=11, fontweight='bold', color='white')
ax2.set_ylabel('Volatility (%)', fontsize=9, color='#888')
ax2.legend(fontsize=7, loc='upper right', framealpha=0.3)
ax2.tick_params(colors='#888', labelsize=8)
ax2.grid(alpha=0.1)
for spine in ax2.spines.values(): spine.set_color('#333')

# ── Panel 3: VRP over time (the edge chart) ──
ax3 = fig.add_subplot(gs[1, :])
# Calculate rolling VRP using VIX if available, else RV proxy
if iv_source == 'VIX':
    # Align VIX with hist dates
    aligned_iv = iv_series.reindex(hist.index, method='ffill')
    hist['iv_proxy'] = aligned_iv
else:
    hist['iv_proxy'] = hist['RV_30d'].shift(-20)  # rough forward proxy

vrp_series = hist['iv_proxy'] - hist['RV_30d']
vrp_plot = vrp_series.tail(180).dropna()

colors_vrp = ['#00ff88' if v > SELL_THRESHOLD else '#ff4444' if v < BUY_THRESHOLD else '#ffaa00'
              for v in vrp_plot.values]
ax3.bar(vrp_plot.index, vrp_plot.values, color=colors_vrp, alpha=0.7, width=1.5)
ax3.axhline(y=0, color='white', linewidth=0.8, alpha=0.3)
ax3.axhline(y=SELL_THRESHOLD, color='#00ff88', linewidth=0.8, linestyle='--', alpha=0.4)
ax3.axhline(y=BUY_THRESHOLD, color='#ff4444', linewidth=0.8, linestyle='--', alpha=0.4)
ax3.set_title('Vol Risk Premium (IV − RV30)  |  Green = Sell Vol Edge  |  Red = Buy Vol',
              fontsize=11, fontweight='bold', color='white')
ax3.set_ylabel('VRP (vol pts)', fontsize=9, color='#888')
ax3.tick_params(colors='#888', labelsize=8)
ax3.grid(alpha=0.1)
for spine in ax3.spines.values(): spine.set_color('#333')

# ── Panel 4: IV Rank gauge ──
ax4 = fig.add_subplot(gs[2, 0])
# Horizontal bar gauge
gauge_colors = ['#ff4444' if iv_rank < 25 else '#ffaa00' if iv_rank < 50 else '#00ff88' if iv_rank < 75 else '#00ffff']
ax4.barh([0], [iv_rank], height=0.5, color=gauge_colors[0], alpha=0.8)
ax4.barh([0], [100], height=0.5, color='#222', alpha=0.3)
ax4.set_xlim(0, 100)
ax4.set_yticks([])
ax4.set_title(f'IV Rank: {iv_rank:.0f}th  |  IV Percentile: {iv_percentile:.0f}th',
              fontsize=11, fontweight='bold', color='white')
# Add markers
ax4.axvline(x=25, color='#444', linewidth=0.5)
ax4.axvline(x=50, color='#444', linewidth=0.5)
ax4.axvline(x=75, color='#444', linewidth=0.5)
ax4.text(12.5, -0.7, 'LOW', ha='center', fontsize=8, color='#666')
ax4.text(37.5, -0.7, 'MID', ha='center', fontsize=8, color='#666')
ax4.text(62.5, -0.7, 'HIGH', ha='center', fontsize=8, color='#666')
ax4.text(87.5, -0.7, 'EXTREME', ha='center', fontsize=8, color='#666')
ax4.set_ylim(-1.2, 1)
ax4.tick_params(colors='#888', labelsize=8)
for spine in ax4.spines.values(): spine.set_color('#333')

# ── Panel 5: Signal summary card ──
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.set_xticks([])
ax5.set_yticks([])
for spine in ax5.spines.values(): spine.set_color('#333')

# Signal text
ax5.text(5, 8.5, signal, fontsize=16, fontweight='bold', color=signal_color,
         ha='center', va='center')
ax5.text(5, 7.0, signal_detail, fontsize=8, color='#888',
         ha='center', va='center', wrap=True,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', edgecolor='#333'))

# Key metrics
metrics = [
    f"ATM IV: {atm_iv:.1f}%",
    f"RV 30d: {rv_30:.1f}%  |  RV 60d: {rv_60:.1f}%",
    f"VRP: {vrp_composite:+.1f} pts  |  Edge: {edge_score:+.1f}",
    f"IV Rank: {iv_rank:.0f}  |  Skew: {skew:+.1f}",
    f"Expiry: {best_exp} ({dte} DTE)",
]
for i, m in enumerate(metrics):
    ax5.text(5, 5.0 - i * 1.0, m, fontsize=9, color='#ccc',
             ha='center', va='center', family='monospace')

ax5.set_title('EDGE SCANNER SIGNAL', fontsize=11, fontweight='bold', color='white')

plt.savefig('iv_rv_edge_scanner.png', dpi=150, facecolor='#0a0a0a',
            bbox_inches='tight', pad_inches=0.3)
plt.show()

# ═══════════════════════════════════════════════════════════════
# STEP 8: Poker Translation
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  ♠ POKER TRANSLATION")
print(f"{'='*60}")
print(f"""
  The market is pricing in {atm_iv:.1f}% annual volatility (IV).
  The stock has ACTUALLY been moving at {rv_30:.1f}% (30-day RV).

  That's a {vrp_30:+.1f} point gap — the market is {'OVER' if vrp_30 > 0 else 'UNDER'}paying
  for insurance by {abs(vrp_30):.1f} vol points.

  ♠ Poker analogy: {"The pot is offering you better than fair odds. You have a 55/45 edge — this is a clear call (sell vol)." if vrp_composite > SELL_THRESHOLD else "The pot odds are roughly fair. No clear edge — fold and wait for a better spot." if vrp_composite > BUY_THRESHOLD else "The pot odds are AGAINST you. The market is underpricing risk. Stay out or buy protection."}

  IV Rank at {iv_rank:.0f} means current IV is {"cheap (low in its range — bad time to sell)" if iv_rank < 25 else "moderate" if iv_rank < 50 else "elevated (good for selling)" if iv_rank < 75 else "extreme (best time to sell — max premium)"}.

  Signal: {signal}
""")

# ═══════════════════════════════════════════════════════════════
# STEP 9: Multi-Ticker Quick Scan (bonus)
# ═══════════════════════════════════════════════════════════════
print(f"{'='*60}")
print(f"  MULTI-TICKER QUICK SCAN")
print(f"{'='*60}")

scan_tickers = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
print(f"\n  {'Ticker':<8} {'RV 30d':>8} {'ATM IV':>8} {'VRP':>8} {'Signal':<20}")
print(f"  {'─'*56}")

for t in scan_tickers:
    try:
        tk = yf.Ticker(t)
        h = tk.history(period='6mo')
        if h.empty: continue
        h['lr'] = np.log(h['Close'] / h['Close'].shift(1))
        rv = h['lr'].rolling(30).std().iloc[-1] * np.sqrt(252) * 100

        # Get ATM IV
        exps = tk.options
        if exps:
            # Pick ~30 DTE
            exp = None
            for e in exps:
                d = (datetime.strptime(e, '%Y-%m-%d').date() - today).days
                if 14 <= d <= 60:
                    exp = e
                    break
            if exp is None: exp = exps[min(1, len(exps)-1)]
            ch = tk.option_chain(exp)
            cp = h['Close'].iloc[-1]
            ch.calls['dist'] = abs(ch.calls['strike'] - cp)
            atm = ch.calls.loc[ch.calls['dist'].idxmin()]
            iv = atm['impliedVolatility'] * 100
        else:
            iv = rv * 1.15  # rough estimate

        vrp = iv - rv
        if vrp > STRONG_SELL_THRESHOLD:
            sig = "🔥 STRONG SELL"
        elif vrp > SELL_THRESHOLD:
            sig = "✅ SELL VOL"
        elif vrp < BUY_THRESHOLD:
            sig = "🛒 BUY VOL"
        else:
            sig = "⏸ HOLD"

        print(f"  {t:<8} {rv:>7.1f}% {iv:>7.1f}% {vrp:>+7.1f}  {sig}")
    except Exception as e:
        print(f"  {t:<8} {'error':>8} — {str(e)[:30]}")

print(f"\n  Done. Run daily before market open for best results.")
print(f"  Next: Program 4 (Earnings Vol Crush Scanner)")
