"""
PROGRAM 1: OPTIONS INTELLIGENCE DASHBOARD
==========================================
Quant Mastery — MBA to Market Maker
Google Colab Ready | Libraries: yfinance, pandas, numpy, matplotlib

Pulls live SPY options chain via yfinance and builds a 4-panel dark
dashboard: vol skew, open interest, term structure, volume by strike.
Includes intel summary with ATM IV, expected 1-sigma move, put/call
ratio, and poker-translated output.

To run in Google Colab:
  !pip install yfinance matplotlib
  Then paste this entire file into a cell and run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

# ── Install yfinance if needed (Colab) ──────────────────────────────
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
print(f"\n{'='*60}")
print(f"  OPTIONS INTELLIGENCE DASHBOARD — {TICKER}")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}\n")

# ── Step 1: Fetch ticker data ────────────────────────────────────────
print("[1/6] Fetching ticker data...")
spy = yf.Ticker(TICKER)
hist = spy.history(period="5d")
spot = hist['Close'].iloc[-1]
print(f"  Current {TICKER} price: ${spot:.2f}")

# ── Step 2: Get available expiration dates ───────────────────────────
print("[2/6] Loading options expirations...")
expirations = spy.options
print(f"  Found {len(expirations)} expiration dates")

# Pick expirations: nearest weekly, ~30d, ~60d, ~90d
today = datetime.now()
exp_dates_dt = [datetime.strptime(e, "%Y-%m-%d") for e in expirations]
exp_dte = [(e, (e - today).days) for e in exp_dates_dt if (e - today).days > 0]

def find_nearest_dte(target_dte, exp_list):
    """Find expiration nearest to target DTE."""
    return min(exp_list, key=lambda x: abs(x[1] - target_dte))

targets = [7, 30, 45, 60, 90]
selected_exps = {}
for t in targets:
    nearest = find_nearest_dte(t, exp_dte)
    key = nearest[0].strftime("%Y-%m-%d")
    selected_exps[key] = nearest[1]

print(f"  Selected expirations: {list(selected_exps.keys())}")

# ── Step 3: Fetch options chains ─────────────────────────────────────
print("[3/6] Fetching options chains...")

# Use the ~30d expiration as the primary chain
primary_exp = find_nearest_dte(30, exp_dte)[0].strftime("%Y-%m-%d")
chain = spy.option_chain(primary_exp)
calls = chain.calls.copy()
puts = chain.puts.copy()

# Filter to reasonable strike range (spot ± 10%)
strike_lo = spot * 0.90
strike_hi = spot * 1.10
calls = calls[(calls['strike'] >= strike_lo) & (calls['strike'] <= strike_hi)].copy()
puts = puts[(puts['strike'] >= strike_lo) & (puts['strike'] <= strike_hi)].copy()

print(f"  Primary expiration: {primary_exp} ({selected_exps[primary_exp]} DTE)")
print(f"  Calls: {len(calls)} strikes | Puts: {len(puts)} strikes")

# ── Step 4: Calculate key metrics ────────────────────────────────────
print("[4/6] Calculating intelligence metrics...")

# ATM IV: find strike closest to spot
atm_idx_call = (calls['strike'] - spot).abs().idxmin()
atm_idx_put = (puts['strike'] - spot).abs().idxmin()
atm_iv_call = calls.loc[atm_idx_call, 'impliedVolatility']
atm_iv_put = puts.loc[atm_idx_put, 'impliedVolatility']
atm_iv = (atm_iv_call + atm_iv_put) / 2
atm_strike = calls.loc[atm_idx_call, 'strike']

# Expected 1-sigma move
dte = selected_exps[primary_exp]
expected_move_pct = atm_iv * np.sqrt(dte / 365)
expected_move_dollar = spot * expected_move_pct

# Put/Call volume ratio
total_call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
total_put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0

# Put/Call OI ratio
total_call_oi = calls['openInterest'].sum()
total_put_oi = puts['openInterest'].sum()
pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0

# Max pain (strike where total OI value is minimized)
all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
pain = []
for k in all_strikes:
    call_pain = calls[calls['strike'] < k]['openInterest'].sum() * (k - calls[calls['strike'] < k]['strike']).sum() if len(calls[calls['strike'] < k]) > 0 else 0
    put_pain = puts[puts['strike'] > k]['openInterest'].sum() * (puts[puts['strike'] > k]['strike'] - k).sum() if len(puts[puts['strike'] > k]) > 0 else 0
    pain.append((k, call_pain + put_pain))
max_pain_strike = min(pain, key=lambda x: x[1])[0] if pain else spot

# Highest OI strikes
top_call_oi = calls.nlargest(3, 'openInterest')[['strike', 'openInterest']]
top_put_oi = puts.nlargest(3, 'openInterest')[['strike', 'openInterest']]

# Vol skew: 25-delta put IV vs 25-delta call IV (approximate with OTM strikes)
otm_puts = puts[puts['strike'] < spot].copy()
otm_calls = calls[calls['strike'] > spot].copy()

print(f"  ATM IV: {atm_iv*100:.1f}%")
print(f"  Expected 1-sigma move: +/-${expected_move_dollar:.2f} ({expected_move_pct*100:.1f}%)")
print(f"  Put/Call Volume Ratio: {pc_ratio:.2f}")
print(f"  Max Pain: ${max_pain_strike:.0f}")

# ── Step 5: Build term structure ─────────────────────────────────────
print("[5/6] Building IV term structure...")

term_structure = []
for exp_str, dte_val in sorted(selected_exps.items(), key=lambda x: x[1]):
    try:
        ch = spy.option_chain(exp_str)
        c = ch.calls
        p = ch.puts
        # ATM IV for this expiration
        atm_c_idx = (c['strike'] - spot).abs().idxmin()
        atm_p_idx = (p['strike'] - spot).abs().idxmin()
        iv_c = c.loc[atm_c_idx, 'impliedVolatility']
        iv_p = p.loc[atm_p_idx, 'impliedVolatility']
        term_structure.append({
            'expiration': exp_str,
            'dte': dte_val,
            'atm_iv': (iv_c + iv_p) / 2
        })
    except Exception as e:
        print(f"  Warning: Could not fetch {exp_str}: {e}")

term_df = pd.DataFrame(term_structure)
print(f"  Term structure: {len(term_df)} points")

# ── Step 6: Build 4-Panel Dashboard ─────────────────────────────────
print("[6/6] Rendering dashboard...\n")

# Dark theme
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'{TICKER} Options Intelligence Dashboard\n{datetime.now().strftime("%Y-%m-%d")} | Spot: ${spot:.2f} | ATM IV: {atm_iv*100:.1f}%',
             fontsize=14, fontweight='bold', color='#e0e0e0', y=0.98)
fig.patch.set_facecolor('#0d1117')

accent = '#58a6ff'
accent2 = '#f78166'
accent3 = '#7ee787'
grid_color = '#21262d'

for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color=grid_color)

# ── Panel 1: Volatility Skew ────────────────────────────────────────
ax1 = axes[0, 0]
ax1.set_title('Volatility Skew (Smile)', fontsize=11, color='#e0e0e0', pad=10)

# Merge calls and puts IV by strike
call_iv = calls[['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'call_iv'})
put_iv = puts[['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'put_iv'})
skew_df = pd.merge(call_iv, put_iv, on='strike', how='outer').sort_values('strike')

ax1.plot(skew_df['strike'], skew_df['call_iv'] * 100, color=accent, linewidth=2, label='Call IV', marker='o', markersize=3)
ax1.plot(skew_df['strike'], skew_df['put_iv'] * 100, color=accent2, linewidth=2, label='Put IV', marker='o', markersize=3)
ax1.axvline(spot, color=accent3, linestyle='--', alpha=0.7, label=f'Spot ${spot:.0f}')
ax1.set_xlabel('Strike', color='#8b949e', fontsize=9)
ax1.set_ylabel('Implied Volatility (%)', color='#8b949e', fontsize=9)
ax1.legend(fontsize=8, loc='upper right')

# ── Panel 2: Open Interest ──────────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_title('Open Interest by Strike', fontsize=11, color='#e0e0e0', pad=10)

width = (calls['strike'].diff().median() or 1) * 0.35
ax2.bar(calls['strike'] - width/2, calls['openInterest'], width=width,
        color=accent, alpha=0.8, label='Call OI')
ax2.bar(puts['strike'] + width/2, puts['openInterest'], width=width,
        color=accent2, alpha=0.8, label='Put OI')
ax2.axvline(spot, color=accent3, linestyle='--', alpha=0.7, label=f'Spot ${spot:.0f}')
ax2.axvline(max_pain_strike, color='#d2a8ff', linestyle=':', alpha=0.7, label=f'Max Pain ${max_pain_strike:.0f}')
ax2.set_xlabel('Strike', color='#8b949e', fontsize=9)
ax2.set_ylabel('Open Interest', color='#8b949e', fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
ax2.legend(fontsize=8, loc='upper right')

# ── Panel 3: IV Term Structure ───────────────────────────────────────
ax3 = axes[1, 0]
ax3.set_title('IV Term Structure (ATM)', fontsize=11, color='#e0e0e0', pad=10)

if len(term_df) > 0:
    ax3.plot(term_df['dte'], term_df['atm_iv'] * 100, color=accent, linewidth=2.5,
             marker='D', markersize=8, markerfacecolor=accent2, markeredgecolor=accent2)
    for _, row in term_df.iterrows():
        ax3.annotate(f"{row['atm_iv']*100:.1f}%",
                     (row['dte'], row['atm_iv']*100),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=8, color='#e0e0e0', ha='center')
    ax3.set_xlabel('Days to Expiration', color='#8b949e', fontsize=9)
    ax3.set_ylabel('ATM IV (%)', color='#8b949e', fontsize=9)
    ax3.set_xlim(0, max(term_df['dte']) + 10)

# ── Panel 4: Volume by Strike ───────────────────────────────────────
ax4 = axes[1, 1]
ax4.set_title('Volume by Strike (Today)', fontsize=11, color='#e0e0e0', pad=10)

call_vol = calls[calls['volume'] > 0].copy() if 'volume' in calls.columns else pd.DataFrame()
put_vol = puts[puts['volume'] > 0].copy() if 'volume' in puts.columns else pd.DataFrame()

if len(call_vol) > 0:
    ax4.bar(call_vol['strike'] - width/2, call_vol['volume'], width=width,
            color=accent, alpha=0.8, label='Call Vol')
if len(put_vol) > 0:
    ax4.bar(put_vol['strike'] + width/2, put_vol['volume'], width=width,
            color=accent2, alpha=0.8, label='Put Vol')
ax4.axvline(spot, color=accent3, linestyle='--', alpha=0.7)
ax4.set_xlabel('Strike', color='#8b949e', fontsize=9)
ax4.set_ylabel('Volume', color='#8b949e', fontsize=9)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
ax4.legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('program1_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# INTEL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  INTELLIGENCE SUMMARY")
print(f"{'='*60}")
print(f"  Ticker:              {TICKER}")
print(f"  Spot Price:          ${spot:.2f}")
print(f"  Primary Expiration:  {primary_exp} ({dte} DTE)")
print(f"  ATM Strike:          ${atm_strike:.0f}")
print(f"  ATM IV:              {atm_iv*100:.1f}%")
print(f"  Expected 1-sigma Move:  +/-${expected_move_dollar:.2f} ({expected_move_pct*100:.1f}%)")
print(f"  Max Pain Strike:     ${max_pain_strike:.0f}")
print(f"  Put/Call Vol Ratio:  {pc_ratio:.2f}")
print(f"  Put/Call OI Ratio:   {pc_oi_ratio:.2f}")
print(f"\n  Top Call OI Strikes:")
for _, row in top_call_oi.iterrows():
    print(f"    ${row['strike']:.0f}  →  {row['openInterest']:,.0f} contracts")
print(f"\n  Top Put OI Strikes:")
for _, row in top_put_oi.iterrows():
    print(f"    ${row['strike']:.0f}  →  {row['openInterest']:,.0f} contracts")

# ── Term Structure ───────────────────────────────────────────────────
print(f"\n  IV Term Structure:")
for _, row in term_df.iterrows():
    bar = '#' * int(row['atm_iv'] * 200)
    print(f"    {row['dte']:3d} DTE  {row['atm_iv']*100:5.1f}%  {bar}")

# ══════════════════════════════════════════════════════════════════════
# POKER TRANSLATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  POKER TRANSLATION")
print(f"{'='*60}")

# P/C Ratio interpretation
if pc_ratio > 1.2:
    pc_read = "FEAR HEAVY — Puts dominating. Market is buying insurance. Like everyone folding to a 3-bet: scared money."
elif pc_ratio > 0.8:
    pc_read = "BALANCED — Neither side dominating. Standard action, no edge from sentiment alone."
else:
    pc_read = "GREED HEAVY — Calls dominating. Market is chasing upside. Like loose-passive table: lots of callers, few raisers."

# IV level interpretation
if atm_iv > 0.30:
    iv_read = "HIGH STAKES TABLE — IV above 30% means fat premiums. Like a high-rake game: sellers collect more, but swings are bigger."
elif atm_iv > 0.18:
    iv_read = "STANDARD GAME — IV in normal range. Edges come from precision, not from premium size."
else:
    iv_read = "TIGHT TABLE — Low IV means cheap options. Like a nit-fest: hard to extract value selling premium."

# OI concentration
print(f"\n  Market Sentiment (P/C = {pc_ratio:.2f}):")
print(f"    {pc_read}")
print(f"\n  Volatility Read (IV = {atm_iv*100:.1f}%):")
print(f"    {iv_read}")
print(f"\n  Key Levels (from OI):")
print(f"    Resistance ceiling: ${top_call_oi.iloc[0]['strike']:.0f} (highest call OI = gamma wall)")
print(f"    Support floor:      ${top_put_oi.iloc[0]['strike']:.0f} (highest put OI = downside magnet)")
print(f"    Max Pain:           ${max_pain_strike:.0f} (where most options expire worthless)")
print(f"\n  Translation:")
print(f"    The market expects {TICKER} to stay in [{spot - expected_move_dollar:.0f}, {spot + expected_move_dollar:.0f}]")
print(f"    over the next {dte} days (68% confidence, 1-sigma).")
print(f"    OI walls at ${top_put_oi.iloc[0]['strike']:.0f} (support) and ${top_call_oi.iloc[0]['strike']:.0f} (resistance).")
print(f"    Max pain at ${max_pain_strike:.0f} — dealers want price pinned here at expiry.")

print(f"\n{'='*60}")
print(f"  Dashboard saved to program1_dashboard.png")
print(f"{'='*60}\n")
