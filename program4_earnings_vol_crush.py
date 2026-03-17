"""
PROGRAM 4: EARNINGS VOL CRUSH SCANNER
======================================
Quant Mastery — MBA to Market Maker
Google Colab Ready | Libraries: yfinance, pandas, numpy, matplotlib

Scans a universe of stocks for upcoming earnings, identifies which ones
have the highest IV crush potential (IV inflated pre-earnings), and
ranks them by expected theta harvest from selling vol into the event.

The Edge: IV spikes 30-80% before earnings, then collapses overnight.
Selling straddles/strangles into earnings captures this crush — but
only when the premium compensates for the gap risk. This scanner
finds the best risk/reward setups.

To run in Google Colab:
  !pip install yfinance matplotlib
  Then paste this entire file into a cell and run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

# Universe of liquid, optionable stocks with regular earnings
UNIVERSE = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    # Financials
    'JPM', 'GS', 'MS', 'BAC', 'C',
    # Consumer / Retail
    'WMT', 'COST', 'TGT', 'NKE', 'SBUX', 'MCD',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV', 'LLY',
    # Industrials / Energy
    'CAT', 'BA', 'XOM', 'CVX',
    # Semis
    'AMD', 'INTC', 'AVGO', 'MU',
    # Other high-IV names
    'CRM', 'SHOP', 'SQ', 'COIN', 'SNAP', 'ROKU', 'PLTR',
    # ETFs (for reference/hedging)
    'SPY', 'QQQ', 'IWM'
]

# Minimum IV rank to consider (we want elevated IV)
MIN_IVR_THRESHOLD = 40
# Minimum premium as % of stock price to make the trade worthwhile
MIN_PREMIUM_PCT = 1.5

print(f"\n{'='*70}")
print(f"  EARNINGS VOL CRUSH SCANNER")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Universe: {len(UNIVERSE)} tickers | Min IVR: {MIN_IVR_THRESHOLD}%")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def get_iv_rank(ticker_obj, current_iv, lookback=252):
    """
    IV Rank = where current IV sits in the past year's range.
    IVR = (current - min) / (max - min) * 100
    """
    try:
        hist = ticker_obj.history(period="1y")
        if len(hist) < 60:
            return None

        log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

        # Rolling 30-day realized vol as proxy for historical IV levels
        rolling_rv = log_ret.rolling(30).std() * np.sqrt(252)
        rolling_rv = rolling_rv.dropna()

        if len(rolling_rv) < 30:
            return None

        # Use RV range as proxy for IV range
        rv_min = rolling_rv.min()
        rv_max = rolling_rv.max()

        if rv_max - rv_min < 0.01:
            return 50.0

        ivr = (current_iv - rv_min) / (rv_max - rv_min) * 100
        return max(0, min(100, ivr))
    except:
        return None


def get_iv_percentile(ticker_obj, current_iv, lookback=252):
    """
    IV Percentile = % of days in the past year where IV was BELOW current.
    """
    try:
        hist = ticker_obj.history(period="1y")
        if len(hist) < 60:
            return None

        log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        rolling_rv = log_ret.rolling(30).std() * np.sqrt(252)
        rolling_rv = rolling_rv.dropna()

        if len(rolling_rv) < 30:
            return None

        pct = (rolling_rv < current_iv).sum() / len(rolling_rv) * 100
        return pct
    except:
        return None


def get_realized_vol(hist_df, window=30):
    """Calculate annualized realized volatility."""
    log_ret = np.log(hist_df['Close'] / hist_df['Close'].shift(1)).dropna()
    if len(log_ret) < window:
        return None
    rv = log_ret.tail(window).std() * np.sqrt(252)
    return rv


def get_expected_move(atm_iv, spot, dte):
    """Expected 1-sigma move from ATM straddle IV."""
    return spot * atm_iv * np.sqrt(dte / 365)


def get_historical_earnings_moves(ticker_str, n_quarters=8):
    """
    Estimate historical earnings-day moves from price gaps.
    Looks for the largest single-day absolute % moves (earnings proxies).
    """
    try:
        tk = yf.Ticker(ticker_str)
        hist = tk.history(period="2y")
        if len(hist) < 100:
            return None, None

        daily_ret = (hist['Close'] / hist['Close'].shift(1) - 1).abs().dropna()

        # Top N moves as earnings proxies (earnings days = biggest gap days)
        top_moves = daily_ret.nlargest(n_quarters)
        avg_move = top_moves.mean()
        max_move = top_moves.max()

        return avg_move, max_move
    except:
        return None, None


# ══════════════════════════════════════════════════════════════════════
# MAIN SCAN
# ══════════════════════════════════════════════════════════════════════

results = []
errors = []

for i, ticker in enumerate(UNIVERSE):
    pct = (i + 1) / len(UNIVERSE) * 100
    print(f"  [{i+1:2d}/{len(UNIVERSE)}] Scanning {ticker:6s}...", end="")

    try:
        tk = yf.Ticker(ticker)

        # Get spot price
        hist = tk.history(period="5d")
        if len(hist) == 0:
            print(" NO DATA")
            errors.append((ticker, "No price data"))
            continue
        spot = hist['Close'].iloc[-1]

        # Get options expirations
        exps = tk.options
        if len(exps) == 0:
            print(" NO OPTIONS")
            errors.append((ticker, "No options"))
            continue

        # Find nearest expiration (weekly — likely next earnings window)
        today = datetime.now()
        exp_dte = [(e, (datetime.strptime(e, "%Y-%m-%d") - today).days) for e in exps]
        exp_dte = [(e, d) for e, d in exp_dte if 2 <= d <= 45]  # 2-45 DTE range

        if not exp_dte:
            print(" NO VALID EXP")
            errors.append((ticker, "No expiration in range"))
            continue

        # Get front-month chain (~30 DTE preferred, fallback to nearest)
        target_exp = min(exp_dte, key=lambda x: abs(x[1] - 30))
        # Also get the nearest weekly (for earnings crush)
        nearest_exp = min(exp_dte, key=lambda x: x[1])

        exp_str, dte = target_exp
        near_exp_str, near_dte = nearest_exp

        chain = tk.option_chain(exp_str)
        calls = chain.calls
        puts = chain.puts

        if len(calls) == 0 or len(puts) == 0:
            print(" EMPTY CHAIN")
            errors.append((ticker, "Empty chain"))
            continue

        # Find ATM strike
        atm_call_idx = (calls['strike'] - spot).abs().idxmin()
        atm_put_idx = (puts['strike'] - spot).abs().idxmin()

        atm_call = calls.loc[atm_call_idx]
        atm_put = puts.loc[atm_put_idx]

        atm_iv_call = atm_call['impliedVolatility']
        atm_iv_put = atm_put['impliedVolatility']
        atm_iv = (atm_iv_call + atm_iv_put) / 2
        atm_strike = atm_call['strike']

        # Straddle premium
        straddle_price = atm_call['lastPrice'] + atm_put['lastPrice']
        straddle_pct = straddle_price / spot * 100

        # Expected move from straddle
        expected_move = get_expected_move(atm_iv, spot, dte)
        expected_move_pct = expected_move / spot * 100

        # IV Rank & Percentile
        ivr = get_iv_rank(tk, atm_iv)
        ivp = get_iv_percentile(tk, atm_iv)

        # Realized vol (30d)
        hist_1y = tk.history(period="1y")
        rv_30 = get_realized_vol(hist_1y, 30)

        # VRP
        vrp = (atm_iv - rv_30) * 100 if rv_30 else None

        # Historical earnings moves
        avg_earn_move, max_earn_move = get_historical_earnings_moves(ticker)

        # Crush potential: how much of the straddle do we expect to keep?
        # If expected earnings move < straddle price, the seller wins
        crush_ratio = None
        if avg_earn_move and straddle_pct > 0:
            crush_ratio = straddle_pct / (avg_earn_move * 100)
            # >1 means premium > avg move = edge for seller

        # Also get nearest-weekly data for true earnings crush
        near_chain = None
        near_straddle = None
        near_straddle_pct = None
        near_iv = None
        if near_exp_str != exp_str:
            try:
                near_chain = tk.option_chain(near_exp_str)
                nc = near_chain.calls
                np_ = near_chain.puts
                natm_c_idx = (nc['strike'] - spot).abs().idxmin()
                natm_p_idx = (np_['strike'] - spot).abs().idxmin()
                near_iv = (nc.loc[natm_c_idx, 'impliedVolatility'] + np_.loc[natm_p_idx, 'impliedVolatility']) / 2
                near_straddle = nc.loc[natm_c_idx, 'lastPrice'] + np_.loc[natm_p_idx, 'lastPrice']
                near_straddle_pct = near_straddle / spot * 100
            except:
                pass

        # Signal scoring
        score = 0
        signals = []

        if ivr is not None and ivr >= 70:
            score += 3
            signals.append(f"IVR {ivr:.0f}% (HIGH)")
        elif ivr is not None and ivr >= 50:
            score += 2
            signals.append(f"IVR {ivr:.0f}%")
        elif ivr is not None and ivr >= MIN_IVR_THRESHOLD:
            score += 1
            signals.append(f"IVR {ivr:.0f}% (moderate)")

        if vrp is not None and vrp > 5:
            score += 3
            signals.append(f"VRP +{vrp:.1f}pp (FAT)")
        elif vrp is not None and vrp > 2:
            score += 2
            signals.append(f"VRP +{vrp:.1f}pp")

        if crush_ratio is not None and crush_ratio > 1.5:
            score += 3
            signals.append(f"Crush {crush_ratio:.1f}x (STRONG)")
        elif crush_ratio is not None and crush_ratio > 1.0:
            score += 2
            signals.append(f"Crush {crush_ratio:.1f}x")

        if straddle_pct >= 5.0:
            score += 2
            signals.append(f"Premium {straddle_pct:.1f}% (RICH)")
        elif straddle_pct >= MIN_PREMIUM_PCT:
            score += 1
            signals.append(f"Premium {straddle_pct:.1f}%")

        # Determine signal
        if score >= 8:
            signal = "STRONG SELL VOL"
            signal_icon = "🔥"
        elif score >= 5:
            signal = "SELL VOL"
            signal_icon = "✅"
        elif score >= 3:
            signal = "WATCH"
            signal_icon = "👀"
        else:
            signal = "PASS"
            signal_icon = "⏸️"

        results.append({
            'ticker': ticker,
            'spot': spot,
            'atm_strike': atm_strike,
            'expiration': exp_str,
            'dte': dte,
            'atm_iv': atm_iv * 100,
            'rv_30': rv_30 * 100 if rv_30 else None,
            'vrp': vrp,
            'ivr': ivr,
            'ivp': ivp,
            'straddle': straddle_price,
            'straddle_pct': straddle_pct,
            'expected_move_pct': expected_move_pct,
            'avg_earn_move_pct': avg_earn_move * 100 if avg_earn_move else None,
            'max_earn_move_pct': max_earn_move * 100 if max_earn_move else None,
            'crush_ratio': crush_ratio,
            'score': score,
            'signal': signal,
            'signal_icon': signal_icon,
            'signals': signals,
            'near_exp': near_exp_str if near_exp_str != exp_str else None,
            'near_dte': near_dte if near_exp_str != exp_str else None,
            'near_iv': near_iv * 100 if near_iv else None,
            'near_straddle_pct': near_straddle_pct,
        })

        print(f" ${spot:>8.2f} | IV:{atm_iv*100:5.1f}% | IVR:{ivr:5.1f}% | {signal_icon} {signal}" if ivr else f" ${spot:>8.2f} | IV:{atm_iv*100:5.1f}% | {signal_icon} {signal}")

    except Exception as e:
        print(f" ERROR: {e}")
        errors.append((ticker, str(e)))

# ══════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════

df = pd.DataFrame(results)
df = df.sort_values('score', ascending=False).reset_index(drop=True)

print(f"\n{'='*70}")
print(f"  EARNINGS VOL CRUSH RANKINGS")
print(f"  {len(df)} tickers scanned | {len(errors)} errors")
print(f"{'='*70}\n")

# Top opportunities
print(f"  {'Rank':<5} {'Ticker':<7} {'Spot':>8} {'IV':>6} {'RV30':>6} {'VRP':>6} {'IVR':>5} {'Strad%':>7} {'Crush':>6} {'Score':>6} {'Signal':<18}")
print(f"  {'─'*5} {'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*7} {'─'*6} {'─'*6} {'─'*18}")

for i, row in df.iterrows():
    rv_str = f"{row['rv_30']:5.1f}%" if row['rv_30'] else "  N/A"
    vrp_str = f"{row['vrp']:+5.1f}" if row['vrp'] else "  N/A"
    ivr_str = f"{row['ivr']:4.0f}%" if row['ivr'] else " N/A"
    crush_str = f"{row['crush_ratio']:5.1f}x" if row['crush_ratio'] else "  N/A"

    print(f"  {i+1:<5} {row['ticker']:<7} ${row['spot']:>7.2f} {row['atm_iv']:5.1f}% {rv_str} {vrp_str} {ivr_str} {row['straddle_pct']:6.1f}% {crush_str} {row['score']:5d} {row['signal_icon']} {row['signal']}")

# ══════════════════════════════════════════════════════════════════════
# DETAILED TOP 5 ANALYSIS
# ══════════════════════════════════════════════════════════════════════

top_n = min(5, len(df))
print(f"\n{'='*70}")
print(f"  TOP {top_n} DETAILED ANALYSIS")
print(f"{'='*70}")

for i in range(top_n):
    row = df.iloc[i]
    print(f"\n  {'─'*60}")
    print(f"  #{i+1} {row['signal_icon']} {row['ticker']} — {row['signal']} (Score: {row['score']})")
    print(f"  {'─'*60}")
    print(f"  Price:          ${row['spot']:.2f}")
    print(f"  Expiration:     {row['expiration']} ({row['dte']} DTE)")
    print(f"  ATM Strike:     ${row['atm_strike']:.0f}")
    print(f"  ATM IV:         {row['atm_iv']:.1f}%")
    if row['rv_30']:
        print(f"  RV (30d):       {row['rv_30']:.1f}%")
    if row['vrp']:
        print(f"  VRP:            {row['vrp']:+.1f} vol pts")
    if row['ivr']:
        print(f"  IV Rank:        {row['ivr']:.0f}%")
    if row['ivp']:
        print(f"  IV Percentile:  {row['ivp']:.0f}%")
    print(f"  Straddle Price: ${row['straddle']:.2f} ({row['straddle_pct']:.1f}% of stock)")
    print(f"  Expected Move:  +/-{row['expected_move_pct']:.1f}%")
    if row['avg_earn_move_pct']:
        print(f"  Avg Earnings Move:  {row['avg_earn_move_pct']:.1f}%")
        print(f"  Max Earnings Move:  {row['max_earn_move_pct']:.1f}%")
    if row['crush_ratio']:
        print(f"  Crush Ratio:    {row['crush_ratio']:.2f}x", end="")
        if row['crush_ratio'] > 1.5:
            print(f" — Premium is {row['crush_ratio']:.1f}x the avg move. STRONG edge for sellers.")
        elif row['crush_ratio'] > 1.0:
            print(f" — Premium exceeds avg move. Positive EV to sell.")
        else:
            print(f" — Premium < avg move. CAUTION: gap risk > premium.")
    if row['near_exp']:
        print(f"\n  Nearest Weekly: {row['near_exp']} ({row['near_dte']} DTE)")
        if row['near_iv']:
            print(f"  Near-Term IV:   {row['near_iv']:.1f}% (vs {row['atm_iv']:.1f}% monthly)")
        if row['near_straddle_pct']:
            print(f"  Near Straddle:  {row['near_straddle_pct']:.1f}% of stock")

    print(f"\n  Signals: {' | '.join(row['signals'])}")

    # Trade suggestion
    if row['score'] >= 5:
        print(f"\n  TRADE IDEA: Sell ${row['atm_strike']:.0f} straddle @ {row['expiration']}")
        print(f"  Collect: ~${row['straddle']:.2f} per contract (${row['straddle']*100:.0f} notional)")
        print(f"  Breakeven: ${row['atm_strike'] - row['straddle']:.2f} / ${row['atm_strike'] + row['straddle']:.2f}")
        max_loss_pct = 2 * row['straddle'] / row['spot'] * 100
        print(f"  Max risk guidance: Close if loss > ${row['straddle']*2:.2f} ({max_loss_pct:.1f}% of stock)")

# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════

print(f"\n[CHART] Rendering vol crush dashboard...\n")

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'Earnings Vol Crush Scanner — {datetime.now().strftime("%Y-%m-%d")}\n'
             f'{len(df)} tickers | Top signal: {df.iloc[0]["ticker"]} ({df.iloc[0]["signal"]})',
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

# ── Panel 1: Score Ranking ───────────────────────────────────────────
ax1 = axes[0, 0]
ax1.set_title('Crush Score Ranking', fontsize=11, color='#e0e0e0', pad=10)

top_15 = df.head(15)
colors = [accent2 if s >= 8 else accent if s >= 5 else accent4 if s >= 3 else '#8b949e'
          for s in top_15['score']]
bars = ax1.barh(range(len(top_15)-1, -1, -1), top_15['score'], color=colors, alpha=0.85)
ax1.set_yticks(range(len(top_15)-1, -1, -1))
ax1.set_yticklabels(top_15['ticker'], fontsize=9)
ax1.set_xlabel('Crush Score', color='#8b949e', fontsize=9)
for j, (score, ticker) in enumerate(zip(top_15['score'], top_15['ticker'])):
    ax1.text(score + 0.1, len(top_15) - 1 - j, str(score), va='center',
             fontsize=8, color='#e0e0e0')

# ── Panel 2: IV vs RV Scatter ───────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_title('IV vs Realized Vol (30d)', fontsize=11, color='#e0e0e0', pad=10)

has_rv = df.dropna(subset=['rv_30'])
sc = ax2.scatter(has_rv['rv_30'], has_rv['atm_iv'], c=has_rv['score'],
                 cmap='RdYlGn', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
# Diagonal (IV = RV line)
max_val = max(has_rv['atm_iv'].max(), has_rv['rv_30'].max()) if len(has_rv) > 0 else 50
ax2.plot([0, max_val*1.1], [0, max_val*1.1], '--', color='#8b949e', alpha=0.5, label='IV = RV')
ax2.fill_between([0, max_val*1.1], [0, max_val*1.1], [max_val*1.5, max_val*1.5],
                 alpha=0.05, color=accent2, label='IV > RV (sell zone)')
for _, r in has_rv.head(8).iterrows():
    ax2.annotate(r['ticker'], (r['rv_30'], r['atm_iv']),
                 fontsize=7, color='#e0e0e0', xytext=(5, 5),
                 textcoords='offset points')
ax2.set_xlabel('Realized Vol 30d (%)', color='#8b949e', fontsize=9)
ax2.set_ylabel('ATM Implied Vol (%)', color='#8b949e', fontsize=9)
ax2.legend(fontsize=8, loc='upper left')
plt.colorbar(sc, ax=ax2, label='Score', shrink=0.8)

# ── Panel 3: Straddle Premium % ─────────────────────────────────────
ax3 = axes[1, 0]
ax3.set_title('Straddle Premium (% of Stock Price)', fontsize=11, color='#e0e0e0', pad=10)

sorted_by_prem = df.nlargest(15, 'straddle_pct')
colors3 = [accent2 if p >= 5 else accent if p >= 3 else '#8b949e'
           for p in sorted_by_prem['straddle_pct']]
ax3.barh(range(len(sorted_by_prem)-1, -1, -1), sorted_by_prem['straddle_pct'],
         color=colors3, alpha=0.85)
ax3.set_yticks(range(len(sorted_by_prem)-1, -1, -1))
ax3.set_yticklabels(sorted_by_prem['ticker'], fontsize=9)
ax3.set_xlabel('Straddle Premium (% of stock)', color='#8b949e', fontsize=9)
ax3.axvline(MIN_PREMIUM_PCT, color=accent3, linestyle='--', alpha=0.5, label=f'Min threshold ({MIN_PREMIUM_PCT}%)')
ax3.legend(fontsize=8)

# ── Panel 4: Crush Ratio ────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.set_title('Crush Ratio (Straddle / Avg Earnings Move)', fontsize=11, color='#e0e0e0', pad=10)

has_crush = df.dropna(subset=['crush_ratio']).nlargest(15, 'crush_ratio')
if len(has_crush) > 0:
    colors4 = [accent3 if c >= 1.5 else accent if c >= 1.0 else accent2
               for c in has_crush['crush_ratio']]
    ax4.barh(range(len(has_crush)-1, -1, -1), has_crush['crush_ratio'],
             color=colors4, alpha=0.85)
    ax4.set_yticks(range(len(has_crush)-1, -1, -1))
    ax4.set_yticklabels(has_crush['ticker'], fontsize=9)
    ax4.axvline(1.0, color=accent2, linestyle='--', alpha=0.7, label='Breakeven (1.0x)')
    ax4.axvline(1.5, color=accent3, linestyle='--', alpha=0.5, label='Strong edge (1.5x)')
    ax4.set_xlabel('Crush Ratio (premium / avg move)', color='#8b949e', fontsize=9)
    ax4.legend(fontsize=8)
else:
    ax4.text(0.5, 0.5, 'No crush ratio data', ha='center', va='center',
             transform=ax4.transAxes, color='#8b949e')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('program4_crush_scanner.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# POKER TRANSLATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  POKER TRANSLATION — EARNINGS VOL CRUSH")
print(f"{'='*70}")

print(f"""
  The Setup:
    Earnings announcements are like all-in pots in poker tournaments.
    The pot (premium) gets huge before the hand is played (earnings).
    After the hand resolves, the pot goes to one player (IV collapses).

  The Edge:
    Market OVERPRICES the uncertainty before earnings (fear premium).
    On average, the straddle costs MORE than the actual earnings move.
    This is like the casino's edge: they overprice the risk on every bet.

  The Crush Ratio:
    Crush Ratio = Premium Collected / Average Earnings Move
    > 1.5x = Strong edge (like getting 3:2 on a coin flip)
    > 1.0x = Positive EV (like getting 1.1:1 on a coin flip)
    < 1.0x = Negative EV (you're the fish — the gap will eat you)

  The Risk:
    Tail risk is real. One 15% earnings gap wipes 3-4 winning trades.
    Size accordingly: never risk > 2% of account on one earnings play.
    The edge is in the LAW OF LARGE NUMBERS — many small bets, not one big one.

  Today's Best Setup:
    {df.iloc[0]['ticker']} — Score {df.iloc[0]['score']}, {df.iloc[0]['signal']}
    Straddle: {df.iloc[0]['straddle_pct']:.1f}% of stock price
    {' | '.join(df.iloc[0]['signals'])}
""")

print(f"  Dashboard saved to program4_crush_scanner.png")
print(f"{'='*70}\n")
