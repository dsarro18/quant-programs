"""
PROGRAM 6: VOLATILITY SURFACE FITTER & RELATIVE VALUE ENGINE
==============================================================
Quant Mastery — MBA to Market Maker
Google Colab Ready | Libraries: yfinance, pandas, numpy, matplotlib, scipy

Builds a complete implied volatility surface from live options data:
- Fetches full options chains across all available expirations
- Fits the vol smile per expiration using SVI parameterization
- Builds 3D vol surface (strike x expiration x IV)
- Identifies relative value: which options are cheap/rich vs the surface
- Finds calendar spread and butterfly opportunities
- Full 3D visualization with interactive-quality static plots

To run in Google Colab:
  !pip install yfinance matplotlib scipy
  Then paste this entire file into a cell and run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.stats import norm
from datetime import datetime, timedelta
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
MAX_EXPIRATIONS = 8   # How many expirations to use
MIN_DTE = 5           # Skip very near-term
MAX_DTE = 180         # Skip very long-dated
MONEYNESS_RANGE = 0.12  # +/- 12% from ATM
MIN_VOLUME = 10       # Filter illiquid strikes
MIN_OI = 50           # Minimum open interest

print(f"\n{'='*70}")
print(f"  VOLATILITY SURFACE FITTER & RELATIVE VALUE ENGINE")
print(f"  {TICKER} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*70}\n")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: FETCH DATA
# ══════════════════════════════════════════════════════════════════════

print("[1/7] Fetching market data...")
tk = yf.Ticker(TICKER)
hist = tk.history(period="5d")
spot = hist['Close'].iloc[-1]
print(f"  {TICKER} spot: ${spot:.2f}")

print("[2/7] Loading options chains across expirations...")
expirations = tk.options
today = datetime.now()

# Filter expirations
valid_exps = []
for exp in expirations:
    dte = (datetime.strptime(exp, "%Y-%m-%d") - today).days
    if MIN_DTE <= dte <= MAX_DTE:
        valid_exps.append((exp, dte))

# Take evenly spaced expirations
if len(valid_exps) > MAX_EXPIRATIONS:
    step = len(valid_exps) // MAX_EXPIRATIONS
    valid_exps = valid_exps[::step][:MAX_EXPIRATIONS]

print(f"  Using {len(valid_exps)} expirations: {[e[0] for e in valid_exps]}")

# Fetch all chains
all_options = []
for exp_str, dte in valid_exps:
    try:
        chain = tk.option_chain(exp_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        # Filter strikes by moneyness
        lo = spot * (1 - MONEYNESS_RANGE)
        hi = spot * (1 + MONEYNESS_RANGE)

        for _, row in calls.iterrows():
            K = row['strike']
            if lo <= K <= hi and row.get('volume', 0) >= MIN_VOLUME and row.get('openInterest', 0) >= MIN_OI:
                moneyness = np.log(K / spot)
                all_options.append({
                    'expiration': exp_str,
                    'dte': dte,
                    'T': dte / 365.0,
                    'strike': K,
                    'moneyness': moneyness,
                    'type': 'call',
                    'iv': row['impliedVolatility'],
                    'last_price': row['lastPrice'],
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0),
                    'volume': row.get('volume', 0),
                    'oi': row.get('openInterest', 0),
                    'mid': (row.get('bid', 0) + row.get('ask', 0)) / 2,
                })

        for _, row in puts.iterrows():
            K = row['strike']
            if lo <= K <= hi and row.get('volume', 0) >= MIN_VOLUME and row.get('openInterest', 0) >= MIN_OI:
                moneyness = np.log(K / spot)
                all_options.append({
                    'expiration': exp_str,
                    'dte': dte,
                    'T': dte / 365.0,
                    'strike': K,
                    'moneyness': moneyness,
                    'type': 'put',
                    'iv': row['impliedVolatility'],
                    'last_price': row['lastPrice'],
                    'bid': row.get('bid', 0),
                    'ask': row.get('ask', 0),
                    'volume': row.get('volume', 0),
                    'oi': row.get('openInterest', 0),
                    'mid': (row.get('bid', 0) + row.get('ask', 0)) / 2,
                })
    except Exception as e:
        print(f"  Warning: Failed to fetch {exp_str}: {e}")

opts_df = pd.DataFrame(all_options)
print(f"  Total options loaded: {len(opts_df)} ({len(opts_df[opts_df['type']=='call'])} calls, {len(opts_df[opts_df['type']=='put'])} puts)")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: SVI PARAMETERIZATION
# ══════════════════════════════════════════════════════════════════════

print("\n[3/7] Fitting SVI volatility smile per expiration...")

def svi_total_variance(k, a, b, rho, m, sigma):
    """
    SVI (Stochastic Volatility Inspired) parameterization.
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    where w = sigma_BS^2 * T (total implied variance)
    k = log(K/F) (log-moneyness)
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def fit_svi(moneyness, total_var, T):
    """
    Fit SVI parameters to market data for one expiration.
    Returns (a, b, rho, m, sigma) and residual.
    """
    # Initial guess
    atm_var = np.median(total_var)
    x0 = [atm_var, 0.1, -0.3, 0.0, 0.1]

    # Bounds: a>0, b>0, -1<rho<1, m free, sigma>0
    bounds = [
        (0.001, 1.0),    # a
        (0.001, 2.0),    # b
        (-0.99, 0.99),   # rho
        (-0.5, 0.5),     # m
        (0.001, 1.0),    # sigma
    ]

    def objective(params):
        a, b, rho, m, sig = params
        pred = svi_total_variance(moneyness, a, b, rho, m, sig)
        return np.sum((pred - total_var)**2)

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun


svi_params = {}
fit_results = []

for exp_str, dte in valid_exps:
    exp_data = opts_df[opts_df['expiration'] == exp_str].copy()
    if len(exp_data) < 5:
        continue

    # Average IV across calls/puts at same strike
    avg_iv = exp_data.groupby('strike').agg({'iv': 'mean', 'moneyness': 'first'}).reset_index()
    T = dte / 365.0

    # Total variance = IV^2 * T
    k = avg_iv['moneyness'].values
    w = (avg_iv['iv'].values ** 2) * T

    params, residual = fit_svi(k, w, T)
    svi_params[exp_str] = {'params': params, 'T': T, 'dte': dte}

    # Compute fitted IV
    w_fitted = svi_total_variance(k, *params)
    iv_fitted = np.sqrt(w_fitted / T)

    # Store residuals
    for i, (_, row) in enumerate(avg_iv.iterrows()):
        if i < len(iv_fitted):
            fit_results.append({
                'expiration': exp_str,
                'dte': dte,
                'strike': row['strike'],
                'moneyness': row['moneyness'],
                'market_iv': row['iv'],
                'fitted_iv': iv_fitted[i],
                'residual': (row['iv'] - iv_fitted[i]) * 100,  # in vol pts
                'residual_pct': (row['iv'] / iv_fitted[i] - 1) * 100,
            })

    rmse = np.sqrt(residual / len(k)) * 100
    print(f"  {exp_str} ({dte:3d} DTE): {len(avg_iv):2d} strikes | SVI RMSE: {rmse:.2f} vol pts | rho={params[2]:.3f}")

fit_df = pd.DataFrame(fit_results)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: BUILD FULL SURFACE
# ══════════════════════════════════════════════════════════════════════

print("\n[4/7] Building volatility surface...")

# Create a grid for the surface
moneyness_grid = np.linspace(-MONEYNESS_RANGE, MONEYNESS_RANGE, 50)
dte_grid = np.linspace(MIN_DTE, MAX_DTE, 50)

surface = np.zeros((len(dte_grid), len(moneyness_grid)))

for i, dte_val in enumerate(dte_grid):
    T = dte_val / 365.0
    # Interpolate SVI params between available expirations
    dtes_available = sorted(svi_params.keys(), key=lambda x: svi_params[x]['dte'])

    # Find bracketing expirations
    dte_list = [(exp, svi_params[exp]['dte']) for exp in dtes_available]

    if len(dte_list) == 0:
        continue

    # Simple nearest-neighbor for now
    nearest = min(dte_list, key=lambda x: abs(x[1] - dte_val))
    params = svi_params[nearest[0]]['params']

    for j, k in enumerate(moneyness_grid):
        w = svi_total_variance(k, *params)
        if w > 0 and T > 0:
            surface[i, j] = np.sqrt(w / T) * 100  # IV in %
        else:
            surface[i, j] = np.nan

# Strike grid for display
strike_grid = spot * np.exp(moneyness_grid)

print(f"  Surface built: {surface.shape[0]} x {surface.shape[1]} grid")
print(f"  ATM term structure: {surface[:, 25].min():.1f}% to {surface[:, 25].max():.1f}%")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: RELATIVE VALUE — CHEAP/RICH OPTIONS
# ══════════════════════════════════════════════════════════════════════

print("\n[5/7] Scanning for relative value opportunities...")

if len(fit_df) > 0:
    # Rich options (market IV >> fitted IV): candidates to sell
    rich = fit_df[fit_df['residual'] > 0.5].sort_values('residual', ascending=False)
    # Cheap options (market IV << fitted IV): candidates to buy
    cheap = fit_df[fit_df['residual'] < -0.5].sort_values('residual', ascending=True)

    print(f"\n  TOP RICH OPTIONS (sell candidates — IV above surface):")
    print(f"  {'Strike':>8} {'Exp':>12} {'DTE':>5} {'Mkt IV':>8} {'Fit IV':>8} {'Rich':>8}")
    print(f"  {'─'*8} {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*8}")
    for _, row in rich.head(10).iterrows():
        print(f"  ${row['strike']:>7.0f} {row['expiration']:>12} {row['dte']:>5d} {row['market_iv']*100:>7.1f}% {row['fitted_iv']*100:>7.1f}% {row['residual']:>+7.2f}pp")

    print(f"\n  TOP CHEAP OPTIONS (buy candidates — IV below surface):")
    print(f"  {'Strike':>8} {'Exp':>12} {'DTE':>5} {'Mkt IV':>8} {'Fit IV':>8} {'Cheap':>8}")
    print(f"  {'─'*8} {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*8}")
    for _, row in cheap.head(10).iterrows():
        print(f"  ${row['strike']:>7.0f} {row['expiration']:>12} {row['dte']:>5d} {row['market_iv']*100:>7.1f}% {row['fitted_iv']*100:>7.1f}% {row['residual']:>+7.2f}pp")

    # Calendar spread opportunities
    print(f"\n  CALENDAR SPREAD OPPORTUNITIES:")
    print(f"  (Sell rich near-term, buy cheap far-term at same strike)")
    strikes_with_multi_exp = fit_df.groupby('strike').filter(lambda x: len(x) >= 2)
    if len(strikes_with_multi_exp) > 0:
        for strike, group in strikes_with_multi_exp.groupby('strike'):
            group = group.sort_values('dte')
            if len(group) >= 2:
                near = group.iloc[0]
                far = group.iloc[-1]
                spread = near['market_iv'] - far['market_iv']
                if abs(spread) * 100 > 1.0:  # >1 vol pt difference
                    direction = "SELL near / BUY far" if spread > 0 else "BUY near / SELL far"
                    print(f"    K=${strike:.0f}: Near({near['dte']}d) IV={near['market_iv']*100:.1f}% vs Far({far['dte']}d) IV={far['market_iv']*100:.1f}% | Spread: {spread*100:+.1f}pp | {direction}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: SKEW ANALYSIS
# ══════════════════════════════════════════════════════════════════════

print(f"\n[6/7] Skew analysis...")

for exp_str in list(svi_params.keys())[:4]:
    params = svi_params[exp_str]['params']
    T = svi_params[exp_str]['T']
    dte = svi_params[exp_str]['dte']

    # 25-delta put and call moneyness (approximate)
    k_25p = -0.05  # ~5% OTM put
    k_25c = 0.05   # ~5% OTM call
    k_atm = 0.0

    w_25p = svi_total_variance(k_25p, *params)
    w_25c = svi_total_variance(k_25c, *params)
    w_atm = svi_total_variance(k_atm, *params)

    iv_25p = np.sqrt(w_25p / T) * 100 if w_25p > 0 else 0
    iv_25c = np.sqrt(w_25c / T) * 100 if w_25c > 0 else 0
    iv_atm = np.sqrt(w_atm / T) * 100 if w_atm > 0 else 0

    skew = iv_25p - iv_25c
    smile = (iv_25p + iv_25c) / 2 - iv_atm

    print(f"  {exp_str} ({dte}d): ATM={iv_atm:.1f}% | 25dP={iv_25p:.1f}% | 25dC={iv_25c:.1f}% | Skew={skew:+.1f}pp | Smile={smile:+.1f}pp")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n[7/7] Rendering vol surface dashboard...\n")

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')

accent = '#58a6ff'
accent2 = '#f78166'
accent3 = '#7ee787'
accent4 = '#d2a8ff'

# Panel 1: 3D Volatility Surface (large)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('#161b22')

X, Y = np.meshgrid(strike_grid, dte_grid)
surf = ax1.plot_surface(X, Y, surface, cmap='RdYlBu_r', alpha=0.8,
                         edgecolor='none', antialiased=True)
ax1.set_xlabel('Strike ($)', fontsize=9, color='#8b949e', labelpad=10)
ax1.set_ylabel('DTE', fontsize=9, color='#8b949e', labelpad=10)
ax1.set_zlabel('IV (%)', fontsize=9, color='#8b949e', labelpad=10)
ax1.set_title(f'{TICKER} Implied Volatility Surface', fontsize=11, color='#e0e0e0', pad=15)
ax1.tick_params(colors='#8b949e', labelsize=7)
ax1.view_init(elev=25, azim=-60)

# Panel 2: Vol Smile per Expiration (2D)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#161b22')
ax2.set_title('Volatility Smile by Expiration', fontsize=11, color='#e0e0e0', pad=10)

colors_exp = [accent, accent2, accent3, accent4, '#e0e0e0', '#ff6b6b', '#48dbfb', '#feca57']
for idx, (exp_str, pdata) in enumerate(svi_params.items()):
    params = pdata['params']
    T = pdata['T']
    dte = pdata['dte']
    k_range = np.linspace(-MONEYNESS_RANGE, MONEYNESS_RANGE, 100)
    w_range = svi_total_variance(k_range, *params)
    iv_range = np.sqrt(np.maximum(w_range, 0) / T) * 100
    strike_range = spot * np.exp(k_range)
    color = colors_exp[idx % len(colors_exp)]
    ax2.plot(strike_range, iv_range, color=color, linewidth=1.5, label=f'{dte}d', alpha=0.85)

    # Plot market data points
    exp_data = opts_df[opts_df['expiration'] == exp_str]
    avg_by_strike = exp_data.groupby('strike')['iv'].mean()
    ax2.scatter(avg_by_strike.index, avg_by_strike.values * 100, color=color, s=15, alpha=0.6, zorder=5)

ax2.axvline(spot, color='white', linestyle='--', alpha=0.3, label=f'Spot ${spot:.0f}')
ax2.set_xlabel('Strike ($)', color='#8b949e', fontsize=9)
ax2.set_ylabel('Implied Volatility (%)', color='#8b949e', fontsize=9)
ax2.legend(fontsize=7, loc='upper right', ncol=2)
ax2.tick_params(colors='#8b949e', labelsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.15, color='#21262d')

# Panel 3: ATM Term Structure
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#161b22')
ax3.set_title('ATM IV Term Structure', fontsize=11, color='#e0e0e0', pad=10)

atm_term = []
for exp_str, pdata in svi_params.items():
    params = pdata['params']
    T = pdata['T']
    w_atm = svi_total_variance(0, *params)
    iv_atm = np.sqrt(w_atm / T) * 100 if w_atm > 0 else 0
    atm_term.append({'dte': pdata['dte'], 'iv': iv_atm, 'exp': exp_str})

atm_df = pd.DataFrame(atm_term).sort_values('dte')
ax3.plot(atm_df['dte'], atm_df['iv'], color=accent, linewidth=2.5, marker='D',
         markersize=8, markerfacecolor=accent2)
for _, row in atm_df.iterrows():
    ax3.annotate(f"{row['iv']:.1f}%", (row['dte'], row['iv']),
                 textcoords="offset points", xytext=(0, 12),
                 fontsize=8, color='#e0e0e0', ha='center')

ax3.set_xlabel('Days to Expiration', color='#8b949e', fontsize=9)
ax3.set_ylabel('ATM IV (%)', color='#8b949e', fontsize=9)
ax3.tick_params(colors='#8b949e', labelsize=8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, alpha=0.15, color='#21262d')

# Panel 4: Residuals (Rich/Cheap Map)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#161b22')
ax4.set_title('Relative Value: Rich(+) vs Cheap(-) vs Surface', fontsize=11, color='#e0e0e0', pad=10)

if len(fit_df) > 0:
    sc = ax4.scatter(fit_df['strike'], fit_df['dte'], c=fit_df['residual'],
                     cmap='RdYlGn_r', s=40, alpha=0.8, edgecolors='white', linewidth=0.3,
                     vmin=-2, vmax=2)
    plt.colorbar(sc, ax=ax4, label='Residual (vol pts)', shrink=0.8)
    ax4.axvline(spot, color='white', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Strike ($)', color='#8b949e', fontsize=9)
    ax4.set_ylabel('DTE', color='#8b949e', fontsize=9)
    ax4.tick_params(colors='#8b949e', labelsize=8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.15, color='#21262d')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle(f'{TICKER} Volatility Surface Analysis — {datetime.now().strftime("%Y-%m-%d")}\n'
             f'Spot: ${spot:.2f} | {len(opts_df)} options | {len(valid_exps)} expirations | SVI fit',
             fontsize=13, fontweight='bold', color='#e0e0e0', y=0.99)

plt.savefig('program6_vol_surface.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  VOLATILITY SURFACE SUMMARY")
print(f"{'='*70}")
print(f"  Ticker:     {TICKER} (${spot:.2f})")
print(f"  Options:    {len(opts_df)} contracts across {len(valid_exps)} expirations")
print(f"  Model:      SVI (Stochastic Volatility Inspired) parameterization")
if len(fit_df) > 0:
    print(f"  Fit RMSE:   {fit_df['residual'].abs().mean():.2f} vol pts (avg)")
    print(f"  Rich opts:  {len(fit_df[fit_df['residual'] > 0.5])} (IV > surface by >0.5pp)")
    print(f"  Cheap opts: {len(fit_df[fit_df['residual'] < -0.5])} (IV < surface by >0.5pp)")

print(f"\n  POKER TRANSLATION:")
print(f"  The vol surface is the house's price list for every bet.")
print(f"  SVI fitting finds the 'fair' price for each option.")
print(f"  Rich options = overpriced bets (sell them).")
print(f"  Cheap options = underpriced bets (buy them).")
print(f"  Calendar spreads exploit term structure kinks.")
print(f"  This is the options equivalent of finding a poker game")
print(f"  where the rake is wrong on specific bet sizes.")

print(f"\n  Dashboard saved to program6_vol_surface.png")
print(f"{'='*70}\n")
