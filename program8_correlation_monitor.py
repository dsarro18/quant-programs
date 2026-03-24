# =============================================================================
# PROGRAM 8: Correlation & Dispersion Monitor
# =============================================================================
# Description:
#   Tracks SPY vs its top 10 holdings. Computes implied correlation (from index
#   IV vs constituent IV) and realized correlation from rolling returns. Flags
#   dispersion trade opportunities when implied correlation significantly
#   exceeds realized correlation — the classic "sell index vol, buy single-stock
#   vol" setup used by institutional desks.
#
# What it produces:
#   - 10x10 realized correlation heatmap (30-day rolling)
#   - Implied vs realized correlation time series (rolling window)
#   - Dispersion opportunity signal gauge
#   - Sector breakdown and regime classification
#   - Console summary of top dispersion opportunities
#
# Platform: Google Colab  |  Runtime: ~45-90 seconds
#
# Install (run first in Colab):
#   !pip install yfinance numpy scipy matplotlib pandas seaborn
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, date, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

INDEX_TICKER = "SPY"

HOLDINGS = {
    "AAPL":  {"name": "Apple",       "sector": "Technology",       "weight": 0.070},
    "MSFT":  {"name": "Microsoft",   "sector": "Technology",       "weight": 0.065},
    "NVDA":  {"name": "NVIDIA",      "sector": "Technology",       "weight": 0.058},
    "AMZN":  {"name": "Amazon",      "sector": "Consumer Disc.",   "weight": 0.040},
    "GOOGL": {"name": "Alphabet",    "sector": "Communication",    "weight": 0.037},
    "META":  {"name": "Meta",        "sector": "Communication",    "weight": 0.027},
    "TSLA":  {"name": "Tesla",       "sector": "Consumer Disc.",   "weight": 0.023},
    "BRK-B": {"name": "Berkshire",   "sector": "Financials",       "weight": 0.020},
    "JPM":   {"name": "JPMorgan",    "sector": "Financials",       "weight": 0.018},
    "V":     {"name": "Visa",        "sector": "Financials",       "weight": 0.016},
}

TICKERS = [INDEX_TICKER] + list(HOLDINGS.keys())
LOOKBACK_DAYS   = 252     # 1 year of price history
ROLLING_WINDOWS = [21, 63, 126]  # ~1mo, 3mo, 6mo in trading days
NEAR_TERM_EXPIRY_DAYS = 30       # for IV extraction from options chain

DISPERSION_THRESHOLD = 0.08   # implied_corr - realized_corr > this → signal

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_prices(tickers, lookback_days=252):
    """Download adjusted close prices for all tickers."""
    end   = date.today()
    start = end - timedelta(days=lookback_days + 60)   # buffer for weekends
    print(f"  Downloading price history ({start} → {end}) for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)['Close']
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw.dropna(how='all', inplace=True)
    # Keep only last lookback_days rows
    raw = raw.tail(lookback_days)
    return raw


def fetch_atm_iv(ticker, days_to_expiry=30):
    """
    Pull near-ATM implied volatility from the options chain closest to
    days_to_expiry. Returns scalar IV (annualized).
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return np.nan

        today = date.today()
        # Find expiry closest to target
        best_exp = min(exps,
                       key=lambda e: abs((datetime.strptime(e, '%Y-%m-%d').date() - today).days
                                         - days_to_expiry))
        T_days = (datetime.strptime(best_exp, '%Y-%m-%d').date() - today).days
        if T_days <= 0:
            return np.nan

        chain = t.option_chain(best_exp)
        spot  = t.history(period='1d')['Close'].iloc[-1]

        # Calls and puts near ATM
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        calls['dist'] = abs(calls['strike'] - spot)
        puts['dist']  = abs(puts['strike']  - spot)

        atm_calls = calls.nsmallest(3, 'dist')
        atm_puts  = puts.nsmallest(3, 'dist')

        iv_vals = pd.concat([
            atm_calls['impliedVolatility'],
            atm_puts['impliedVolatility']
        ]).dropna()

        if iv_vals.empty:
            return np.nan
        return float(iv_vals.median())

    except Exception:
        return np.nan

# =============================================================================
# CORRELATION CALCULATIONS
# =============================================================================

def rolling_correlation_matrix(returns_df, window=21):
    """
    Compute rolling average pairwise correlation for the most recent `window`
    days. Returns the full correlation matrix for the last window.
    """
    recent = returns_df.tail(window).dropna()
    return recent.corr()


def rolling_avg_pairwise_corr(returns_df, window=21):
    """
    Compute rolling average off-diagonal pairwise correlation (scalar)
    over the full history, using a rolling window.
    """
    cols = [c for c in returns_df.columns if c != INDEX_TICKER]
    sub  = returns_df[cols].dropna()

    avg_corrs = []
    dates_out = []
    for i in range(window, len(sub) + 1):
        chunk = sub.iloc[i - window:i]
        corr  = chunk.corr()
        # Upper triangle, off-diagonal
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        avg_c = upper.stack().mean()
        avg_corrs.append(avg_c)
        dates_out.append(sub.index[i - 1])

    return pd.Series(avg_corrs, index=dates_out)


def implied_index_correlation(index_iv, constituent_ivs, weights):
    """
    Derive implied correlation from the formula:
        σ_index² ≈ Σ_i Σ_j w_i w_j ρ_ij σ_i σ_j
    For a single ρ (uniform), solve:
        ρ_implied = (σ_index² - Σ_i w_i² σ_i²) / (Σ_i Σ_{j≠i} w_i w_j σ_i σ_j)
    Returns scalar implied correlation.
    """
    # Filter to tickers with valid IVs
    valid = {t: iv for t, iv in constituent_ivs.items() if not np.isnan(iv)}
    if not valid:
        return np.nan

    total_w = sum(weights[t] for t in valid)
    normalized_w = {t: weights[t] / total_w for t in valid}

    tickers = list(valid.keys())
    ivs_arr = np.array([valid[t] for t in tickers])
    w_arr   = np.array([normalized_w[t] for t in tickers])

    # Numerator: σ_index² - Σ w_i² σ_i²
    numerator = index_iv ** 2 - np.sum(w_arr ** 2 * ivs_arr ** 2)

    # Denominator: Σ_i Σ_{j≠i} w_i w_j σ_i σ_j
    n = len(tickers)
    denom = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                denom += w_arr[i] * w_arr[j] * ivs_arr[i] * ivs_arr[j]

    if denom == 0:
        return np.nan
    rho = np.clip(numerator / denom, -1.0, 1.0)
    return rho

# =============================================================================
# DISPERSION SIGNAL & REGIME
# =============================================================================

def dispersion_signal_score(implied_corr, realized_corr):
    """
    Dispersion score: how attractive is a long-dispersion trade?
      > 0 → implied corr > realized corr → sell index vol, buy stock vol
      < 0 → realized corr > implied corr → index vol cheap
    Returns value in [-1, 1].
    """
    if np.isnan(implied_corr) or np.isnan(realized_corr):
        return np.nan
    gap = implied_corr - realized_corr
    # Normalize: gap > 0.15 = full signal (1.0)
    return np.clip(gap / 0.15, -1.0, 1.0)


def classify_regime(avg_realized_corr):
    """Classify market regime from average realized correlation."""
    if avg_realized_corr > 0.70:
        return "CRISIS / MACRO-DRIVEN", "#e74c3c"
    elif avg_realized_corr > 0.50:
        return "ELEVATED CORRELATION", "#f39c12"
    elif avg_realized_corr > 0.30:
        return "NORMAL", "#3498db"
    else:
        return "LOW CORR / STOCK-PICKING", "#2ecc71"

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_dashboard(prices, returns, corr_matrices, implied_corr_val,
                   rolling_corr_series, dispersion_score,
                   constituent_ivs, index_iv, agg_realized):
    """Render the 4-panel correlation & dispersion dashboard."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 16), facecolor='#0d1117')
    fig.suptitle("SPY Correlation & Dispersion Monitor",
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)

    # Custom diverging colormap: red-white-green
    cmap = LinearSegmentedColormap.from_list(
        "corr_map", ["#e74c3c", "#2a2a3e", "#2ecc71"])

    # ── Panel 1: Correlation Heatmap ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#151b27')

    non_spy_cols = [c for c in returns.columns if c != INDEX_TICKER]
    corr_sub = corr_matrices['21d'].loc[non_spy_cols, non_spy_cols]
    mask_upper = np.zeros_like(corr_sub.values, dtype=bool)
    mask_upper[np.triu_indices_from(mask_upper, k=1)] = True

    im = ax1.imshow(corr_sub.values, cmap=cmap, vmin=-0.5, vmax=1.0,
                    aspect='auto')
    ax1.set_xticks(range(len(non_spy_cols)))
    ax1.set_yticks(range(len(non_spy_cols)))
    ax1.set_xticklabels(non_spy_cols, rotation=45, ha='right', color='#ccc', fontsize=8)
    ax1.set_yticklabels(non_spy_cols, color='#ccc', fontsize=8)
    ax1.set_title("21-Day Realized Correlation (Top 10 Holdings)",
                  color='white', fontsize=10, fontweight='bold', pad=8)

    for i in range(len(non_spy_cols)):
        for j in range(len(non_spy_cols)):
            val = corr_sub.values[i, j]
            txt_color = 'white' if abs(val) > 0.5 else '#aaa'
            ax1.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color=txt_color, fontsize=7.5)

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(colors='#aaa')

    # ── Panel 2: Implied vs Realized Correlation Time Series ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#151b27')

    # Plot rolling realized corr (21d)
    roll_21 = rolling_corr_series.get('21d', pd.Series(dtype=float))
    roll_63 = rolling_corr_series.get('63d', pd.Series(dtype=float))

    if not roll_21.empty:
        ax2.plot(roll_21.index, roll_21.values, color='#3498db', linewidth=1.5,
                 label='Realized Corr (21d)')
    if not roll_63.empty:
        ax2.plot(roll_63.index, roll_63.values, color='#9b59b6', linewidth=1.5,
                 label='Realized Corr (63d)', alpha=0.85)

    # Horizontal line for current implied correlation
    if not np.isnan(implied_corr_val):
        ax2.axhline(implied_corr_val, color='#e74c3c', linewidth=2,
                    linestyle='--', label=f'Implied Corr ({implied_corr_val:.2f})')
        # Shade the gap
        last_date = roll_21.index[-1] if not roll_21.empty else date.today()
        ax2.annotate(
            f" Gap = {implied_corr_val - float(roll_21.iloc[-1]):.2f}",
            xy=(last_date, implied_corr_val),
            color='#e74c3c', fontsize=9, va='center'
        )

    ax2.set_ylim(-0.1, 1.05)
    ax2.set_title("Implied vs Realized Correlation", color='white',
                  fontsize=10, fontweight='bold', pad=8)
    ax2.set_xlabel("Date", color='#aaa', fontsize=9)
    ax2.set_ylabel("Correlation", color='#aaa', fontsize=9)
    ax2.tick_params(colors='#ccc', labelsize=8)
    ax2.legend(fontsize=8, facecolor='#1a1f2e', labelcolor='white', framealpha=0.8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 3: Dispersion Signal Gauge ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#151b27')

    # Draw a semicircle gauge
    theta = np.linspace(0, np.pi, 200)
    # Zone colors
    zone_boundaries = [0, 0.3, 0.6, 1.0]  # weak/moderate/strong
    zone_colors     = ['#e74c3c', '#f39c12', '#2ecc71']
    for k in range(3):
        t1 = np.pi * (1 - zone_boundaries[k+1])
        t2 = np.pi * (1 - zone_boundaries[k])
        theta_zone = np.linspace(t1, t2, 50)
        ax3.fill_between(np.cos(theta_zone), np.sin(theta_zone),
                         np.cos(theta_zone) * 0.7, np.sin(theta_zone) * 0.7,
                         color=zone_colors[k], alpha=0.4)

    ax3.plot(np.cos(theta), np.sin(theta), color='#555', linewidth=2)
    ax3.plot(np.cos(theta) * 0.7, np.sin(theta) * 0.7, color='#555', linewidth=2)

    # Needle
    score_clamped = float(np.clip(dispersion_score if not np.isnan(dispersion_score) else 0,
                                   0, 1))
    needle_angle = np.pi * (1 - score_clamped)
    ax3.annotate("", xy=(np.cos(needle_angle) * 0.85, np.sin(needle_angle) * 0.85),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-0.15, 1.2)
    ax3.axis('off')
    ax3.set_title("Dispersion Opportunity Signal", color='white',
                  fontsize=10, fontweight='bold', pad=8)

    score_label = "STRONG SIGNAL" if score_clamped > 0.67 else \
                  "MODERATE SIGNAL" if score_clamped > 0.33 else "WEAK SIGNAL"
    score_color = "#2ecc71" if score_clamped > 0.67 else \
                  "#f39c12" if score_clamped > 0.33 else "#e74c3c"
    ax3.text(0, -0.1, score_label, ha='center', color=score_color,
             fontsize=12, fontweight='bold')
    ax3.text(0, -0.3,
             f"Impl Corr={implied_corr_val:.2f}  Realized={agg_realized:.2f}  "
             f"Gap={implied_corr_val - agg_realized:.2f}",
             ha='center', color='#aaa', fontsize=9)

    # Labels
    ax3.text(-1.1, 0.0, "SELL\nIDX VOL", ha='center', color='#2ecc71',
             fontsize=8, alpha=0.7)
    ax3.text(1.1, 0.0, "NEUTRAL", ha='center', color='#e74c3c',
             fontsize=8, alpha=0.7)
    ax3.text(0, 1.0, "MAX\nDISPERSION", ha='center', color='#f39c12',
             fontsize=8, alpha=0.7)

    # ── Panel 4: Sector Breakdown + Individual IVs ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#151b27')

    tickers_sorted = sorted(constituent_ivs.keys(),
                            key=lambda t: constituent_ivs.get(t, 0) or 0,
                            reverse=True)
    iv_vals = [constituent_ivs.get(t, np.nan) for t in tickers_sorted]
    # Color by sector
    sector_colors = {
        "Technology": "#3498db",
        "Consumer Disc.": "#e67e22",
        "Communication": "#9b59b6",
        "Financials": "#2ecc71",
    }
    bar_colors = [sector_colors.get(HOLDINGS[t]['sector'], '#aaa')
                  if t in HOLDINGS else '#7f8c8d' for t in tickers_sorted]

    valid_mask = [not np.isnan(v) for v in iv_vals]
    valid_tickers = [t for t, m in zip(tickers_sorted, valid_mask) if m]
    valid_ivs     = [v for v, m in zip(iv_vals, valid_mask) if m]
    valid_colors  = [c for c, m in zip(bar_colors, valid_mask) if m]

    bars = ax4.barh(valid_tickers, [v * 100 for v in valid_ivs],
                    color=valid_colors, edgecolor='#333', height=0.6)
    if not np.isnan(index_iv):
        ax4.axvline(index_iv * 100, color='white', linewidth=2,
                    linestyle='--', label=f'SPY IV ({index_iv*100:.1f}%)')
        ax4.legend(fontsize=8, facecolor='#1a1f2e', labelcolor='white')

    ax4.set_title("Constituent IV vs SPY IV (30-day ATM)", color='white',
                  fontsize=10, fontweight='bold', pad=8)
    ax4.set_xlabel("Implied Volatility (%)", color='#aaa', fontsize=9)
    ax4.tick_params(colors='#ccc', labelsize=9)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#333')
    for bar, val in zip(bars, valid_ivs):
        ax4.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val*100:.1f}%", va='center', color='white', fontsize=8)

    plt.savefig("correlation_monitor.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\nDashboard saved to correlation_monitor.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 8: Correlation & Dispersion Monitor")
    print("=" * 65)

    # 1. Fetch price history
    print("\n[1/5] Fetching price history...")
    prices = fetch_prices(TICKERS, lookback_days=LOOKBACK_DAYS)
    returns = prices.pct_change().dropna()
    print(f"  Got {len(prices)} trading days of data for {len(prices.columns)} tickers.")

    # 2. Compute rolling correlation matrices
    print("\n[2/5] Computing rolling correlation matrices...")
    corr_matrices = {}
    rolling_corr_series = {}
    for w in ROLLING_WINDOWS:
        label = f"{w}d"
        corr_matrices[label] = rolling_correlation_matrix(returns, window=w)
        rolling_corr_series[label] = rolling_avg_pairwise_corr(returns, window=w)
        avg = rolling_corr_series[label].iloc[-1] if len(rolling_corr_series[label]) > 0 else np.nan
        print(f"  {label} avg pairwise corr: {avg:.3f}")

    agg_realized_21 = float(rolling_corr_series['21d'].iloc[-1]) \
        if len(rolling_corr_series['21d']) > 0 else np.nan

    # 3. Fetch current implied volatilities
    print("\n[3/5] Fetching implied volatilities (this takes ~30 seconds)...")
    constituent_ivs = {}
    for ticker in HOLDINGS.keys():
        iv = fetch_atm_iv(ticker, days_to_expiry=NEAR_TERM_EXPIRY_DAYS)
        constituent_ivs[ticker] = iv
        status = f"{iv*100:.1f}%" if not np.isnan(iv) else "N/A"
        print(f"  {ticker:6s}: ATM IV = {status}")

    index_iv = fetch_atm_iv(INDEX_TICKER, days_to_expiry=NEAR_TERM_EXPIRY_DAYS)
    print(f"  {INDEX_TICKER:6s}: ATM IV = {index_iv*100:.1f}%" if not np.isnan(index_iv) else f"  {INDEX_TICKER}: IV unavailable")

    # 4. Compute implied correlation
    print("\n[4/5] Computing implied correlation...")
    weights = {t: HOLDINGS[t]['weight'] for t in HOLDINGS}
    implied_corr = implied_index_correlation(index_iv, constituent_ivs, weights)
    if not np.isnan(implied_corr):
        print(f"  Implied Correlation : {implied_corr:.3f}")
    else:
        print("  Implied Correlation : N/A (insufficient IV data)")
        implied_corr = agg_realized_21  # fallback for visualization

    gap = implied_corr - agg_realized_21 if not np.isnan(agg_realized_21) else np.nan
    d_score = dispersion_signal_score(implied_corr, agg_realized_21)

    # 5. Regime classification
    print("\n[5/5] Classifying market regime...")
    regime_label, _ = classify_regime(agg_realized_21)
    print(f"  21d Realized Corr   : {agg_realized_21:.3f}")
    print(f"  Implied Corr        : {implied_corr:.3f}")
    print(f"  Dispersion Gap      : {gap:.3f}" if not np.isnan(gap) else "  Dispersion Gap: N/A")
    print(f"  Dispersion Score    : {d_score:.2f}" if not np.isnan(d_score) else "  Dispersion Score: N/A")
    print(f"  Market Regime       : {regime_label}")

    if not np.isnan(d_score) and d_score > DISPERSION_THRESHOLD / 0.15:
        print("\n  *** DISPERSION OPPORTUNITY FLAGGED ***")
        print("  Strategy: Sell SPY straddle / Buy single-stock straddles on")
        high_iv = sorted([(t, v) for t, v in constituent_ivs.items()
                           if not np.isnan(v)], key=lambda x: x[1], reverse=True)[:3]
        for t, v in high_iv:
            print(f"    {t}: IV={v*100:.1f}%")

    print("\nRendering dashboard...")
    plot_dashboard(
        prices, returns, corr_matrices,
        implied_corr, rolling_corr_series, d_score,
        constituent_ivs, index_iv, agg_realized_21
    )
    print("Done.")


if __name__ == "__main__":
    main()
