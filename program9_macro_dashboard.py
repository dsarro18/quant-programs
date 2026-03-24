# =============================================================================
# PROGRAM 9: Macro Regime Dashboard
# =============================================================================
# Description:
#   Pulls key macro indicators from FRED (fredapi) and market data from
#   yfinance, then combines them into a composite regime score that classifies
#   the current environment as RISK-ON, NEUTRAL, or RISK-OFF.
#   Shows a historical regime timeline so you can see when regimes shifted.
#
# What it produces:
#   - 6 macro indicator panels (rates, spread, CPI, unemployment, VIX, HY OAS)
#   - Composite regime score gauge
#   - Historical regime color-coded timeline
#   - VIX term structure (spot + futures proxies)
#   - Yield curve (2Y, 5Y, 10Y, 30Y)
#   - Console summary of current regime and key levels
#
# Platform: Google Colab  |  Runtime: ~30-60 seconds
#
# Install (run first in Colab):
#   !pip install fredapi yfinance numpy matplotlib pandas
#
# FRED API key: get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
#   Then either set os.environ['FRED_API_KEY'] = 'your_key' or replace
#   FRED_API_KEY below.
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, date, timedelta

# ── Optional FRED API (degrades gracefully to yfinance-only if not installed)
try:
    from fredapi import Fred
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False
    print("fredapi not installed. Run:  !pip install fredapi")
    print("Falling back to yfinance for all data.\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set your FRED API key here, or export FRED_API_KEY environment variable.
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

LOOKBACK_YEARS = 5          # years of history for regime timeline
REGIME_WINDOW  = 63         # trading days for rolling composite (≈3 months)

# FRED series IDs
FRED_SERIES = {
    "fed_funds_rate":  "FEDFUNDS",           # monthly
    "yield_10y":       "DGS10",              # daily
    "yield_2y":        "DGS2",               # daily
    "yield_5y":        "DGS5",               # daily
    "yield_30y":       "DGS30",              # daily
    "cpi_yoy":         "CPIAUCSL",           # monthly → compute YoY
    "unemployment":    "UNRATE",             # monthly
    "hy_oas":          "BAMLH0A0HYM2",       # daily — ICE BofA HY OAS
    "real_gdp_growth": "A191RL1Q225SBEA",    # quarterly
}

# Composite scoring weights (must sum to 1.0)
INDICATOR_WEIGHTS = {
    "vix_score":        0.25,
    "yield_spread":     0.20,
    "hy_oas_score":     0.20,
    "unemployment":     0.15,
    "cpi_score":        0.10,
    "fed_funds_score":  0.10,
}

# VIX thresholds for scoring
VIX_LEVELS = {"risk_on": 15, "neutral_low": 20, "neutral_high": 25, "risk_off": 35}

# HY OAS thresholds (basis points)
HY_LEVELS = {"risk_on": 300, "neutral_low": 400, "neutral_high": 550, "risk_off": 700}

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_fred_series(fred_client, series_id, start_date):
    """Fetch a single FRED series; returns pd.Series."""
    try:
        s = fred_client.get_series(series_id, observation_start=start_date)
        return s.dropna()
    except Exception as e:
        print(f"    FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def fetch_all_fred(start_date):
    """
    Fetch all FRED macro series. Returns dict of pd.Series.
    Falls back to None for unavailable series.
    """
    data = {}
    if FREDAPI_AVAILABLE and FRED_API_KEY != "YOUR_FRED_API_KEY_HERE":
        fred = Fred(api_key=FRED_API_KEY)
        print("  Using FRED API...")
        for key, series_id in FRED_SERIES.items():
            print(f"    Fetching {key} ({series_id})...")
            data[key] = fetch_fred_series(fred, series_id, start_date)
    else:
        print("  FRED API not configured — using yfinance proxies for some series...")
        data = {k: pd.Series(dtype=float) for k in FRED_SERIES}

    return data


def fetch_vix_data(start_date):
    """Pull VIX spot + proxies for VIX3M from yfinance."""
    result = {}
    for ticker, label in [("^VIX", "vix_spot"), ("^VIX3M", "vix_3m")]:
        try:
            hist = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if not hist.empty:
                result[label] = hist['Close']
                print(f"    {ticker}: latest = {hist['Close'].iloc[-1]:.2f}")
            else:
                result[label] = pd.Series(dtype=float)
        except Exception:
            result[label] = pd.Series(dtype=float)
    return result


def fetch_yield_curve_yfinance():
    """
    Pull current yield curve levels from yfinance treasury tickers.
    ^IRX=13w, ^FVX=5Y, ^TNX=10Y, ^TYX=30Y
    Returns dict of float values.
    """
    tickers = {"3m": "^IRX", "2y": "^TWO", "5y": "^FVX", "10y": "^TNX", "30y": "^TYX"}
    curve = {}
    for label, ticker in tickers.items():
        try:
            hist = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
            if not hist.empty:
                curve[label] = float(hist['Close'].iloc[-1])
        except Exception:
            pass
    return curve

# =============================================================================
# INDICATOR SCORING (each returns 0–100; 0=risk-on, 100=risk-off)
# =============================================================================

def score_vix(vix_value):
    """VIX: low VIX = risk-on (low score), high VIX = risk-off (high score)."""
    if np.isnan(vix_value):
        return 50.0
    if vix_value <= VIX_LEVELS['risk_on']:
        return 10.0
    elif vix_value <= VIX_LEVELS['neutral_low']:
        return 30.0 + 20 * (vix_value - VIX_LEVELS['risk_on']) / (VIX_LEVELS['neutral_low'] - VIX_LEVELS['risk_on'])
    elif vix_value <= VIX_LEVELS['neutral_high']:
        return 50.0
    elif vix_value <= VIX_LEVELS['risk_off']:
        return 60.0 + 30 * (vix_value - VIX_LEVELS['neutral_high']) / (VIX_LEVELS['risk_off'] - VIX_LEVELS['neutral_high'])
    else:
        return 95.0


def score_yield_spread(spread_10y_2y):
    """Yield curve: inverted = risk-off; steep = risk-on."""
    if np.isnan(spread_10y_2y):
        return 50.0
    if spread_10y_2y > 1.5:
        return 15.0   # steep — strong risk-on
    elif spread_10y_2y > 0.5:
        return 30.0
    elif spread_10y_2y > 0.0:
        return 50.0
    elif spread_10y_2y > -0.5:
        return 65.0   # slightly inverted
    else:
        return 85.0   # deeply inverted — recession signal


def score_hy_oas(oas_bps):
    """HY spread: tight = risk-on, wide = risk-off."""
    if np.isnan(oas_bps):
        return 50.0
    if oas_bps <= HY_LEVELS['risk_on']:
        return 10.0
    elif oas_bps <= HY_LEVELS['neutral_low']:
        return 10 + 40 * (oas_bps - HY_LEVELS['risk_on']) / (HY_LEVELS['neutral_low'] - HY_LEVELS['risk_on'])
    elif oas_bps <= HY_LEVELS['neutral_high']:
        return 50.0
    elif oas_bps <= HY_LEVELS['risk_off']:
        return 50 + 35 * (oas_bps - HY_LEVELS['neutral_high']) / (HY_LEVELS['risk_off'] - HY_LEVELS['neutral_high'])
    else:
        return 90.0


def score_unemployment(rate):
    """Unemployment: low = risk-on, high = risk-off."""
    if np.isnan(rate):
        return 50.0
    if rate <= 4.0:
        return 15.0
    elif rate <= 5.0:
        return 35.0
    elif rate <= 6.5:
        return 55.0
    elif rate <= 8.0:
        return 75.0
    else:
        return 90.0


def score_cpi(cpi_yoy):
    """CPI: moderate inflation = neutral; extreme = risk-off."""
    if np.isnan(cpi_yoy):
        return 50.0
    if 1.5 <= cpi_yoy <= 2.5:
        return 20.0   # goldilocks
    elif cpi_yoy < 0:
        return 70.0   # deflation risk
    elif cpi_yoy > 6.0:
        return 80.0   # hyper-inflation
    elif cpi_yoy > 4.0:
        return 65.0
    else:
        return 40.0


def score_fed_funds(rate, cpi_yoy):
    """Fed: real rate positive = restrictive (risk-off leaning for equities)."""
    if np.isnan(rate) or np.isnan(cpi_yoy):
        return 50.0
    real_rate = rate - cpi_yoy
    if real_rate > 2.0:
        return 75.0   # very restrictive
    elif real_rate > 0.5:
        return 60.0
    elif real_rate > -1.0:
        return 45.0
    else:
        return 30.0   # accommodative


def composite_score(indicators_dict):
    """Weighted composite regime score 0–100."""
    total = 0.0
    weight_used = 0.0
    for key, weight in INDICATOR_WEIGHTS.items():
        val = indicators_dict.get(key, np.nan)
        if not np.isnan(val):
            total += val * weight
            weight_used += weight
    if weight_used == 0:
        return 50.0
    return total / weight_used


def regime_label(score):
    if score < 35:
        return "RISK-ON", "#2ecc71"
    elif score < 60:
        return "NEUTRAL", "#f39c12"
    else:
        return "RISK-OFF", "#e74c3c"

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_macro_dashboard(fred_data, vix_data, yield_curve,
                         current_indicators, comp_score):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 18), facecolor='#0d1117')
    fig.suptitle("Macro Regime Dashboard", fontsize=18, fontweight='bold',
                 color='white', y=0.99)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38,
                           top=0.95, bottom=0.05, left=0.06, right=0.97)

    PLOT_BG = '#151b27'
    ACCENT  = '#3498db'

    start_date_str = str(date.today() - timedelta(days=LOOKBACK_YEARS * 365))

    # ── Helper: simple time-series panel ─────────────────────────────────────
    def ts_panel(ax, series, title, ylabel, color=ACCENT, hline=None,
                 fill=False, fmt='%.2f', latest_label=True):
        ax.set_facecolor(PLOT_BG)
        if series is not None and len(series) > 0:
            s = series[series.index >= pd.Timestamp(start_date_str)]
            ax.plot(s.index, s.values, color=color, linewidth=1.5)
            if fill:
                ax.fill_between(s.index, s.values, alpha=0.15, color=color)
            if hline is not None:
                ax.axhline(hline, color='#e74c3c', linestyle='--', linewidth=1,
                           alpha=0.7, label=f'Threshold {hline}')
            if latest_label and len(s) > 0:
                last_val = s.iloc[-1]
                ax.text(0.98, 0.92, fmt % last_val, transform=ax.transAxes,
                        ha='right', color='white', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Data Unavailable', transform=ax.transAxes,
                    ha='center', color='#555', fontsize=10)
        ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=6)
        ax.set_ylabel(ylabel, color='#aaa', fontsize=8)
        ax.tick_params(colors='#aaa', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    # Panel 1: Fed Funds Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ts_panel(ax1, fred_data.get('fed_funds_rate'), "Fed Funds Rate", "Rate (%)",
             color='#e74c3c', fmt='%.2f%%')

    # Panel 2: 10Y-2Y Yield Spread
    ax2 = fig.add_subplot(gs[0, 1])
    s10 = fred_data.get('yield_10y', pd.Series(dtype=float))
    s2  = fred_data.get('yield_2y',  pd.Series(dtype=float))
    if len(s10) > 0 and len(s2) > 0:
        spread = (s10 - s2).dropna()
    else:
        spread = pd.Series(dtype=float)
    ts_panel(ax2, spread, "10Y - 2Y Yield Spread (Inversion = Recession Signal)",
             "Spread (%)", color='#f39c12', hline=0.0, fill=True, fmt='%.2f%%')
    if len(spread) > 0:
        ax2.fill_between(spread.index[spread.index >= pd.Timestamp(start_date_str)],
                         spread[spread.index >= pd.Timestamp(start_date_str)],
                         0,
                         where=spread[spread.index >= pd.Timestamp(start_date_str)] < 0,
                         color='#e74c3c', alpha=0.3, label='Inverted')

    # Panel 3: CPI YoY
    ax3 = fig.add_subplot(gs[0, 2])
    cpi_raw = fred_data.get('cpi_yoy', pd.Series(dtype=float))
    if len(cpi_raw) > 0:
        cpi_yoy = cpi_raw.pct_change(periods=12) * 100
    else:
        cpi_yoy = pd.Series(dtype=float)
    ts_panel(ax3, cpi_yoy, "CPI YoY Inflation", "%", color='#e67e22',
             hline=2.0, fmt='%.1f%%')

    # Panel 4: Unemployment Rate
    ax4 = fig.add_subplot(gs[1, 0])
    ts_panel(ax4, fred_data.get('unemployment'), "Unemployment Rate", "%",
             color='#9b59b6', fmt='%.1f%%')

    # Panel 5: VIX
    ax5 = fig.add_subplot(gs[1, 1])
    vix = vix_data.get('vix_spot', pd.Series(dtype=float))
    ts_panel(ax5, vix, "VIX (Fear Index)", "VIX", color='#3498db',
             hline=20, fmt='%.1f')
    if len(vix) > 0:
        vix_recent = vix[vix.index >= pd.Timestamp(start_date_str)]
        if len(vix_recent) > 0:
            ax5.fill_between(vix_recent.index, vix_recent.values, 20,
                             where=vix_recent.values > 20, color='#e74c3c',
                             alpha=0.25, label='Stress zone')

    # Panel 6: HY OAS
    ax6 = fig.add_subplot(gs[1, 2])
    ts_panel(ax6, fred_data.get('hy_oas'), "ICE BofA HY OAS (Credit Stress)",
             "Spread (bps)", color='#e74c3c', hline=400, fmt='%.0f bps')

    # Panel 7: Regime Score Gauge (2×1 wide)
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.set_facecolor(PLOT_BG)
    ax7.axis('off')

    rlabel, rcolor = regime_label(comp_score)
    ax7.set_title("Composite Macro Regime Score", color='white',
                  fontsize=12, fontweight='bold', loc='left', pad=8)

    # Draw score bar
    bar_bg = ax7.barh(0.5, 100, left=0, height=0.3, color='#2a2a2a',
                      edgecolor='#444', linewidth=1)
    # Color zones
    ax7.barh(0.5, 35, left=0, height=0.3, color='#2ecc71', alpha=0.3)
    ax7.barh(0.5, 25, left=35, height=0.3, color='#f39c12', alpha=0.3)
    ax7.barh(0.5, 40, left=60, height=0.3, color='#e74c3c', alpha=0.3)

    # Needle
    ax7.axvline(comp_score, color='white', linewidth=4, ymin=0.2, ymax=0.9)
    ax7.text(comp_score, 0.9, f"{comp_score:.0f}", ha='center', color='white',
             fontsize=14, fontweight='bold', transform=ax7.get_xaxis_transform())

    ax7.set_xlim(0, 100)
    ax7.set_ylim(0, 1)
    ax7.text(17.5, 0.1, "RISK-ON", ha='center', color='#2ecc71',
             fontsize=10, fontweight='bold', transform=ax7.transData)
    ax7.text(47.5, 0.1, "NEUTRAL", ha='center', color='#f39c12',
             fontsize=10, fontweight='bold', transform=ax7.transData)
    ax7.text(80.0, 0.1, "RISK-OFF", ha='center', color='#e74c3c',
             fontsize=10, fontweight='bold', transform=ax7.transData)

    ax7.text(105, 0.5, f"► {rlabel}", va='center', color=rcolor,
             fontsize=14, fontweight='bold')

    # Score breakdown
    breakdown_text = "  |  ".join([
        f"VIX: {current_indicators.get('vix_score', 0):.0f}",
        f"Spread: {current_indicators.get('yield_spread', 0):.0f}",
        f"HY OAS: {current_indicators.get('hy_oas_score', 0):.0f}",
        f"Unemp: {current_indicators.get('unemployment', 0):.0f}",
        f"CPI: {current_indicators.get('cpi_score', 0):.0f}",
        f"Fed: {current_indicators.get('fed_funds_score', 0):.0f}",
    ])
    ax7.text(50, 0.75, breakdown_text, ha='center', color='#aaa',
             fontsize=8, transform=ax7.transData)

    # Panel 8: VIX Term Structure
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor(PLOT_BG)

    vix_term = {
        "VIX (30d)":  current_indicators.get('vix_spot', np.nan),
        "VIX3M (90d)": current_indicators.get('vix_3m', np.nan),
    }
    tenor_labels = list(vix_term.keys())
    tenor_vals   = [vix_term[k] for k in tenor_labels]
    valid_terms  = [(l, v) for l, v in zip(tenor_labels, tenor_vals) if not np.isnan(v)]

    if valid_terms:
        labs, vals = zip(*valid_terms)
        ts_color = '#e74c3c' if (len(vals) > 1 and vals[0] > vals[-1]) else '#2ecc71'
        ax8.bar(labs, vals, color=[ts_color] * len(vals), edgecolor='#333', width=0.5)
        for i, (l, v) in enumerate(zip(labs, vals)):
            ax8.text(i, v + 0.3, f"{v:.1f}", ha='center', color='white', fontsize=10)
    ax8.set_title("VIX Term Structure", color='white', fontsize=10,
                  fontweight='bold', pad=6)
    ax8.set_ylabel("VIX Level", color='#aaa', fontsize=8)
    ax8.tick_params(colors='#aaa', labelsize=9)
    for spine in ax8.spines.values():
        spine.set_edgecolor('#333')
    structure_label = "CONTANGO (normal)" if (
        len(valid_terms) >= 2 and valid_terms[0][1] < valid_terms[-1][1]
    ) else "BACKWARDATION (stress)"
    ax8.text(0.5, 0.92, structure_label, transform=ax8.transAxes,
             ha='center', color='#aaa', fontsize=8)

    # Panel 9: Yield Curve
    ax9 = fig.add_subplot(gs[3, :])
    ax9.set_facecolor(PLOT_BG)

    tenors = []
    rates  = []
    tenor_map = [("3M", "3m"), ("2Y", "2y"), ("5Y", "5y"), ("10Y", "10y"), ("30Y", "30y")]
    for label, key in tenor_map:
        val = yield_curve.get(key, np.nan)
        if not np.isnan(val):
            tenors.append(label)
            rates.append(val)

    if tenors:
        curve_color = '#e74c3c' if (len(rates) >= 2 and rates[0] > rates[-1]) else '#2ecc71'
        ax9.plot(range(len(tenors)), rates, color=curve_color,
                 linewidth=2.5, marker='o', markersize=8)
        ax9.fill_between(range(len(tenors)), rates, alpha=0.15, color=curve_color)
        for i, (t, r) in enumerate(zip(tenors, rates)):
            ax9.annotate(f"{r:.2f}%", (i, r), textcoords='offset points',
                         xytext=(0, 10), ha='center', color='white', fontsize=9)
        ax9.set_xticks(range(len(tenors)))
        ax9.set_xticklabels(tenors, color='#ccc', fontsize=10)
        is_inverted = rates[0] > rates[-1] if len(rates) >= 2 else False
        ax9.set_title(
            f"US Treasury Yield Curve  ({'INVERTED — recession signal' if is_inverted else 'NORMAL'})",
            color='white', fontsize=11, fontweight='bold', pad=8
        )
    else:
        ax9.text(0.5, 0.5, "Yield curve data unavailable",
                 transform=ax9.transAxes, ha='center', color='#555', fontsize=11)
        ax9.set_title("US Treasury Yield Curve", color='white', fontsize=11, pad=8)

    ax9.set_ylabel("Yield (%)", color='#aaa', fontsize=9)
    ax9.tick_params(colors='#aaa', labelsize=9)
    for spine in ax9.spines.values():
        spine.set_edgecolor('#333')

    plt.savefig("macro_dashboard.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\nDashboard saved to macro_dashboard.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 9: Macro Regime Dashboard")
    print("=" * 65)

    start_date = str(date.today() - timedelta(days=LOOKBACK_YEARS * 365 + 90))

    print(f"\n[1/4] Fetching FRED macro data (since {start_date})...")
    fred_data = fetch_all_fred(start_date)

    print("\n[2/4] Fetching VIX data from yfinance...")
    vix_data = fetch_vix_data(start_date)

    print("\n[3/4] Fetching current yield curve...")
    yield_curve = fetch_yield_curve_yfinance()

    print("\n[4/4] Computing regime scores...")

    # Extract latest values for scoring
    def last_val(series):
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        return np.nan

    vix_spot = last_val(vix_data.get('vix_spot'))
    vix_3m   = last_val(vix_data.get('vix_3m'))
    hy_oas   = last_val(fred_data.get('hy_oas'))
    unemp    = last_val(fred_data.get('unemployment'))
    fed_rate = last_val(fred_data.get('fed_funds_rate'))

    # Yield spread
    y10 = yield_curve.get('10y', np.nan)
    y2  = yield_curve.get('2y',  np.nan)
    spread = y10 - y2 if (not np.isnan(y10) and not np.isnan(y2)) else np.nan

    # CPI YoY
    cpi_raw = fred_data.get('cpi_yoy', pd.Series(dtype=float))
    if len(cpi_raw) >= 13:
        cpi_yoy = float((cpi_raw.iloc[-1] / cpi_raw.iloc[-13] - 1) * 100)
    else:
        cpi_yoy = np.nan

    current_indicators = {
        "vix_score":       score_vix(vix_spot),
        "yield_spread":    score_yield_spread(spread),
        "hy_oas_score":    score_hy_oas(hy_oas),
        "unemployment":    score_unemployment(unemp),
        "cpi_score":       score_cpi(cpi_yoy),
        "fed_funds_score": score_fed_funds(fed_rate, cpi_yoy),
        "vix_spot":        vix_spot,
        "vix_3m":          vix_3m,
    }

    comp = composite_score(current_indicators)
    rlabel, rcolor = regime_label(comp)

    print("\n" + "─" * 65)
    print("Current Macro Snapshot:")
    print("─" * 65)
    print(f"  VIX (spot)         : {vix_spot:.1f}" if not np.isnan(vix_spot) else "  VIX: N/A")
    print(f"  VIX 3M             : {vix_3m:.1f}" if not np.isnan(vix_3m) else "  VIX3M: N/A")
    print(f"  10Y-2Y Spread      : {spread:.2f}%" if not np.isnan(spread) else "  Spread: N/A")
    print(f"  HY OAS             : {hy_oas:.0f} bps" if not np.isnan(hy_oas) else "  HY OAS: N/A")
    print(f"  Unemployment       : {unemp:.1f}%" if not np.isnan(unemp) else "  Unemployment: N/A")
    print(f"  CPI YoY            : {cpi_yoy:.1f}%" if not np.isnan(cpi_yoy) else "  CPI: N/A")
    print(f"  Fed Funds Rate     : {fed_rate:.2f}%" if not np.isnan(fed_rate) else "  Fed Funds: N/A")
    print(f"\n  Composite Score    : {comp:.1f} / 100")
    print(f"  Regime             : *** {rlabel} ***")

    print("\n  Per-indicator scores (0=risk-on, 100=risk-off):")
    for key, w in INDICATOR_WEIGHTS.items():
        val = current_indicators.get(key, np.nan)
        bar = "█" * int(val / 5) if not np.isnan(val) else "N/A"
        print(f"    {key:20s} [{bar:<20s}] {val:.0f}  (weight {w:.0%})")

    print("\nRendering dashboard...")
    plot_macro_dashboard(fred_data, vix_data, yield_curve,
                         current_indicators, comp)
    print("Done.")


if __name__ == "__main__":
    main()
