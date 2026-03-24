# =============================================================================
# PROGRAM 11: Options Screener & Alert System
# =============================================================================
# Description:
#   Scans 30+ tickers for high-probability vol trading setups. For each:
#   pulls current IV (from options chain), calculates 30-day realized vol,
#   computes IVR and IVP (IV Rank and IV Percentile), scores each ticker on
#   a composite signal, and filters for actionable trades. Outputs a ranked
#   table of opportunities with trade recommendations.
#
# What it produces:
#   - Scatter plot: IVR vs IV-RV gap (the two key axes)
#   - Top 10 opportunities table with trade type recommendations
#   - Sector distribution of top opportunities
#   - IVP heatmap across the universe
#   - Console ranked output with trade notes
#   - Optional: email alert via smtplib (fill in credentials to enable)
#
# Platform: Google Colab  |  Runtime: ~3-6 minutes (network-bound)
#
# Install (run first in Colab):
#   !pip install yfinance numpy scipy matplotlib pandas
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, percentileofscore
import yfinance as yf
from datetime import datetime, date, timedelta

# =============================================================================
# UNIVERSE & CONFIGURATION
# =============================================================================

UNIVERSE = {
    # ETFs
    "SPY":  {"name": "S&P 500 ETF",          "sector": "ETF"},
    "QQQ":  {"name": "Nasdaq 100 ETF",        "sector": "ETF"},
    "IWM":  {"name": "Russell 2000 ETF",      "sector": "ETF"},
    "GLD":  {"name": "Gold ETF",              "sector": "Commodity"},
    "TLT":  {"name": "20Y Treasury ETF",      "sector": "Fixed Income"},
    "XLE":  {"name": "Energy ETF",            "sector": "Energy"},
    "XLF":  {"name": "Financials ETF",        "sector": "Financials"},
    "XLK":  {"name": "Technology ETF",        "sector": "Technology"},
    "XBI":  {"name": "Biotech ETF",           "sector": "Healthcare"},
    # Mega-cap tech
    "AAPL": {"name": "Apple",                 "sector": "Technology"},
    "MSFT": {"name": "Microsoft",             "sector": "Technology"},
    "NVDA": {"name": "NVIDIA",                "sector": "Technology"},
    "AMZN": {"name": "Amazon",                "sector": "Consumer Disc."},
    "GOOGL":{"name": "Alphabet",              "sector": "Communication"},
    "META": {"name": "Meta",                  "sector": "Communication"},
    "TSLA": {"name": "Tesla",                 "sector": "Consumer Disc."},
    # Financials
    "JPM":  {"name": "JPMorgan",              "sector": "Financials"},
    "GS":   {"name": "Goldman Sachs",         "sector": "Financials"},
    "BAC":  {"name": "Bank of America",       "sector": "Financials"},
    # Healthcare/Biotech
    "JNJ":  {"name": "Johnson & Johnson",     "sector": "Healthcare"},
    "PFE":  {"name": "Pfizer",                "sector": "Healthcare"},
    "MRNA": {"name": "Moderna",               "sector": "Healthcare"},
    # Energy
    "XOM":  {"name": "ExxonMobil",            "sector": "Energy"},
    "CVX":  {"name": "Chevron",               "sector": "Energy"},
    # Consumer
    "AMZN": {"name": "Amazon",                "sector": "Consumer Disc."},
    "WMT":  {"name": "Walmart",               "sector": "Consumer"},
    "COST": {"name": "Costco",                "sector": "Consumer"},
    # Industrial/Other
    "CAT":  {"name": "Caterpillar",           "sector": "Industrials"},
    "BA":   {"name": "Boeing",                "sector": "Industrials"},
    "COIN": {"name": "Coinbase",              "sector": "Crypto"},
    "MSTR": {"name": "MicroStrategy",         "sector": "Crypto"},
}

# Screening filters (adjust to widen/narrow the funnel)
MIN_IVR             = 40      # IV Rank must be at least 40 for vol selling
MIN_IV_RV_GAP       = 3.0     # IV - 30d RV must be at least 3% for vol selling
EARNINGS_BUFFER_DAYS = 7      # skip if earnings within this many days
MIN_OPTION_VOLUME   = 100     # skip if avg daily option volume < this (liquidity)
TARGET_DTE          = 30      # target days to expiry for IV calculation
IV_HISTORY_DAYS     = 252     # days of IV history for IVR/IVP calc

# Email alert (optional — fill in to enable)
EMAIL_CONFIG = {
    "enabled":   False,            # set True to enable
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "username":  "your_email@gmail.com",
    "password":  "your_app_password",   # Gmail App Password
    "to_email":  "your_email@gmail.com",
}

# =============================================================================
# DATA FETCHING
# =============================================================================

def get_hist_prices(ticker, days=IV_HISTORY_DAYS + 30):
    """Download price history."""
    try:
        end   = date.today()
        start = end - timedelta(days=days)
        hist = yf.download(ticker, start=start, end=end,
                           auto_adjust=True, progress=False)
        return hist['Close'] if not hist.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def realized_vol(prices, window=30):
    """Compute annualized realized volatility from log returns."""
    if len(prices) < window + 1:
        return np.nan
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rv = log_returns.tail(window).std() * np.sqrt(252)
    return float(rv)


def get_current_iv_and_data(ticker, target_dte=TARGET_DTE):
    """
    Pull current ATM IV and option volume/OI from the options chain.
    Returns (iv, avg_volume, put_call_ratio).
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return np.nan, 0, np.nan

        today = date.today()
        best_exp = min(
            exps,
            key=lambda e: abs((datetime.strptime(e, '%Y-%m-%d').date() - today).days - target_dte)
        )
        T_days = (datetime.strptime(best_exp, '%Y-%m-%d').date() - today).days
        if T_days <= 0:
            return np.nan, 0, np.nan

        hist = t.history(period='1d')
        if hist.empty:
            return np.nan, 0, np.nan
        spot = float(hist['Close'].iloc[-1])

        chain = t.option_chain(best_exp)
        calls, puts = chain.calls.copy(), chain.puts.copy()

        calls['dist'] = abs(calls['strike'] - spot)
        puts['dist']  = abs(puts['strike']  - spot)
        atm_calls = calls.nsmallest(3, 'dist')
        atm_puts  = puts.nsmallest(3, 'dist')

        iv_vals = pd.concat([
            atm_calls['impliedVolatility'],
            atm_puts['impliedVolatility']
        ]).dropna()
        iv = float(iv_vals.median()) if not iv_vals.empty else np.nan

        # Liquidity: total option volume
        total_call_vol = float(calls['volume'].fillna(0).sum())
        total_put_vol  = float(puts['volume'].fillna(0).sum())
        total_vol = total_call_vol + total_put_vol
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan

        return iv, total_vol, pc_ratio

    except Exception:
        return np.nan, 0, np.nan


def get_iv_history_proxy(prices, windows=(21, 63, 126, 252)):
    """
    Build a proxy IV history using rolling realized vol as a stand-in.
    (True IV history requires paid data. This is the free approach.)
    For IVR/IVP we'll use the 1-year high/low of the 30d RV as our proxy.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rv_series = log_returns.rolling(30).std() * np.sqrt(252)
    return rv_series.dropna()


def compute_ivr_ivp(current_iv, rv_history):
    """
    Compute IV Rank and IV Percentile using RV history as proxy.
    IVR = (current - 52w_low) / (52w_high - 52w_low) * 100
    IVP = percentile rank of current vs 52w history
    """
    if len(rv_history) < 30:
        return np.nan, np.nan
    hist_52w = rv_history.tail(252)
    low_52w  = float(hist_52w.min())
    high_52w = float(hist_52w.max())
    if high_52w == low_52w:
        return 50.0, 50.0
    ivr = (current_iv - low_52w) / (high_52w - low_52w) * 100
    ivp = percentileofscore(hist_52w.values, current_iv)
    return float(np.clip(ivr, 0, 100)), float(np.clip(ivp, 0, 100))


def check_earnings_proximity(ticker, buffer_days=EARNINGS_BUFFER_DAYS):
    """
    Check if earnings are within buffer_days using yfinance calendar.
    Returns (True, days_away) if too close, (False, days_away) if safe.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty:
            return False, 999
        # calendar is a DataFrame with columns like 'Earnings Date'
        if 'Earnings Date' in cal.columns:
            earn_dates = cal['Earnings Date'].dropna()
        elif 'Earnings High' in cal.columns:
            earn_dates = pd.Series([cal.iloc[0, 0]])
        else:
            return False, 999

        today_ts = pd.Timestamp(date.today())
        future   = [d for d in pd.to_datetime(earn_dates) if d >= today_ts]
        if not future:
            return False, 999
        nearest = min(future)
        days_away = (nearest - today_ts).days
        return days_away <= buffer_days, int(days_away)
    except Exception:
        return False, 999

# =============================================================================
# SCORING ENGINE
# =============================================================================

def score_opportunity(ivr, ivp, iv_rv_gap, put_call_ratio):
    """
    Composite opportunity score 0-100 for a vol-selling setup.
    Higher = more attractive.
    """
    score = 0.0
    weight_total = 0.0

    # IVR component (weight 35%)
    if not np.isnan(ivr):
        score       += (ivr / 100) * 35
        weight_total += 35

    # IVP component (weight 25%)
    if not np.isnan(ivp):
        score       += (ivp / 100) * 25
        weight_total += 25

    # IV-RV gap component (weight 30%) — normalized: 0% gap → 0, 15% gap → 30
    if not np.isnan(iv_rv_gap):
        gap_score = np.clip(iv_rv_gap / 0.15, 0, 1)
        score       += gap_score * 30
        weight_total += 30

    # Put/call ratio (weight 10%) — elevated puts → fear → higher IV
    if not np.isnan(put_call_ratio):
        pc_score = np.clip((put_call_ratio - 0.5) / 1.5, 0, 1)
        score       += pc_score * 10
        weight_total += 10

    if weight_total == 0:
        return np.nan
    return score * (100 / weight_total) if weight_total < 100 else score


def recommend_strategy(ivr, iv_rv_gap, put_call_ratio):
    """Suggest a specific options strategy based on the signal."""
    if np.isnan(ivr) or np.isnan(iv_rv_gap):
        return "MONITOR"
    if ivr > 70 and iv_rv_gap > 0.08:
        return "IRON CONDOR / SHORT STRADDLE"
    elif ivr > 60 and iv_rv_gap > 0.05:
        return "CASH-SECURED PUT / COVERED CALL"
    elif ivr > 50 and iv_rv_gap > 0.03:
        return "CREDIT SPREAD (PUT OR CALL)"
    elif ivr < 30:
        return "LONG STRADDLE / LONG CALL"  # vol buying setup
    else:
        return "WAIT / MONITOR"

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_screener_dashboard(results_df):
    """Render the 4-panel screener dashboard."""
    df = results_df.copy()
    df = df[df['composite_score'].notna()].copy()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 16), facecolor='#0d1117')
    fig.suptitle("Options Screener — Vol Opportunity Scanner",
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           top=0.94, bottom=0.07, left=0.06, right=0.97)

    PLOT_BG = '#151b27'

    SECTOR_COLORS = {
        "ETF": "#3498db", "Technology": "#9b59b6", "Financials": "#2ecc71",
        "Healthcare": "#e74c3c", "Energy": "#e67e22", "Communication": "#1abc9c",
        "Consumer Disc.": "#f39c12", "Consumer": "#f39c12", "Industrials": "#7f8c8d",
        "Commodity": "#d4ac0d", "Fixed Income": "#85c1e9", "Crypto": "#f0b27a",
    }

    # ── Panel 1: IVR vs IV-RV Gap Scatter ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PLOT_BG)

    valid = df[df['iv_rv_gap'].notna() & df['ivr'].notna()]
    for _, row in valid.iterrows():
        color = SECTOR_COLORS.get(row.get('sector', ''), '#aaa')
        size  = max(20, row['composite_score'] * 0.6) if not np.isnan(row['composite_score']) else 20
        ax1.scatter(row['ivr'], row['iv_rv_gap'] * 100,
                    s=size, color=color, alpha=0.8, edgecolors='#333', linewidths=0.5)
        ax1.annotate(row['ticker'],
                     (row['ivr'], row['iv_rv_gap'] * 100),
                     textcoords='offset points', xytext=(4, 2),
                     color='white', fontsize=7, alpha=0.9)

    # Threshold lines
    ax1.axvline(MIN_IVR, color='#f39c12', linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'IVR={MIN_IVR} threshold')
    ax1.axhline(MIN_IV_RV_GAP, color='#2ecc71', linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'IV-RV gap={MIN_IV_RV_GAP}% threshold')

    # Sweet spot box
    x_right = ax1.get_xlim()[1] if ax1.get_xlim()[1] > 100 else 100
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((MIN_IVR, MIN_IV_RV_GAP), x_right - MIN_IVR - 5,
                         (ax1.get_ylim()[1] if ax1.get_ylim()[1] > 20 else 30) - MIN_IV_RV_GAP - 2,
                         boxstyle="round,pad=0.5", linewidth=1.5,
                         edgecolor='#2ecc71', facecolor='#2ecc71', alpha=0.06)
    ax1.add_patch(box)
    ax1.text(MIN_IVR + 2, MIN_IV_RV_GAP + 0.5, "VOL SELLING ZONE",
             color='#2ecc71', fontsize=9, alpha=0.7)

    ax1.set_title("IV Rank vs IV-RV Gap — Universe Scatter (bubble size = score)",
                  color='white', fontsize=10, fontweight='bold', pad=8)
    ax1.set_xlabel("IV Rank (IVR)", color='#aaa', fontsize=9)
    ax1.set_ylabel("IV - Realized Vol (%)", color='#aaa', fontsize=9)
    ax1.legend(fontsize=8, facecolor='#1a1f2e', labelcolor='white', framealpha=0.8)
    ax1.tick_params(colors='#aaa', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 2: IVP Heatmap ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PLOT_BG)

    ivp_valid = df[df['ivp'].notna()].sort_values('ivp', ascending=False).head(20)
    cmap = LinearSegmentedColormap.from_list("ivp_map", ["#2ecc71", "#f39c12", "#e74c3c"])

    bars = ax2.barh(ivp_valid['ticker'], ivp_valid['ivp'],
                    color=[cmap(v / 100) for v in ivp_valid['ivp']],
                    edgecolor='#333', height=0.65)
    ax2.axvline(50, color='white', linewidth=1, linestyle='--', alpha=0.5)
    for bar, val in zip(bars, ivp_valid['ivp'].values):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va='center', color='white', fontsize=7.5)
    ax2.set_title("IV Percentile (IVP) — Top 20", color='white', fontsize=10,
                  fontweight='bold', pad=8)
    ax2.set_xlabel("IVP", color='#aaa', fontsize=8)
    ax2.set_xlim(0, 115)
    ax2.tick_params(colors='#ccc', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 3: Top 10 Opportunities Table ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(PLOT_BG)
    ax3.axis('off')
    ax3.set_title("Top Ranked Opportunities", color='white', fontsize=11,
                  fontweight='bold', loc='left', pad=8)

    top10 = df.sort_values('composite_score', ascending=False).head(10).reset_index(drop=True)
    col_headers = ["Rank", "Ticker", "Sector", "IV%", "RV30%", "IV-RV", "IVR", "IVP",
                   "Score", "Strategy Suggestion"]
    rows_data = []
    for i, row in top10.iterrows():
        rows_data.append([
            f"#{i+1}",
            row['ticker'],
            row.get('sector', ''),
            f"{row['current_iv']*100:.1f}%" if not np.isnan(row.get('current_iv', np.nan)) else "N/A",
            f"{row['rv30']*100:.1f}%"       if not np.isnan(row.get('rv30', np.nan)) else "N/A",
            f"{row['iv_rv_gap']*100:+.1f}%" if not np.isnan(row.get('iv_rv_gap', np.nan)) else "N/A",
            f"{row['ivr']:.0f}"             if not np.isnan(row.get('ivr', np.nan)) else "N/A",
            f"{row['ivp']:.0f}"             if not np.isnan(row.get('ivp', np.nan)) else "N/A",
            f"{row['composite_score']:.0f}" if not np.isnan(row.get('composite_score', np.nan)) else "N/A",
            row.get('strategy', 'N/A'),
        ])

    row_colors = []
    for i in range(len(rows_data)):
        row_colors.append(['#1a1f2e'] * len(col_headers))

    tbl = ax3.table(cellText=rows_data, colLabels=col_headers,
                    cellLoc='center', loc='center', cellColours=row_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.65)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333')
        if ri == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='#7fb3ff', fontweight='bold')

    # ── Panel 4: Sector Distribution of Top 15 ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PLOT_BG)

    top15 = df.sort_values('composite_score', ascending=False).head(15)
    sector_counts = top15['sector'].value_counts()
    sec_colors = [SECTOR_COLORS.get(s, '#aaa') for s in sector_counts.index]
    wedges, texts, autotexts = ax4.pie(
        sector_counts.values,
        labels=sector_counts.index,
        colors=sec_colors,
        autopct='%1.0f%%',
        pctdistance=0.8,
        startangle=90,
        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5}
    )
    for text in texts:
        text.set_color('white')
        text.set_fontsize(8)
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(7)
    ax4.set_title("Sector Distribution (Top 15)", color='white', fontsize=10,
                  fontweight='bold', pad=8)

    plt.savefig("options_screener.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\nDashboard saved to options_screener.png")
    plt.show()

# =============================================================================
# EMAIL ALERT
# =============================================================================

def send_email_alert(top_opportunities, config):
    """Send email with top 5 opportunities (requires Gmail App Password)."""
    if not config['enabled']:
        return
    try:
        body_lines = ["<h2>Options Screener — Top Opportunities</h2>",
                      f"<p>As of {date.today()}</p>",
                      "<table border='1' cellpadding='4' style='border-collapse:collapse'>",
                      "<tr><th>Rank</th><th>Ticker</th><th>IV%</th><th>IVR</th>"
                      "<th>Score</th><th>Strategy</th></tr>"]
        for i, row in top_opportunities.head(5).iterrows():
            body_lines.append(
                f"<tr><td>#{i+1}</td><td><b>{row['ticker']}</b></td>"
                f"<td>{row['current_iv']*100:.1f}%</td>"
                f"<td>{row['ivr']:.0f}</td>"
                f"<td>{row['composite_score']:.0f}</td>"
                f"<td>{row['strategy']}</td></tr>"
            )
        body_lines.append("</table>")
        body = "\n".join(body_lines)

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Options Screener Alert — {date.today()}"
        msg['From']    = config['username']
        msg['To']      = config['to_email']
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.sendmail(config['username'], config['to_email'], msg.as_string())
        print(f"  Email alert sent to {config['to_email']}")
    except Exception as e:
        print(f"  Email send failed: {e}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 11: Options Screener & Alert System")
    print("=" * 65)
    print(f"\nScanning {len(UNIVERSE)} tickers — estimated time: 3-6 minutes\n")

    tickers = list(UNIVERSE.keys())
    results = []

    for idx, ticker in enumerate(tickers):
        info = UNIVERSE[ticker]
        print(f"  [{idx+1:2d}/{len(tickers)}] {ticker:6s} ({info['name'][:20]:<20}) ", end='', flush=True)

        # Price history
        prices = get_hist_prices(ticker)
        if len(prices) < 60:
            print("SKIP (no price data)")
            continue

        # Current IV and option stats
        current_iv, opt_volume, pc_ratio = get_current_iv_and_data(ticker)
        if np.isnan(current_iv):
            print("SKIP (no IV)")
            continue

        # Realized vol
        rv30 = realized_vol(prices, window=30)
        rv_history = get_iv_history_proxy(prices)

        # IVR and IVP
        ivr, ivp = compute_ivr_ivp(current_iv, rv_history)

        # IV-RV gap
        iv_rv_gap = current_iv - rv30 if not np.isnan(rv30) else np.nan

        # Earnings check
        too_close, days_to_earn = check_earnings_proximity(ticker)
        if too_close:
            print(f"SKIP (earnings in {days_to_earn}d)")
            continue

        # Liquidity check
        if opt_volume < MIN_OPTION_VOLUME:
            print(f"LOW LIQUIDITY (vol={opt_volume:.0f})")
            # Still include but flag it

        # Score
        score = score_opportunity(ivr, ivp, iv_rv_gap, pc_ratio)
        strategy = recommend_strategy(ivr, iv_rv_gap, pc_ratio)

        results.append({
            "ticker":        ticker,
            "name":          info['name'],
            "sector":        info['sector'],
            "current_iv":    current_iv,
            "rv30":          rv30,
            "iv_rv_gap":     iv_rv_gap,
            "ivr":           ivr,
            "ivp":           ivp,
            "pc_ratio":      pc_ratio,
            "opt_volume":    opt_volume,
            "days_to_earn":  days_to_earn,
            "composite_score": score,
            "strategy":      strategy,
        })

        print(f"IV={current_iv*100:.1f}%  RV={rv30*100:.1f}%  "
              f"Gap={iv_rv_gap*100:+.1f}%  IVR={ivr:.0f}  Score={score:.0f}")

        # Rate limit — be polite to yfinance
        time.sleep(0.5)

    if not results:
        print("\nNo results — check internet connection or widen filters.")
        return

    df = pd.DataFrame(results)

    # ── Apply filters for actionable setups ───────────────────────────────────
    actionable = df[
        (df['ivr'] >= MIN_IVR) &
        (df['iv_rv_gap'] >= MIN_IV_RV_GAP / 100) &
        (df['composite_score'] >= 45)
    ].sort_values('composite_score', ascending=False)

    print("\n" + "=" * 65)
    print(f"  SCREENER RESULTS — {len(df)} tickers scanned, "
          f"{len(actionable)} actionable setups")
    print("=" * 65)

    print(f"\n{'Rank':>4}  {'Ticker':>6}  {'IV%':>6}  {'RV30%':>6}  "
          f"{'Gap':>6}  {'IVR':>5}  {'IVP':>5}  {'Score':>5}  Strategy")
    print("  " + "-" * 80)
    for i, row in actionable.head(10).iterrows():
        rank = df.sort_values('composite_score', ascending=False).index.get_loc(i) + 1
        print(f"  #{rank:<3}  {row['ticker']:>6}  "
              f"{row['current_iv']*100:>5.1f}%  "
              f"{row['rv30']*100:>5.1f}%  "
              f"{row['iv_rv_gap']*100:>+5.1f}%  "
              f"{row['ivr']:>5.0f}  "
              f"{row['ivp']:>5.0f}  "
              f"{row['composite_score']:>5.0f}  "
              f"{row['strategy']}")

    if len(actionable) == 0:
        print("  No tickers passed all filters. Consider lowering MIN_IVR or MIN_IV_RV_GAP.")

    # Email alert
    if EMAIL_CONFIG['enabled']:
        print("\nSending email alert...")
        send_email_alert(actionable.reset_index(drop=True), EMAIL_CONFIG)

    print("\nRendering dashboard...")
    plot_screener_dashboard(df)
    print("Done.")


if __name__ == "__main__":
    main()
