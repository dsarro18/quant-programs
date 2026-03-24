# =============================================================================
# PROGRAM 10: Trade Journal Analytics Engine
# =============================================================================
# Description:
#   Reads your trade history from a CSV (or auto-generates sample data if the
#   file doesn't exist). Calculates all the performance metrics a professional
#   trader needs: equity curve, Sharpe ratio, win rates, behavioral biases,
#   and strategy comparison. Finds your actual edge — and your blind spots.
#
# What it produces:
#   - Equity curve with underwater drawdown overlay
#   - Rolling 30-day Sharpe ratio
#   - Win rate, avg win/loss, profit factor by strategy
#   - Behavioral heatmap: P&L by day-of-week and hour
#   - Holding period distribution and P&L correlation
#   - Full metrics printout to console
#
# CSV format (date, ticker, strategy, direction, entry_price, exit_price,
#              contracts, pnl):
#   2026-01-05,SPY,iron_condor,short,520.0,515.0,1,320.00
#
# Platform: Google Colab  |  Runtime: ~5-10 seconds
#
# Install (run first in Colab):
#   !pip install numpy pandas matplotlib seaborn
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, date, timedelta
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

JOURNAL_CSV_PATH = "trade_journal.csv"    # path to your CSV; auto-generated if missing
ROLLING_SHARPE_WINDOW = 30                # trading days for rolling Sharpe
RISK_FREE_DAILY = 0.0525 / 252            # annual risk-free rate / 252

# Strategies expected (used for color-coding and filtering)
STRATEGIES = [
    "iron_condor",
    "covered_call",
    "long_call",
    "long_put",
    "cash_secured_put",
    "straddle",
    "vertical_spread",
]

STRATEGY_COLORS = {
    "iron_condor":      "#3498db",
    "covered_call":     "#2ecc71",
    "long_call":        "#f39c12",
    "long_put":         "#e74c3c",
    "cash_secured_put": "#9b59b6",
    "straddle":         "#1abc9c",
    "vertical_spread":  "#e67e22",
}

# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def generate_sample_journal(n_trades=180, output_path=JOURNAL_CSV_PATH):
    """
    Generate realistic sample trade journal data and save to CSV.
    Simulates 6 months of options trading with realistic P&L distributions.
    """
    print(f"  Generating {n_trades} sample trades → {output_path}")
    random.seed(42)
    np.random.seed(42)

    tickers    = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMZN", "TSLA", "META"]
    directions = ["long", "short"]

    # Each strategy has different win rate and avg P&L characteristics
    strategy_params = {
        "iron_condor":      {"win_rate": 0.72, "avg_win": 280, "avg_loss": -420, "freq": 0.25},
        "covered_call":     {"win_rate": 0.68, "avg_win": 180, "avg_loss": -250, "freq": 0.18},
        "cash_secured_put": {"win_rate": 0.70, "avg_win": 200, "avg_loss": -380, "freq": 0.18},
        "long_call":        {"win_rate": 0.38, "avg_win": 850, "avg_loss": -290, "freq": 0.12},
        "long_put":         {"win_rate": 0.35, "avg_win": 700, "avg_loss": -260, "freq": 0.08},
        "straddle":         {"win_rate": 0.45, "avg_win": 620, "avg_loss": -380, "freq": 0.10},
        "vertical_spread":  {"win_rate": 0.58, "avg_win": 320, "avg_loss": -280, "freq": 0.09},
    }

    # Cumulative frequency for strategy selection
    strategy_list = list(strategy_params.keys())
    cum_freq = np.cumsum([strategy_params[s]['freq'] for s in strategy_list])

    start_date = date.today() - timedelta(days=200)
    rows = []

    for _ in range(n_trades):
        # Pick strategy
        r = random.random()
        strat = strategy_list[np.searchsorted(cum_freq, r)]
        params = strategy_params[strat]

        # Pick date (business days only)
        offset = random.randint(0, 195)
        trade_date = start_date + timedelta(days=offset)
        while trade_date.weekday() >= 5:
            trade_date += timedelta(days=1)

        # Entry hour (market hours: 9–16)
        entry_hour = random.choices(
            [9, 10, 11, 12, 13, 14, 15],
            weights=[0.25, 0.18, 0.12, 0.08, 0.12, 0.15, 0.10])[0]

        # Ticker
        ticker = random.choice(tickers)

        # Win or loss
        win = random.random() < params['win_rate']
        if win:
            pnl = abs(np.random.normal(params['avg_win'], params['avg_win'] * 0.4))
        else:
            pnl = -abs(np.random.normal(abs(params['avg_loss']), abs(params['avg_loss']) * 0.35))

        pnl = round(pnl, 2)

        # Generate plausible entry/exit prices (not critical for analytics)
        base_price = {"SPY": 540, "QQQ": 470, "AAPL": 225, "NVDA": 880,
                      "MSFT": 410, "AMZN": 195, "TSLA": 270, "META": 555}.get(ticker, 100)
        entry_px = round(base_price * (1 + np.random.normal(0, 0.02)), 2)
        exit_px  = round(entry_px + (pnl / (abs(pnl) * 0.1 + 1)) * np.random.uniform(0.5, 2.0), 2)
        contracts = random.choice([1, 1, 1, 2, 2, 3])
        direction = "short" if strat in ["iron_condor", "covered_call", "cash_secured_put"] else "long"

        # Holding period (days)
        if strat in ["iron_condor", "covered_call", "cash_secured_put"]:
            hold_days = random.randint(14, 45)
        elif strat in ["long_call", "long_put"]:
            hold_days = random.randint(1, 21)
        else:
            hold_days = random.randint(5, 30)

        close_date = trade_date + timedelta(days=hold_days)

        rows.append({
            "date":        trade_date.strftime('%Y-%m-%d'),
            "close_date":  close_date.strftime('%Y-%m-%d'),
            "hour":        entry_hour,
            "ticker":      ticker,
            "strategy":    strat,
            "direction":   direction,
            "entry_price": entry_px,
            "exit_price":  exit_px,
            "contracts":   contracts,
            "hold_days":   hold_days,
            "pnl":         pnl,
        })

    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"  Sample journal saved ({len(df)} trades).")
    return df

# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def load_journal(path=JOURNAL_CSV_PATH):
    """Load CSV or generate sample if missing."""
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'])
        print(f"  Loaded {len(df)} trades from {path}")
    else:
        print(f"  {path} not found — generating sample data...")
        df = generate_sample_journal(output_path=path)
        df['date'] = pd.to_datetime(df['date'])
    required = ['date', 'ticker', 'strategy', 'direction', 'pnl']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    if 'hold_days' not in df.columns:
        df['hold_days'] = 1
    if 'hour' not in df.columns:
        df['hour'] = 10
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_equity_curve(df):
    """Cumulative P&L and drawdown series."""
    equity = df.set_index('date')['pnl'].cumsum()
    rolling_max = equity.cummax()
    drawdown = equity - rolling_max
    return equity, drawdown


def rolling_sharpe(df, window=ROLLING_SHARPE_WINDOW):
    """Rolling Sharpe ratio (excess return / vol)."""
    daily = df.groupby('date')['pnl'].sum()
    daily_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='B')
    daily = daily.reindex(daily_idx, fill_value=0.0)

    def sharpe_window(returns):
        excess = returns - RISK_FREE_DAILY
        if excess.std() == 0:
            return np.nan
        return (excess.mean() / excess.std()) * np.sqrt(252)

    return daily.rolling(window).apply(sharpe_window, raw=True)


def strategy_breakdown(df):
    """Per-strategy performance metrics."""
    rows = []
    for strat in df['strategy'].unique():
        sub = df[df['strategy'] == strat]
        wins  = sub[sub['pnl'] > 0]
        losses = sub[sub['pnl'] <= 0]
        win_rate = len(wins) / len(sub) if len(sub) > 0 else 0
        avg_win  = wins['pnl'].mean()  if len(wins)  > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = (wins['pnl'].sum() / abs(losses['pnl'].sum())
                         if abs(losses['pnl'].sum()) > 0 else np.inf)
        total_pnl = sub['pnl'].sum()
        rows.append({
            "strategy":      strat,
            "n_trades":      len(sub),
            "win_rate":      win_rate,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "profit_factor": profit_factor,
            "total_pnl":     total_pnl,
        })
    return pd.DataFrame(rows).sort_values('total_pnl', ascending=False)


def behavioral_analysis(df):
    """P&L by day-of-week and hour."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek   # 0=Mon
    df['hour'] = df['hour'].fillna(10).astype(int)

    dow_pnl = df.groupby('day_of_week')['pnl'].agg(['sum', 'mean', 'count'])
    dow_pnl.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][:len(dow_pnl)]

    hour_pnl = df.groupby('hour')['pnl'].agg(['sum', 'mean', 'count'])

    return dow_pnl, hour_pnl


def overall_stats(df):
    """Compute headline statistics."""
    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    total  = len(df)
    equity, dd = compute_equity_curve(df)
    sharpe_ser = rolling_sharpe(df)
    final_sharpe = sharpe_ser.iloc[-1] if len(sharpe_ser) > 0 else np.nan
    max_dd = dd.min()
    return {
        "total_trades":    total,
        "win_rate":        len(wins) / total if total > 0 else 0,
        "avg_win":         wins['pnl'].mean() if len(wins) > 0 else 0,
        "avg_loss":        losses['pnl'].mean() if len(losses) > 0 else 0,
        "profit_factor":   (wins['pnl'].sum() / abs(losses['pnl'].sum())
                            if abs(losses['pnl'].sum()) > 0 else np.inf),
        "total_pnl":       df['pnl'].sum(),
        "max_drawdown":    max_dd,
        "sharpe":          final_sharpe,
        "best_trade":      df['pnl'].max(),
        "worst_trade":     df['pnl'].min(),
        "avg_hold_days":   df['hold_days'].mean() if 'hold_days' in df.columns else np.nan,
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_dashboard(df, equity, drawdown, sharpe_series, strat_df,
                   dow_pnl, hour_pnl, stats):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 18), facecolor='#0d1117')
    fig.suptitle("Trade Journal Analytics Engine",
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                           top=0.95, bottom=0.05, left=0.06, right=0.97)

    PLOT_BG = '#151b27'

    # ── Panel 1 (2 wide): Equity Curve + Drawdown ────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PLOT_BG)

    color_pos = '#2ecc71'
    color_neg = '#e74c3c'

    ax1.plot(equity.index, equity.values, color=color_pos, linewidth=2,
             label='Equity Curve', zorder=3)
    ax1.fill_between(equity.index, equity.values, 0,
                     where=equity.values >= 0, color=color_pos, alpha=0.15)
    ax1.fill_between(equity.index, equity.values, 0,
                     where=equity.values < 0, color=color_neg, alpha=0.2)
    ax1.axhline(0, color='#555', linewidth=0.8)

    ax2_twin = ax1.twinx()
    ax2_twin.fill_between(drawdown.index, drawdown.values, 0,
                          color='#e74c3c', alpha=0.35, label='Drawdown')
    ax2_twin.set_ylabel("Drawdown ($)", color='#e74c3c', fontsize=8)
    ax2_twin.tick_params(colors='#e74c3c', labelsize=7)

    ax1.set_title("Equity Curve & Drawdown", color='white', fontsize=11,
                  fontweight='bold', pad=8)
    ax1.set_ylabel("Cumulative P&L ($)", color='#aaa', fontsize=9)
    ax1.tick_params(colors='#aaa', labelsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    final_pnl = equity.iloc[-1] if len(equity) > 0 else 0
    ax1.text(0.98, 0.92, f"Total P&L: ${final_pnl:,.0f}",
             transform=ax1.transAxes, ha='right', color='white',
             fontsize=11, fontweight='bold')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 2: Rolling Sharpe ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PLOT_BG)

    sharpe_clean = sharpe_series.dropna()
    if len(sharpe_clean) > 0:
        ax2.plot(sharpe_clean.index, sharpe_clean.values, color='#3498db', linewidth=1.5)
        ax2.axhline(0, color='#555', linewidth=0.8)
        ax2.axhline(1, color='#2ecc71', linewidth=1, linestyle='--', alpha=0.7,
                    label='Sharpe=1')
        ax2.fill_between(sharpe_clean.index, sharpe_clean.values, 0,
                         where=sharpe_clean.values >= 0, color='#3498db', alpha=0.15)
        ax2.fill_between(sharpe_clean.index, sharpe_clean.values, 0,
                         where=sharpe_clean.values < 0, color='#e74c3c', alpha=0.2)
        last_s = sharpe_clean.iloc[-1]
        ax2.text(0.98, 0.92, f"Current: {last_s:.2f}",
                 transform=ax2.transAxes, ha='right', color='white',
                 fontsize=10, fontweight='bold')
    ax2.set_title(f"Rolling {ROLLING_SHARPE_WINDOW}d Sharpe Ratio", color='white',
                  fontsize=10, fontweight='bold', pad=8)
    ax2.set_ylabel("Sharpe", color='#aaa', fontsize=8)
    ax2.tick_params(colors='#aaa', labelsize=7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 3: Strategy P&L Comparison ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PLOT_BG)

    strats  = strat_df['strategy'].values
    totals  = strat_df['total_pnl'].values
    s_colors = [STRATEGY_COLORS.get(s, '#aaa') for s in strats]
    bars = ax3.barh(strats, totals, color=s_colors, edgecolor='#333', height=0.6)
    ax3.axvline(0, color='white', linewidth=1, alpha=0.5)
    for bar, val in zip(bars, totals):
        x = val + (max(abs(totals)) * 0.02 * np.sign(val))
        ax3.text(x, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}", va='center', color='white', fontsize=8)
    ax3.set_title("Total P&L by Strategy", color='white', fontsize=10,
                  fontweight='bold', pad=8)
    ax3.set_xlabel("P&L ($)", color='#aaa', fontsize=8)
    ax3.tick_params(colors='#ccc', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 4: Win Rate & Profit Factor ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PLOT_BG)

    x = np.arange(len(strat_df))
    width = 0.35
    wr_bars = ax4.bar(x - width/2, strat_df['win_rate'] * 100, width,
                      color='#3498db', alpha=0.85, label='Win Rate %', edgecolor='#333')
    ax4.axhline(50, color='#555', linewidth=0.8, linestyle='--')
    ax4_twin = ax4.twinx()
    pf_vals = np.clip(strat_df['profit_factor'].replace([np.inf, -np.inf], 5.0), 0, 5)
    pf_bars = ax4_twin.bar(x + width/2, pf_vals, width,
                            color='#2ecc71', alpha=0.85, label='Profit Factor',
                            edgecolor='#333')
    ax4_twin.axhline(1.0, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.7)
    ax4_twin.set_ylabel("Profit Factor", color='#2ecc71', fontsize=8)
    ax4_twin.tick_params(colors='#2ecc71', labelsize=7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strat_df['strategy'], rotation=35, ha='right',
                        color='#ccc', fontsize=7)
    ax4.set_title("Win Rate & Profit Factor by Strategy", color='white',
                  fontsize=10, fontweight='bold', pad=8)
    ax4.set_ylabel("Win Rate (%)", color='#3498db', fontsize=8)
    ax4.tick_params(colors='#aaa', labelsize=7)
    for spine in ax4.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 5: Behavioral — Day of Week ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PLOT_BG)

    dow_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in dow_pnl['sum'].values]
    ax5.bar(dow_pnl.index, dow_pnl['sum'].values, color=dow_colors, edgecolor='#333', width=0.6)
    ax5.axhline(0, color='white', linewidth=0.8, alpha=0.5)
    ax5.set_title("P&L by Day of Week", color='white', fontsize=10,
                  fontweight='bold', pad=8)
    ax5.set_ylabel("Total P&L ($)", color='#aaa', fontsize=8)
    ax5.tick_params(colors='#ccc', labelsize=9)
    for spine in ax5.spines.values():
        spine.set_edgecolor('#333')
    best_day = dow_pnl['sum'].idxmax()
    ax5.text(0.5, 0.92, f"Best day: {best_day}", transform=ax5.transAxes,
             ha='center', color='#2ecc71', fontsize=9)

    # ── Panel 6: Behavioral — Hour of Day ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor(PLOT_BG)

    hour_pnl_sorted = hour_pnl.sort_index()
    h_colors = ['#2ecc71' if v >= 0 else '#e74c3c'
                for v in hour_pnl_sorted['sum'].values]
    ax6.bar([f"{h:02d}:00" for h in hour_pnl_sorted.index],
            hour_pnl_sorted['sum'].values, color=h_colors, edgecolor='#333', width=0.6)
    ax6.axhline(0, color='white', linewidth=0.8, alpha=0.5)
    ax6.set_title("P&L by Entry Hour", color='white', fontsize=10,
                  fontweight='bold', pad=8)
    ax6.set_ylabel("Total P&L ($)", color='#aaa', fontsize=8)
    ax6.tick_params(colors='#ccc', labelsize=8)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for spine in ax6.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 7: Holding Period Distribution ─────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor(PLOT_BG)

    if 'hold_days' in df.columns:
        hold = df['hold_days'].dropna()
        ax7.hist(hold.values, bins=20, color='#9b59b6', edgecolor='#333',
                 alpha=0.85)
        ax7.axvline(hold.mean(), color='white', linewidth=1.5, linestyle='--',
                    label=f'Mean={hold.mean():.1f}d')
        ax7.legend(fontsize=8, facecolor='#1a1f2e', labelcolor='white')
    ax7.set_title("Holding Period Distribution", color='white', fontsize=10,
                  fontweight='bold', pad=8)
    ax7.set_xlabel("Days Held", color='#aaa', fontsize=8)
    ax7.set_ylabel("# Trades", color='#aaa', fontsize=8)
    ax7.tick_params(colors='#aaa', labelsize=8)
    for spine in ax7.spines.values():
        spine.set_edgecolor('#333')

    # ── Panel 8: Headline Stats Summary ──────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor(PLOT_BG)
    ax8.axis('off')
    ax8.set_title("Performance Summary", color='white', fontsize=10,
                  fontweight='bold', pad=8)

    summary = [
        ("Total Trades",     f"{stats['total_trades']}"),
        ("Win Rate",         f"{stats['win_rate']*100:.1f}%"),
        ("Avg Win",          f"${stats['avg_win']:,.0f}"),
        ("Avg Loss",         f"${stats['avg_loss']:,.0f}"),
        ("Profit Factor",    f"{min(stats['profit_factor'], 99):.2f}"),
        ("Total P&L",        f"${stats['total_pnl']:,.0f}"),
        ("Max Drawdown",     f"${stats['max_drawdown']:,.0f}"),
        ("Sharpe (rolling)", f"{stats['sharpe']:.2f}" if not np.isnan(stats['sharpe']) else "N/A"),
        ("Best Trade",       f"${stats['best_trade']:,.0f}"),
        ("Worst Trade",      f"${stats['worst_trade']:,.0f}"),
        ("Avg Hold (days)",  f"{stats['avg_hold_days']:.1f}" if not np.isnan(stats['avg_hold_days']) else "N/A"),
    ]

    for j, (label, value) in enumerate(summary):
        y = 0.95 - j * 0.087
        # Color-code P&L fields
        if any(x in label for x in ["P&L", "Win", "Loss", "Factor", "Sharpe", "Drawdown"]):
            try:
                num = float(value.replace('$', '').replace('%', '').replace(',', ''))
                color = '#2ecc71' if num > 0 else '#e74c3c' if num < 0 else 'white'
            except Exception:
                color = 'white'
        else:
            color = '#ccc'
        ax8.text(0.05, y, label, transform=ax8.transAxes, color='#888', fontsize=9)
        ax8.text(0.95, y, value, transform=ax8.transAxes, ha='right',
                 color=color, fontsize=9, fontweight='bold')
        ax8.axhline(y=y - 0.035, xmin=0.03, xmax=0.97, color='#333',
                    linewidth=0.5, transform=ax8.transAxes)

    plt.savefig("journal_analytics.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\nDashboard saved to journal_analytics.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  PROGRAM 10: Trade Journal Analytics Engine")
    print("=" * 65)

    print(f"\n[1/5] Loading trade journal from '{JOURNAL_CSV_PATH}'...")
    df = load_journal(JOURNAL_CSV_PATH)

    print("\n[2/5] Computing equity curve and drawdown...")
    equity, drawdown = compute_equity_curve(df)
    max_dd = drawdown.min()
    print(f"  Final equity: ${equity.iloc[-1]:,.0f}")
    print(f"  Max drawdown: ${max_dd:,.0f}")

    print("\n[3/5] Computing rolling Sharpe ratio...")
    sharpe_series = rolling_sharpe(df)
    last_sharpe = sharpe_series.dropna().iloc[-1] if len(sharpe_series.dropna()) > 0 else np.nan
    print(f"  Current {ROLLING_SHARPE_WINDOW}d Sharpe: {last_sharpe:.2f}")

    print("\n[4/5] Analyzing strategy performance...")
    strat_df = strategy_breakdown(df)
    print("\n  Strategy Breakdown:")
    print(f"  {'Strategy':<20} {'Trades':>6} {'WinRate':>8} {'PF':>6} {'Total P&L':>10}")
    print("  " + "-" * 55)
    for _, row in strat_df.iterrows():
        print(f"  {row['strategy']:<20} {row['n_trades']:>6} "
              f"{row['win_rate']*100:>7.1f}% {min(row['profit_factor'],99):>6.2f} "
              f"${row['total_pnl']:>9,.0f}")

    print("\n[5/5] Behavioral analysis...")
    dow_pnl, hour_pnl = behavioral_analysis(df)
    best_dow  = dow_pnl['sum'].idxmax()
    worst_dow = dow_pnl['sum'].idxmin()
    print(f"  Best day: {best_dow}  Worst day: {worst_dow}")

    stats = overall_stats(df)

    print("\n" + "─" * 65)
    print(f"  OVERALL STATS:")
    print(f"  Win Rate:       {stats['win_rate']*100:.1f}%")
    print(f"  Profit Factor:  {min(stats['profit_factor'],99):.2f}")
    print(f"  Total P&L:      ${stats['total_pnl']:,.0f}")
    print(f"  Max Drawdown:   ${stats['max_drawdown']:,.0f}")
    print(f"  Sharpe Ratio:   {stats['sharpe']:.2f}" if not np.isnan(stats['sharpe']) else "  Sharpe: N/A")

    if stats['profit_factor'] > 1.5 and stats['win_rate'] > 0.55:
        print("\n  EDGE CONFIRMED: Strong win rate + profit factor. Stay the course.")
    elif stats['profit_factor'] < 1.0:
        print("\n  WARNING: Profit factor < 1.0 — losing money on average. Review strategy.")

    print("\nRendering dashboard...")
    plot_dashboard(df, equity, drawdown, sharpe_series, strat_df,
                   dow_pnl, hour_pnl, stats)
    print("Done.")


if __name__ == "__main__":
    main()
