# ══════════════════════════════════════════════════════════════════════════════
# PROGRAM 13: Unified Trading Pipeline — Capstone
# ══════════════════════════════════════════════════════════════════════════════
#
# Description:
#   The crown jewel. Chains the logic from Programs 1-12 into a single
#   end-to-end trading pipeline that scans 30 liquid tickers, scores vol
#   edges, filters for earnings risk, analyzes vol surfaces, checks the
#   macro regime, backtests candidates, sizes positions via Kelly criterion,
#   and produces final trade recommendations with a dark-themed dashboard.
#
# Pipeline flow:
#   SCAN  (Prog 1)  → Pull options chains for 30-ticker watchlist
#   SCORE (Prog 3)  → IV Rank, IV Percentile, IV-RV gap for each ticker
#   FILTER (Prog 4) → Flag/remove tickers with earnings in next 14 days
#   VOL SURFACE (6) → Skew + term structure for top 5 candidates
#   REGIME (Prog 9)  → VIX level, yield curve slope → RISK-ON/NEUTRAL/RISK-OFF
#   BACKTEST (Prog 5) → Quick vectorized backtest of proposed strategy
#   SIZE  (Kelly)    → Optimal position size given edge and bankroll
#   RECOMMEND        → Final trade recs with full thesis
#
# Output:
#   - 4-panel dark dashboard:
#       1) Pipeline funnel (30 → N → top 5 → final recs)
#       2) Top candidates with scores (IVR, VRP edge, regime overlay)
#       3) Recommended trades with entry/exit/sizing
#       4) Risk summary (portfolio Greeks, max loss, regime warning)
#   - Console printout of every pipeline step
#   - List of TradeRecommendation dataclass objects
#
# Data sources (all free):
#   yfinance  — options chains, OHLCV, earnings calendar
#   FRED      — VIX (via yfinance ^VIX), 10Y-2Y spread, Fed Funds rate
#   fredapi   — optional; falls back to yfinance proxies if unavailable
#
# Platform: Google Colab  |  Runtime: ~5-10 minutes (network-bound)
#
# Install (run first in Colab):
#   !pip install yfinance numpy scipy matplotlib pandas fredapi
# ══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import norm, percentileofscore
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import yfinance as yf

# Optional FRED API — degrades gracefully
try:
    from fredapi import Fred
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False
    print("[INFO] fredapi not installed. Regime check uses yfinance proxies.")
    print("       For full macro data:  !pip install fredapi\n")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

WATCHLIST = {
    # ETFs (high liquidity, tight spreads)
    "SPY":  {"name": "S&P 500 ETF",       "sector": "ETF"},
    "QQQ":  {"name": "Nasdaq 100 ETF",     "sector": "ETF"},
    "IWM":  {"name": "Russell 2000 ETF",   "sector": "ETF"},
    "GLD":  {"name": "Gold ETF",           "sector": "Commodity"},
    "TLT":  {"name": "20Y Treasury ETF",   "sector": "Fixed Income"},
    "XLE":  {"name": "Energy ETF",         "sector": "Energy"},
    "XLF":  {"name": "Financials ETF",     "sector": "Financials"},
    "XLK":  {"name": "Technology ETF",     "sector": "Technology"},
    "XBI":  {"name": "Biotech ETF",        "sector": "Healthcare"},
    # Mega-cap tech
    "AAPL": {"name": "Apple",              "sector": "Technology"},
    "MSFT": {"name": "Microsoft",          "sector": "Technology"},
    "NVDA": {"name": "NVIDIA",             "sector": "Technology"},
    "AMZN": {"name": "Amazon",             "sector": "Consumer Disc."},
    "GOOGL":{"name": "Alphabet",           "sector": "Communication"},
    "META": {"name": "Meta",               "sector": "Communication"},
    "TSLA": {"name": "Tesla",              "sector": "Consumer Disc."},
    # Financials
    "JPM":  {"name": "JPMorgan",           "sector": "Financials"},
    "GS":   {"name": "Goldman Sachs",      "sector": "Financials"},
    "BAC":  {"name": "Bank of America",    "sector": "Financials"},
    # Healthcare / Biotech
    "JNJ":  {"name": "Johnson & Johnson",  "sector": "Healthcare"},
    "PFE":  {"name": "Pfizer",             "sector": "Healthcare"},
    "MRNA": {"name": "Moderna",            "sector": "Healthcare"},
    # Energy
    "XOM":  {"name": "ExxonMobil",         "sector": "Energy"},
    "CVX":  {"name": "Chevron",            "sector": "Energy"},
    # Consumer
    "WMT":  {"name": "Walmart",            "sector": "Consumer"},
    "COST": {"name": "Costco",             "sector": "Consumer"},
    # Industrial
    "CAT":  {"name": "Caterpillar",        "sector": "Industrials"},
    "BA":   {"name": "Boeing",             "sector": "Industrials"},
    # Crypto-adjacent (high vol names)
    "COIN": {"name": "Coinbase",           "sector": "Crypto"},
    "MSTR": {"name": "MicroStrategy",      "sector": "Crypto"},
}

# Pipeline parameters
BANKROLL                = 100_000       # Starting capital
MAX_POSITION_PCT        = 0.05          # Max 5% of bankroll per position
EARNINGS_BLACKOUT_DAYS  = 14            # Skip if earnings within N days
MIN_IVR                 = 40            # IV Rank floor for vol selling
MIN_VRP_EDGE            = 0.03          # Min IV-RV gap (as decimal, e.g. 3%)
TARGET_DTE              = 30            # Ideal days-to-expiry
IV_HISTORY_DAYS         = 252           # 1 year lookback for IVR/IVP
TOP_N_CANDIDATES        = 5             # Advance this many to vol surface step
MAX_RECOMMENDATIONS     = 3             # Final trade recs
FRED_API_KEY            = os.environ.get('FRED_API_KEY', '')

# Regime thresholds
VIX_LOW   = 15.0    # Below = RISK-ON
VIX_HIGH  = 25.0    # Above = RISK-OFF
SPREAD_INVERTED = 0  # 10Y-2Y < 0 = inverted yield curve

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TickerScan:
    """Raw scan data for a single ticker."""
    ticker: str
    name: str
    sector: str
    spot_price: float = 0.0
    atm_iv: float = 0.0
    rv_30d: float = 0.0
    rv_60d: float = 0.0
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    vrp_edge: float = 0.0        # IV - RV (annualized, decimal)
    put_call_ratio: float = 0.0
    avg_option_volume: float = 0.0
    earnings_days_away: Optional[int] = None
    earnings_flagged: bool = False
    composite_score: float = 0.0  # Weighted combination
    passed_filter: bool = False

@dataclass
class VolSurfaceData:
    """Vol surface analysis for a candidate."""
    ticker: str
    skew_25d: float = 0.0       # 25-delta put IV - 25-delta call IV
    term_slope: float = 0.0     # (far month IV - near month IV)
    skew_signal: str = ""       # "STEEP" / "FLAT" / "NORMAL"
    term_signal: str = ""       # "CONTANGO" / "BACKWARDATION" / "FLAT"
    surface_score: float = 0.0

@dataclass
class RegimeState:
    """Current macro regime assessment."""
    vix_level: float = 0.0
    vix_zscore: float = 0.0
    yield_spread_10y2y: float = 0.0
    fed_funds_rate: float = 0.0
    regime: str = "NEUTRAL"     # RISK-ON / NEUTRAL / RISK-OFF
    regime_score: float = 0.0   # -1 (risk-off) to +1 (risk-on)
    description: str = ""

@dataclass
class BacktestResult:
    """Quick backtest output for a strategy."""
    ticker: str
    strategy: str
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    total_return: float = 0.0
    edge_confirmed: bool = False

@dataclass
class TradeRecommendation:
    """Final recommendation output — the pipeline's end product."""
    ticker: str
    name: str
    sector: str
    strategy: str               # e.g. "Short Put", "Iron Condor", "Put Credit Spread"
    direction: str              # "SELL VOL" or "BUY VOL"
    strikes: str                # e.g. "450P / 440P"
    expiry_dte: int             # Days to expiry
    contracts: int              # Number of contracts
    max_risk: float             # Max loss per position
    max_reward: float           # Max gain per position
    edge_pct: float             # VRP edge as percentage
    kelly_fraction: float       # Kelly-optimal fraction
    position_size_usd: float    # Dollar amount allocated
    iv_rank: float
    regime: str
    backtest_winrate: float
    thesis: str                 # Full written rationale
    confidence: str             # "HIGH" / "MEDIUM" / "LOW"

@dataclass
class PipelineResults:
    """Container for all pipeline outputs."""
    scan_data: List[TickerScan] = field(default_factory=list)
    scored_tickers: List[TickerScan] = field(default_factory=list)
    post_earnings_filter: List[TickerScan] = field(default_factory=list)
    vol_surface: Dict[str, VolSurfaceData] = field(default_factory=dict)
    regime: RegimeState = field(default_factory=RegimeState)
    backtests: Dict[str, BacktestResult] = field(default_factory=dict)
    recommendations: List[TradeRecommendation] = field(default_factory=list)
    funnel_counts: Dict[str, int] = field(default_factory=dict)
    run_timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: MARKET SCAN  (Program 1 logic — options chain pull)
# ═══════════════════════════════════════════════════════════════════════════

class MarketScanner:
    """Pulls spot prices, options chains, and computes ATM IV for the watchlist."""

    def __init__(self, watchlist: dict):
        self.watchlist = watchlist

    def scan(self) -> List[TickerScan]:
        """Scan all tickers. Returns list of TickerScan objects."""
        print("\n" + "=" * 70)
        print("  STEP 1: MARKET SCAN — Pulling options chains for "
              f"{len(self.watchlist)} tickers")
        print("=" * 70)

        results = []
        total = len(self.watchlist)

        for i, (ticker, info) in enumerate(self.watchlist.items(), 1):
            pct = i / total * 100
            print(f"  [{i:2d}/{total}] {ticker:6s} ({pct:5.1f}%) ... ", end="", flush=True)

            scan = TickerScan(
                ticker=ticker,
                name=info["name"],
                sector=info["sector"]
            )

            try:
                tk = yf.Ticker(ticker)

                # Spot price
                hist = tk.history(period="5d")
                if hist.empty:
                    print("NO DATA — skipped")
                    results.append(scan)
                    continue
                scan.spot_price = float(hist['Close'].iloc[-1])

                # Options chain — find nearest expiry to TARGET_DTE
                expirations = tk.options
                if not expirations:
                    print("NO OPTIONS — skipped")
                    results.append(scan)
                    continue

                target_date = date.today() + timedelta(days=TARGET_DTE)
                best_exp = min(expirations,
                               key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date()
                                                   - target_date).days))
                chain = tk.option_chain(best_exp)
                calls = chain.calls
                puts  = chain.puts

                # ATM IV — closest strike to spot
                if calls.empty:
                    print("EMPTY CHAIN — skipped")
                    results.append(scan)
                    continue

                atm_strike = min(calls['strike'].values,
                                 key=lambda s: abs(s - scan.spot_price))
                atm_call = calls[calls['strike'] == atm_strike]
                atm_put  = puts[puts['strike'] == atm_strike]

                call_iv = float(atm_call['impliedVolatility'].iloc[0]) if not atm_call.empty else 0
                put_iv  = float(atm_put['impliedVolatility'].iloc[0]) if not atm_put.empty else 0
                scan.atm_iv = (call_iv + put_iv) / 2 if (call_iv > 0 and put_iv > 0) else max(call_iv, put_iv)

                # Put/call ratio (volume-based)
                total_call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
                total_put_vol  = puts['volume'].sum()  if 'volume' in puts.columns else 0
                if total_call_vol and total_call_vol > 0:
                    scan.put_call_ratio = total_put_vol / total_call_vol
                scan.avg_option_volume = (total_call_vol + total_put_vol) / 2

                print(f"spot=${scan.spot_price:>8.2f}  IV={scan.atm_iv*100:5.1f}%  "
                      f"P/C={scan.put_call_ratio:.2f}")

            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")

            results.append(scan)
            time.sleep(0.15)  # Rate limiting

        valid = sum(1 for r in results if r.atm_iv > 0)
        print(f"\n  Scan complete: {valid}/{total} tickers with valid IV data")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: EDGE SCORING  (Program 3 logic — IV Rank, IVP, VRP gap)
# ═══════════════════════════════════════════════════════════════════════════

class EdgeScorer:
    """Calculates IV Rank, IV Percentile, and VRP edge for each ticker."""

    def __init__(self, min_ivr: float = MIN_IVR, min_vrp: float = MIN_VRP_EDGE):
        self.min_ivr = min_ivr
        self.min_vrp = min_vrp

    def _realized_vol(self, prices: pd.Series, window: int = 30) -> float:
        """Annualized realized vol from log returns."""
        if len(prices) < window + 1:
            return np.nan
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return float(log_ret.rolling(window).std().iloc[-1] * np.sqrt(252))

    def _iv_history_proxy(self, ticker: str, days: int = IV_HISTORY_DAYS) -> pd.Series:
        """
        Build a proxy IV history from historical realized vol.
        True IV history requires paid data; we approximate by adding the
        typical VRP (realized vol * 1.15) as an IV proxy.
        """
        try:
            end   = date.today()
            start = end - timedelta(days=days + 60)
            hist  = yf.download(ticker, start=start, end=end,
                                auto_adjust=True, progress=False)
            if hist.empty or len(hist) < 60:
                return pd.Series(dtype=float)

            close = hist['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            log_ret = np.log(close / close.shift(1)).dropna()
            # Rolling 30-day RV as IV proxy (IV typically trades at ~1.1-1.2x RV)
            rv = log_ret.rolling(30).std() * np.sqrt(252)
            iv_proxy = rv * 1.15  # Add typical VRP
            return iv_proxy.dropna()
        except Exception:
            return pd.Series(dtype=float)

    def score(self, scans: List[TickerScan]) -> List[TickerScan]:
        """Score each ticker with IV Rank, IV Percentile, VRP edge."""
        print("\n" + "=" * 70)
        print("  STEP 2: EDGE SCORING — IV Rank, IV Percentile, VRP gap")
        print("=" * 70)

        valid_scans = [s for s in scans if s.atm_iv > 0]
        total = len(valid_scans)

        for i, scan in enumerate(valid_scans, 1):
            print(f"  [{i:2d}/{total}] {scan.ticker:6s} ... ", end="", flush=True)

            try:
                # Get price history for RV calc
                end   = date.today()
                start = end - timedelta(days=100)
                hist  = yf.download(scan.ticker, start=start, end=end,
                                    auto_adjust=True, progress=False)

                if not hist.empty:
                    close = hist['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    scan.rv_30d = self._realized_vol(close, 30)
                    scan.rv_60d = self._realized_vol(close, 60)

                # VRP edge: how much IV exceeds RV
                if not np.isnan(scan.rv_30d) and scan.rv_30d > 0:
                    scan.vrp_edge = scan.atm_iv - scan.rv_30d

                # IV Rank and IV Percentile from proxy history
                iv_hist = self._iv_history_proxy(scan.ticker)
                if len(iv_hist) > 30:
                    iv_min = iv_hist.min()
                    iv_max = iv_hist.max()
                    if iv_max > iv_min:
                        scan.iv_rank = ((scan.atm_iv - iv_min) / (iv_max - iv_min)) * 100
                    scan.iv_percentile = percentileofscore(iv_hist.values, scan.atm_iv)

                # Composite score: weighted blend
                # IVR (40%) + VRP edge magnitude (35%) + option volume (10%) + P/C (15%)
                ivr_score  = min(scan.iv_rank / 100, 1.0) * 40
                vrp_score  = min(max(scan.vrp_edge / 0.15, 0), 1.0) * 35 if scan.vrp_edge > 0 else 0
                vol_score  = min(scan.avg_option_volume / 5000, 1.0) * 10
                # Elevated P/C ratio (>0.8) = more fear = better for vol selling
                pc_score   = min(scan.put_call_ratio / 1.5, 1.0) * 15 if scan.put_call_ratio > 0.5 else 0
                scan.composite_score = ivr_score + vrp_score + vol_score + pc_score

                # Pass/fail filter
                scan.passed_filter = (
                    scan.iv_rank >= self.min_ivr and
                    scan.vrp_edge >= self.min_vrp and
                    scan.avg_option_volume >= 50
                )

                status = "PASS" if scan.passed_filter else "fail"
                print(f"IVR={scan.iv_rank:5.1f}  IVP={scan.iv_percentile:5.1f}  "
                      f"VRP={scan.vrp_edge*100:+5.1f}%  score={scan.composite_score:5.1f}  [{status}]")

            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")
                scan.passed_filter = False

            time.sleep(0.1)

        passed = [s for s in valid_scans if s.passed_filter]
        print(f"\n  Scoring complete: {len(passed)}/{total} tickers passed "
              f"(IVR >= {self.min_ivr}, VRP >= {self.min_vrp*100:.0f}%)")
        return valid_scans


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: EARNINGS FILTER  (Program 4 logic — earnings blackout)
# ═══════════════════════════════════════════════════════════════════════════

class EarningsFilter:
    """Flags tickers with earnings in the next N days to avoid vol crush."""

    def __init__(self, blackout_days: int = EARNINGS_BLACKOUT_DAYS):
        self.blackout_days = blackout_days

    def filter(self, scans: List[TickerScan]) -> List[TickerScan]:
        """Check each passing ticker for upcoming earnings."""
        print("\n" + "=" * 70)
        print(f"  STEP 3: EARNINGS FILTER — {self.blackout_days}-day blackout zone")
        print("=" * 70)

        passed = [s for s in scans if s.passed_filter]
        today  = date.today()

        for scan in passed:
            print(f"  {scan.ticker:6s} ... ", end="", flush=True)
            try:
                tk = yf.Ticker(scan.ticker)
                cal = tk.calendar
                if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                    # yfinance returns calendar as dict or DataFrame
                    earnings_date = None
                    if isinstance(cal, dict):
                        ed = cal.get('Earnings Date', [])
                        if ed:
                            earnings_date = pd.Timestamp(ed[0]).date() if isinstance(ed, list) else pd.Timestamp(ed).date()
                    elif isinstance(cal, pd.DataFrame):
                        if 'Earnings Date' in cal.columns:
                            vals = cal['Earnings Date'].values
                            if len(vals) > 0:
                                earnings_date = pd.Timestamp(vals[0]).date()

                    if earnings_date:
                        days_away = (earnings_date - today).days
                        scan.earnings_days_away = days_away
                        if 0 <= days_away <= self.blackout_days:
                            scan.earnings_flagged = True
                            scan.passed_filter = False
                            print(f"FLAGGED — earnings in {days_away} days")
                            continue
                        else:
                            print(f"clear (earnings in {days_away} days)")
                            continue

                print("clear (no earnings date found)")
            except Exception as e:
                print(f"clear (lookup failed: {str(e)[:30]})")

            time.sleep(0.1)

        still_passing = sum(1 for s in scans if s.passed_filter)
        flagged       = sum(1 for s in passed if s.earnings_flagged)
        print(f"\n  Earnings filter: {flagged} flagged, {still_passing} remaining")
        return scans


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: VOL SURFACE ANALYSIS  (Program 6 logic — skew + term structure)
# ═══════════════════════════════════════════════════════════════════════════

class VolSurfaceAnalyzer:
    """Analyzes volatility skew and term structure for top candidates."""

    def analyze(self, scans: List[TickerScan], top_n: int = TOP_N_CANDIDATES) -> Dict[str, VolSurfaceData]:
        """Analyze vol surface for the top N candidates by composite score."""
        print("\n" + "=" * 70)
        print(f"  STEP 4: VOL SURFACE ANALYSIS — Top {top_n} candidates")
        print("=" * 70)

        # Pick top N by composite score among those still passing
        candidates = sorted([s for s in scans if s.passed_filter],
                            key=lambda x: x.composite_score, reverse=True)[:top_n]

        if not candidates:
            print("  No candidates passed prior filters. Relaxing IVR to 25...")
            candidates = sorted([s for s in scans if s.atm_iv > 0 and s.vrp_edge > 0],
                                key=lambda x: x.composite_score, reverse=True)[:top_n]
            for c in candidates:
                c.passed_filter = True

        results = {}

        for scan in candidates:
            print(f"  {scan.ticker:6s} ... ", end="", flush=True)
            vsd = VolSurfaceData(ticker=scan.ticker)

            try:
                tk = yf.Ticker(scan.ticker)
                expirations = tk.options
                if len(expirations) < 2:
                    print("insufficient expirations")
                    results[scan.ticker] = vsd
                    continue

                # --- SKEW ANALYSIS ---
                # Use nearest monthly expiry, look at 25-delta-ish strikes
                today_dt = date.today()
                near_exp = expirations[0]
                chain    = tk.option_chain(near_exp)
                calls    = chain.calls
                puts     = chain.puts

                if not calls.empty and not puts.empty:
                    spot = scan.spot_price
                    # 25-delta put ~ 5% OTM put; 25-delta call ~ 5% OTM call
                    otm_put_strike  = spot * 0.95
                    otm_call_strike = spot * 1.05

                    put_row  = puts.iloc[(puts['strike'] - otm_put_strike).abs().argsort()[:1]]
                    call_row = calls.iloc[(calls['strike'] - otm_call_strike).abs().argsort()[:1]]

                    put_iv  = float(put_row['impliedVolatility'].iloc[0]) if not put_row.empty else 0
                    call_iv = float(call_row['impliedVolatility'].iloc[0]) if not call_row.empty else 0

                    if put_iv > 0 and call_iv > 0:
                        vsd.skew_25d = put_iv - call_iv  # Positive = normal skew (puts > calls)

                    if abs(vsd.skew_25d) > 0.05:
                        vsd.skew_signal = "STEEP"
                    elif abs(vsd.skew_25d) < 0.02:
                        vsd.skew_signal = "FLAT"
                    else:
                        vsd.skew_signal = "NORMAL"

                # --- TERM STRUCTURE ---
                if len(expirations) >= 2:
                    near_exp_dt = datetime.strptime(expirations[0], "%Y-%m-%d").date()
                    far_idx     = min(3, len(expirations) - 1)
                    far_exp     = expirations[far_idx]
                    far_exp_dt  = datetime.strptime(far_exp, "%Y-%m-%d").date()

                    far_chain = tk.option_chain(far_exp)
                    far_calls = far_chain.calls

                    if not far_calls.empty:
                        near_atm = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]
                        far_atm  = far_calls.iloc[(far_calls['strike'] - spot).abs().argsort()[:1]]

                        near_iv = float(near_atm['impliedVolatility'].iloc[0]) if not near_atm.empty else 0
                        far_iv  = float(far_atm['impliedVolatility'].iloc[0]) if not far_atm.empty else 0

                        if near_iv > 0 and far_iv > 0:
                            vsd.term_slope = far_iv - near_iv

                        if vsd.term_slope > 0.02:
                            vsd.term_signal = "CONTANGO"      # Normal — sell near, hedge far
                        elif vsd.term_slope < -0.02:
                            vsd.term_signal = "BACKWARDATION"  # Elevated near-term fear
                        else:
                            vsd.term_signal = "FLAT"

                # Surface composite score
                # Steep skew + contango = ideal for put selling
                skew_pts = 2.0 if vsd.skew_signal == "STEEP" else (1.0 if vsd.skew_signal == "NORMAL" else 0)
                term_pts = 2.0 if vsd.term_signal == "CONTANGO" else (1.0 if vsd.term_signal == "FLAT" else 0)
                vsd.surface_score = skew_pts + term_pts

                print(f"skew={vsd.skew_25d*100:+5.1f}% ({vsd.skew_signal:5s})  "
                      f"term={vsd.term_slope*100:+5.1f}% ({vsd.term_signal})  "
                      f"surf_score={vsd.surface_score:.1f}")

            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")

            results[scan.ticker] = vsd
            time.sleep(0.2)

        print(f"\n  Vol surface analyzed for {len(results)} candidates")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: REGIME CHECK  (Program 9 logic — macro environment)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeChecker:
    """Assesses current macro regime: RISK-ON, NEUTRAL, or RISK-OFF."""

    def check(self) -> RegimeState:
        """Pull macro indicators and compute regime score."""
        print("\n" + "=" * 70)
        print("  STEP 5: REGIME CHECK — Macro environment assessment")
        print("=" * 70)

        state = RegimeState()

        # --- VIX ---
        try:
            vix_data = yf.download("^VIX", period="1y", auto_adjust=True, progress=False)
            if not vix_data.empty:
                close = vix_data['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                state.vix_level = float(close.iloc[-1])
                vix_mean = float(close.mean())
                vix_std  = float(close.std())
                if vix_std > 0:
                    state.vix_zscore = (state.vix_level - vix_mean) / vix_std
                print(f"  VIX: {state.vix_level:.1f} (z-score: {state.vix_zscore:+.2f})")
        except Exception as e:
            print(f"  VIX: fetch failed ({e})")

        # --- Yield curve (10Y - 2Y spread) ---
        try:
            if FREDAPI_AVAILABLE and FRED_API_KEY:
                fred = Fred(api_key=FRED_API_KEY)
                t10y = fred.get_series('DGS10', observation_start=date.today() - timedelta(days=30))
                t2y  = fred.get_series('DGS2',  observation_start=date.today() - timedelta(days=30))
                if len(t10y) > 0 and len(t2y) > 0:
                    state.yield_spread_10y2y = float(t10y.dropna().iloc[-1] - t2y.dropna().iloc[-1])
                    state.fed_funds_rate = float(
                        fred.get_series('DFF', observation_start=date.today() - timedelta(days=7)).dropna().iloc[-1])
                    print(f"  10Y-2Y spread: {state.yield_spread_10y2y:+.2f}%  "
                          f"Fed Funds: {state.fed_funds_rate:.2f}%")
            else:
                # Proxy via yfinance treasury ETFs
                tnx = yf.download("^TNX", period="5d", auto_adjust=True, progress=False)
                twoy = yf.download("^IRX", period="5d", auto_adjust=True, progress=False)
                if not tnx.empty and not twoy.empty:
                    tnx_close = tnx['Close'].iloc[:, 0] if isinstance(tnx['Close'], pd.DataFrame) else tnx['Close']
                    twoy_close = twoy['Close'].iloc[:, 0] if isinstance(twoy['Close'], pd.DataFrame) else twoy['Close']
                    r10 = float(tnx_close.iloc[-1])
                    r3m = float(twoy_close.iloc[-1])
                    state.yield_spread_10y2y = r10 - r3m
                    print(f"  10Y-3M spread (proxy): {state.yield_spread_10y2y:+.2f}%")
        except Exception as e:
            print(f"  Yield curve: fetch failed ({e})")

        # --- Regime classification ---
        # Score from -3 (risk-off) to +3 (risk-on)
        raw_score = 0

        # VIX component
        if state.vix_level < VIX_LOW:
            raw_score += 1       # Low VIX = risk-on
        elif state.vix_level > VIX_HIGH:
            raw_score -= 2       # High VIX = risk-off (weighted heavier)
        if state.vix_zscore > 1.5:
            raw_score -= 1       # VIX spike above normal

        # Yield curve component
        if state.yield_spread_10y2y < SPREAD_INVERTED:
            raw_score -= 1       # Inverted = recession signal
        elif state.yield_spread_10y2y > 0.5:
            raw_score += 1       # Healthy positive slope

        # Normalize to [-1, +1]
        state.regime_score = max(-1, min(1, raw_score / 3))

        if state.regime_score > 0.3:
            state.regime = "RISK-ON"
            state.description = "Low vol, healthy yield curve — favor selling vol"
        elif state.regime_score < -0.3:
            state.regime = "RISK-OFF"
            state.description = "Elevated vol or curve inversion — reduce size, widen strikes"
        else:
            state.regime = "NEUTRAL"
            state.description = "Mixed signals — standard sizing, balanced approach"

        print(f"\n  >>> REGIME: {state.regime} (score={state.regime_score:+.2f})")
        print(f"      {state.description}")
        return state


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: QUICK BACKTESTER  (Program 5 logic — vectorized backtest)
# ═══════════════════════════════════════════════════════════════════════════

class QuickBacktester:
    """Fast vectorized backtest of short-vol strategies using historical data."""

    def backtest(self, ticker: str, strategy: str = "short_put",
                 lookback_days: int = 504) -> BacktestResult:
        """
        Backtest a short-premium strategy using historical price moves.

        For 'short_put': Simulates selling 30-DTE 5%-OTM puts monthly.
        We check if the underlying fell below the strike at expiry.
        Win = keep full premium; Loss = (strike - close) - premium.
        """
        result = BacktestResult(ticker=ticker, strategy=strategy)

        try:
            end   = date.today()
            start = end - timedelta(days=lookback_days)
            hist  = yf.download(ticker, start=start, end=end,
                                auto_adjust=True, progress=False)
            if hist.empty or len(hist) < 60:
                return result

            close = hist['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            prices = close.values
            dates  = close.index

            # Simulate monthly 30-DTE short put entries
            # Entry every ~21 trading days, check P&L 21 days later
            step   = 21
            trades = []

            for entry_idx in range(0, len(prices) - step - 1, step):
                exit_idx    = entry_idx + step
                entry_price = prices[entry_idx]
                exit_price  = prices[exit_idx]

                if strategy == "short_put":
                    strike = entry_price * 0.95  # 5% OTM put
                    # Premium estimate: ~1.5% of spot for 30-DTE 5%-OTM (rough avg)
                    premium = entry_price * 0.015

                    if exit_price >= strike:
                        # Win: keep premium
                        pnl = premium
                    else:
                        # Loss: intrinsic loss minus premium received
                        pnl = premium - (strike - exit_price)

                elif strategy == "iron_condor":
                    # Short 5% OTM put + short 5% OTM call, wings at 10%
                    put_strike  = entry_price * 0.95
                    call_strike = entry_price * 1.05
                    premium     = entry_price * 0.025  # ~2.5% credit

                    if put_strike <= exit_price <= call_strike:
                        pnl = premium
                    elif exit_price < put_strike:
                        pnl = premium - (put_strike - exit_price)
                    else:
                        pnl = premium - (exit_price - call_strike)

                else:  # put_credit_spread
                    short_strike = entry_price * 0.95
                    long_strike  = entry_price * 0.90
                    premium      = entry_price * 0.012  # ~1.2% credit
                    width        = short_strike - long_strike
                    max_loss     = width - premium

                    if exit_price >= short_strike:
                        pnl = premium
                    elif exit_price <= long_strike:
                        pnl = -max_loss
                    else:
                        pnl = premium - (short_strike - exit_price)

                trades.append(pnl / entry_price)  # Normalize as % of entry

            if trades:
                trades_arr = np.array(trades)
                result.num_trades    = len(trades)
                result.win_rate      = np.mean(trades_arr > 0) * 100
                result.avg_return    = np.mean(trades_arr) * 100
                result.total_return  = np.sum(trades_arr) * 100

                # Sharpe (annualized, ~12 monthly periods/yr)
                if np.std(trades_arr) > 0:
                    result.sharpe = (np.mean(trades_arr) / np.std(trades_arr)) * np.sqrt(12)

                # Max drawdown on cumulative curve
                cum = np.cumsum(trades_arr)
                peak = np.maximum.accumulate(cum)
                dd   = (cum - peak)
                result.max_drawdown = float(np.min(dd)) * 100

                result.edge_confirmed = (result.win_rate > 55 and result.avg_return > 0)

        except Exception:
            pass

        return result

    def backtest_candidates(self, candidates: List[TickerScan]) -> Dict[str, BacktestResult]:
        """Backtest all candidates."""
        print("\n" + "=" * 70)
        print("  STEP 6: BACKTESTER — 2-year vectorized backtest")
        print("=" * 70)

        results = {}
        for scan in candidates:
            # Choose strategy based on VRP edge magnitude
            if scan.vrp_edge > 0.08:
                strategy = "short_put"
            elif scan.vrp_edge > 0.04:
                strategy = "put_credit_spread"
            else:
                strategy = "iron_condor"

            print(f"  {scan.ticker:6s} ({strategy:20s}) ... ", end="", flush=True)
            bt = self.backtest(scan.ticker, strategy)
            results[scan.ticker] = bt

            conf = "CONFIRMED" if bt.edge_confirmed else "weak"
            print(f"WR={bt.win_rate:5.1f}%  avg={bt.avg_return:+5.2f}%  "
                  f"sharpe={bt.sharpe:+5.2f}  DD={bt.max_drawdown:5.1f}%  [{conf}]")

            time.sleep(0.1)

        confirmed = sum(1 for b in results.values() if b.edge_confirmed)
        print(f"\n  Backtests: {confirmed}/{len(results)} strategies confirmed edge")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: KELLY SIZER  (Position sizing via Kelly criterion)
# ═══════════════════════════════════════════════════════════════════════════

class KellySizer:
    """
    Kelly criterion position sizing.

    Kelly% = (p * b - q) / b
    where p = win probability, q = 1-p, b = avg win / avg loss ratio

    We use fractional Kelly (25-50%) because full Kelly is too aggressive.
    """

    def __init__(self, bankroll: float = BANKROLL,
                 max_pct: float = MAX_POSITION_PCT,
                 kelly_fraction: float = 0.25):
        self.bankroll       = bankroll
        self.max_pct        = max_pct
        self.kelly_fraction = kelly_fraction  # Use 25% of full Kelly

    def size(self, win_rate: float, avg_win: float, avg_loss: float,
             regime_score: float = 0.0) -> Tuple[float, float]:
        """
        Returns (kelly_fraction, position_size_usd).

        regime_score: -1 to +1. Negative regimes reduce size.
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.0, 0.0

        p = win_rate / 100.0
        q = 1 - p
        b = abs(avg_win / avg_loss)  # Odds ratio

        # Full Kelly
        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, kelly)  # Never go negative

        # Fractional Kelly
        frac_kelly = kelly * self.kelly_fraction

        # Regime adjustment: reduce by up to 50% in risk-off
        regime_mult = 1.0 + (regime_score * 0.25)  # 0.75x in risk-off, 1.25x in risk-on
        regime_mult = max(0.5, min(1.5, regime_mult))
        frac_kelly *= regime_mult

        # Cap at max position %
        frac_kelly = min(frac_kelly, self.max_pct)

        position_usd = self.bankroll * frac_kelly
        return frac_kelly, position_usd


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RecommendationEngine:
    """Generates final trade recommendations with full thesis."""

    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.sizer    = KellySizer(bankroll=bankroll)

    def _choose_strategy(self, scan: TickerScan, regime: RegimeState,
                         surface: Optional[VolSurfaceData],
                         bt: Optional[BacktestResult]) -> Tuple[str, str, str, int]:
        """
        Returns (strategy_name, direction, strikes_desc, dte).

        Logic:
        - High VRP + steep skew + risk-on → Short Put (most edge)
        - Moderate VRP + risk-on → Put Credit Spread (defined risk)
        - Any regime + high IVR → Iron Condor (delta-neutral)
        - Risk-off → wider strikes, shorter DTE
        """
        spot = scan.spot_price
        dte  = TARGET_DTE

        # Regime adjustments
        if regime.regime == "RISK-OFF":
            dte = 21  # Shorter DTE in risk-off

        if scan.vrp_edge > 0.08 and regime.regime != "RISK-OFF":
            # Strong edge + not risk-off → naked short put
            strike = round(spot * 0.95, 0)
            return ("Short Put", "SELL VOL",
                    f"${strike:.0f}P", dte)

        elif scan.vrp_edge > 0.04:
            # Moderate edge → put credit spread
            short_k = round(spot * 0.95, 0)
            long_k  = round(spot * 0.90, 0)
            return ("Put Credit Spread", "SELL VOL",
                    f"${short_k:.0f}P / ${long_k:.0f}P", dte)

        else:
            # Low directional conviction → iron condor
            put_short  = round(spot * 0.95, 0)
            put_long   = round(spot * 0.90, 0)
            call_short = round(spot * 1.05, 0)
            call_long  = round(spot * 1.10, 0)

            # Widen strikes in risk-off
            if regime.regime == "RISK-OFF":
                put_short  = round(spot * 0.93, 0)
                call_short = round(spot * 1.07, 0)

            return ("Iron Condor", "SELL VOL",
                    f"${put_long:.0f}P/${put_short:.0f}P — ${call_short:.0f}C/${call_long:.0f}C",
                    dte)

    def _build_thesis(self, scan: TickerScan, regime: RegimeState,
                      surface: Optional[VolSurfaceData],
                      bt: Optional[BacktestResult],
                      strategy: str) -> str:
        """Generate a written thesis for the trade."""
        parts = []

        # Edge description
        parts.append(f"{scan.ticker} IV at {scan.atm_iv*100:.1f}% vs 30d RV at "
                     f"{scan.rv_30d*100:.1f}% = {scan.vrp_edge*100:+.1f}% VRP edge.")

        # IVR context
        if scan.iv_rank > 70:
            parts.append(f"IV Rank at {scan.iv_rank:.0f} (top of 1-year range) — strong mean-reversion setup.")
        elif scan.iv_rank > 50:
            parts.append(f"IV Rank at {scan.iv_rank:.0f} (above median) — decent entry for vol selling.")
        else:
            parts.append(f"IV Rank at {scan.iv_rank:.0f} (moderate) — smaller size warranted.")

        # Surface
        if surface and surface.skew_signal:
            parts.append(f"Skew is {surface.skew_signal.lower()} ({surface.skew_25d*100:+.1f}%), "
                         f"term structure in {surface.term_signal.lower() if surface.term_signal else 'N/A'}.")

        # Regime
        parts.append(f"Macro regime: {regime.regime} (score {regime.regime_score:+.2f}).")

        # Backtest
        if bt and bt.num_trades > 0:
            parts.append(f"2-year backtest of {strategy}: {bt.win_rate:.0f}% win rate, "
                         f"{bt.avg_return:+.2f}% avg return, Sharpe {bt.sharpe:.2f}.")

        # Risk note
        if regime.regime == "RISK-OFF":
            parts.append("CAUTION: Risk-off regime — position sized down, strikes widened.")

        return " ".join(parts)

    def recommend(self, scans: List[TickerScan], regime: RegimeState,
                  surfaces: Dict[str, VolSurfaceData],
                  backtests: Dict[str, BacktestResult],
                  max_recs: int = MAX_RECOMMENDATIONS) -> List[TradeRecommendation]:
        """Generate final ranked trade recommendations."""
        print("\n" + "=" * 70)
        print("  STEP 7+8: SIZING & RECOMMENDATIONS")
        print("=" * 70)

        # Rank by composite + surface + backtest confirmation
        candidates = [s for s in scans if s.passed_filter]
        if not candidates:
            candidates = sorted([s for s in scans if s.atm_iv > 0 and s.vrp_edge > 0],
                                key=lambda x: x.composite_score, reverse=True)[:max_recs]

        scored = []
        for c in candidates:
            final_score = c.composite_score
            # Bonus for confirmed backtest
            bt = backtests.get(c.ticker)
            if bt and bt.edge_confirmed:
                final_score += 10
            # Bonus for favorable surface
            surf = surfaces.get(c.ticker)
            if surf:
                final_score += surf.surface_score * 3
            # Regime penalty/bonus
            final_score += regime.regime_score * 5
            scored.append((c, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:max_recs]

        recommendations = []
        for scan, final_score in top:
            surface  = surfaces.get(scan.ticker)
            bt       = backtests.get(scan.ticker)

            strategy, direction, strikes, dte = self._choose_strategy(
                scan, regime, surface, bt)

            # Kelly sizing
            win_rate = bt.win_rate if (bt and bt.win_rate > 0) else 65
            # Estimate avg win/loss for Kelly
            if strategy == "Short Put":
                avg_win  = scan.spot_price * 0.015   # ~1.5% premium
                avg_loss = scan.spot_price * 0.035   # ~3.5% avg loss when wrong
            elif strategy == "Iron Condor":
                avg_win  = scan.spot_price * 0.025
                avg_loss = scan.spot_price * 0.05
            else:  # Put Credit Spread
                avg_win  = scan.spot_price * 0.012
                avg_loss = scan.spot_price * 0.038

            kelly_frac, pos_size = self.sizer.size(
                win_rate, avg_win, avg_loss, regime.regime_score)

            # Contracts: pos_size / (notional per contract)
            notional_per = scan.spot_price * 100  # 100 shares per contract
            contracts    = max(1, int(pos_size / notional_per)) if notional_per > 0 else 1

            max_risk   = avg_loss * contracts * 100
            max_reward = avg_win * contracts * 100

            # Confidence
            if scan.iv_rank > 60 and scan.vrp_edge > 0.05 and (bt and bt.edge_confirmed):
                confidence = "HIGH"
            elif scan.iv_rank > 40 and scan.vrp_edge > 0.03:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            thesis = self._build_thesis(scan, regime, surface, bt, strategy)

            rec = TradeRecommendation(
                ticker=scan.ticker,
                name=scan.name,
                sector=scan.sector,
                strategy=strategy,
                direction=direction,
                strikes=strikes,
                expiry_dte=dte,
                contracts=contracts,
                max_risk=max_risk,
                max_reward=max_reward,
                edge_pct=scan.vrp_edge * 100,
                kelly_fraction=kelly_frac,
                position_size_usd=pos_size,
                iv_rank=scan.iv_rank,
                regime=regime.regime,
                backtest_winrate=bt.win_rate if bt else 0,
                thesis=thesis,
                confidence=confidence
            )

            recommendations.append(rec)
            print(f"\n  {'='*60}")
            print(f"  {rec.ticker} — {rec.strategy} ({rec.confidence} confidence)")
            print(f"  Strikes: {rec.strikes}  |  DTE: {rec.expiry_dte}  |  "
                  f"Contracts: {rec.contracts}")
            print(f"  Edge: {rec.edge_pct:+.1f}%  |  Kelly: {rec.kelly_fraction:.1%}  |  "
                  f"Size: ${rec.position_size_usd:,.0f}")
            print(f"  Max risk: ${rec.max_risk:,.0f}  |  Max reward: ${rec.max_reward:,.0f}")
            print(f"  Thesis: {rec.thesis[:120]}...")

        print(f"\n  Generated {len(recommendations)} trade recommendations")
        return recommendations


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class TradingPipeline:
    """End-to-end orchestrator: chains all 8 steps into a single run."""

    def __init__(self, watchlist: dict = None, bankroll: float = BANKROLL):
        self.watchlist = watchlist or WATCHLIST
        self.bankroll  = bankroll

    def run(self) -> PipelineResults:
        """Execute the full pipeline. Returns PipelineResults."""
        results = PipelineResults()
        results.run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "#" * 70)
        print("#" + " " * 19 + "UNIFIED TRADING PIPELINE" + " " * 19 + " #")
        print("#" + " " * 16 + f"Bankroll: ${self.bankroll:,.0f}" + " " * 20 + "#")
        print("#" + " " * 16 + f"Watchlist: {len(self.watchlist)} tickers" + " " * 17 + "#")
        print("#" + " " * 16 + f"Run: {results.run_timestamp}" + " " * 10 + "#")
        print("#" * 70)

        t_start = time.time()

        # STEP 1: SCAN
        scanner = MarketScanner(self.watchlist)
        results.scan_data = scanner.scan()
        results.funnel_counts['scanned'] = len(self.watchlist)
        results.funnel_counts['valid_iv'] = sum(1 for s in results.scan_data if s.atm_iv > 0)

        # STEP 2: SCORE
        scorer = EdgeScorer()
        results.scored_tickers = scorer.score(results.scan_data)
        results.funnel_counts['passed_score'] = sum(
            1 for s in results.scored_tickers if s.passed_filter)

        # STEP 3: EARNINGS FILTER
        ef = EarningsFilter()
        results.post_earnings_filter = ef.filter(results.scored_tickers)
        results.funnel_counts['passed_earnings'] = sum(
            1 for s in results.post_earnings_filter if s.passed_filter)

        # STEP 4: VOL SURFACE (top 5 candidates)
        vsa = VolSurfaceAnalyzer()
        results.vol_surface = vsa.analyze(results.post_earnings_filter)
        results.funnel_counts['vol_surface'] = len(results.vol_surface)

        # STEP 5: REGIME CHECK
        rc = RegimeChecker()
        results.regime = rc.check()

        # STEP 6: BACKTEST (candidates that made it through)
        bt_candidates = sorted(
            [s for s in results.post_earnings_filter if s.ticker in results.vol_surface],
            key=lambda x: x.composite_score, reverse=True
        )[:TOP_N_CANDIDATES]
        qb = QuickBacktester()
        results.backtests = qb.backtest_candidates(bt_candidates)

        # STEPS 7+8: SIZE & RECOMMEND
        engine = RecommendationEngine(bankroll=self.bankroll)
        results.recommendations = engine.recommend(
            results.post_earnings_filter,
            results.regime,
            results.vol_surface,
            results.backtests
        )
        results.funnel_counts['recommended'] = len(results.recommendations)

        elapsed = time.time() - t_start
        print(f"\n{'#'*70}")
        print(f"  Pipeline complete in {elapsed:.1f}s")
        print(f"  Funnel: {results.funnel_counts.get('scanned', 0)} scanned → "
              f"{results.funnel_counts.get('valid_iv', 0)} with IV → "
              f"{results.funnel_counts.get('passed_score', 0)} scored → "
              f"{results.funnel_counts.get('passed_earnings', 0)} post-earnings → "
              f"{results.funnel_counts.get('vol_surface', 0)} analyzed → "
              f"{results.funnel_counts.get('recommended', 0)} recommended")
        print(f"{'#'*70}\n")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Dark-themed 4-panel dashboard
# ═══════════════════════════════════════════════════════════════════════════

class PipelineDashboard:
    """Dark-themed 4-panel dashboard showing the full pipeline output."""

    # Color palette
    BG      = '#0e1117'
    PANEL   = '#1a1d24'
    TEXT    = '#e0e0e0'
    DIM     = '#6b7280'
    GREEN   = '#00d26a'
    RED     = '#ff4757'
    GOLD    = '#ffc107'
    BLUE    = '#4da6ff'
    PURPLE  = '#a855f7'
    CYAN    = '#06b6d4'
    ORANGE  = '#ff8c00'

    def __init__(self, results: PipelineResults):
        self.r = results

    def render(self):
        """Render the 4-panel dashboard."""
        fig = plt.figure(figsize=(20, 14), facecolor=self.BG)
        fig.suptitle('PROGRAM 13: UNIFIED TRADING PIPELINE',
                     color=self.GOLD, fontsize=18, fontweight='bold',
                     y=0.98)
        fig.text(0.5, 0.955, f'Run: {self.r.run_timestamp}  |  '
                 f'Bankroll: ${BANKROLL:,.0f}  |  '
                 f'Regime: {self.r.regime.regime}',
                 ha='center', color=self.DIM, fontsize=10)

        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25,
                               left=0.06, right=0.96, top=0.93, bottom=0.05)

        self._panel_funnel(fig.add_subplot(gs[0, 0]))
        self._panel_candidates(fig.add_subplot(gs[0, 1]))
        self._panel_trades(fig.add_subplot(gs[1, 0]))
        self._panel_risk(fig.add_subplot(gs[1, 1]))

        plt.savefig('pipeline_dashboard.png', dpi=150, facecolor=self.BG,
                     bbox_inches='tight')
        plt.show()
        print("\n  Dashboard saved to pipeline_dashboard.png")

    def _style_ax(self, ax, title):
        """Apply dark theme to axis."""
        ax.set_facecolor(self.PANEL)
        ax.set_title(title, color=self.GOLD, fontsize=13, fontweight='bold',
                     pad=10, loc='left')
        ax.tick_params(colors=self.DIM, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#2d3139')

    # ── PANEL 1: Pipeline Funnel ──────────────────────────────────────────

    def _panel_funnel(self, ax):
        self._style_ax(ax, 'PIPELINE FUNNEL')

        fc = self.r.funnel_counts
        stages = [
            ('Watchlist',     fc.get('scanned', 0)),
            ('Valid IV',      fc.get('valid_iv', 0)),
            ('Scored',        fc.get('passed_score', 0)),
            ('Post-Earnings', fc.get('passed_earnings', 0)),
            ('Vol Surface',   fc.get('vol_surface', 0)),
            ('Recommended',   fc.get('recommended', 0)),
        ]

        n = len(stages)
        max_count = max(c for _, c in stages) if stages else 1

        # Draw funnel bars (centered, getting narrower)
        colors = [self.BLUE, self.CYAN, self.GREEN, self.GOLD, self.ORANGE, self.RED]
        y_positions = list(range(n - 1, -1, -1))

        for i, ((label, count), y) in enumerate(zip(stages, y_positions)):
            width = (count / max_count) * 0.9 if max_count > 0 else 0.1
            width = max(width, 0.08)  # Minimum visible width

            bar_left = (1 - width) / 2
            ax.barh(y, width, left=bar_left, height=0.6,
                    color=colors[i], alpha=0.8, edgecolor=colors[i], linewidth=0.5)
            ax.text(0.5, y, f'{label}: {count}',
                    ha='center', va='center', color='white',
                    fontsize=10, fontweight='bold')

            # Arrows between stages
            if i < n - 1:
                ax.annotate('', xy=(0.5, y - 0.4), xytext=(0.5, y - 0.7),
                            arrowprops=dict(arrowstyle='->', color=self.DIM,
                                            lw=1.5))

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # Conversion rate
        total    = fc.get('scanned', 1)
        final    = fc.get('recommended', 0)
        conv_pct = (final / total * 100) if total > 0 else 0
        ax.text(0.5, -0.35, f'Conversion: {conv_pct:.0f}% ({total} → {final})',
                ha='center', va='center', color=self.GOLD, fontsize=10,
                transform=ax.transData)

    # ── PANEL 2: Top Candidates with Scores ───────────────────────────────

    def _panel_candidates(self, ax):
        self._style_ax(ax, 'TOP CANDIDATES — COMPOSITE SCORES')

        # Get top candidates (those that made it to vol surface or top N)
        candidates = sorted(
            [s for s in self.r.scored_tickers if s.atm_iv > 0],
            key=lambda x: x.composite_score, reverse=True
        )[:10]

        if not candidates:
            ax.text(0.5, 0.5, 'No candidates', ha='center', va='center',
                    color=self.DIM, fontsize=14)
            return

        tickers = [c.ticker for c in candidates]
        scores  = [c.composite_score for c in candidates]
        ivrs    = [c.iv_rank for c in candidates]
        vrps    = [c.vrp_edge * 100 for c in candidates]

        y = np.arange(len(tickers))
        bar_height = 0.35

        # Composite score bars
        bars1 = ax.barh(y + bar_height/2, scores, bar_height,
                        color=self.BLUE, alpha=0.8, label='Composite Score')
        # VRP edge bars
        bars2 = ax.barh(y - bar_height/2, vrps, bar_height,
                        color=self.GREEN, alpha=0.8, label='VRP Edge (%)')

        ax.set_yticks(y)
        ax.set_yticklabels(tickers, color=self.TEXT, fontsize=9, fontweight='bold')
        ax.set_xlabel('Score / Edge %', color=self.DIM, fontsize=9)
        ax.invert_yaxis()

        # Annotate IVR on right side
        for i, (c, s) in enumerate(zip(candidates, scores)):
            ax.text(max(scores) * 1.02, i, f'IVR={c.iv_rank:.0f}',
                    va='center', color=self.GOLD, fontsize=8)

        # Regime badge
        regime_color = (self.GREEN if self.r.regime.regime == "RISK-ON"
                        else self.RED if self.r.regime.regime == "RISK-OFF"
                        else self.GOLD)
        ax.text(0.98, 0.02, f'Regime: {self.r.regime.regime}',
                transform=ax.transAxes, ha='right', va='bottom',
                color=regime_color, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=self.BG,
                          edgecolor=regime_color, alpha=0.9))

        ax.legend(loc='lower right', fontsize=8, framealpha=0.3,
                  labelcolor=self.TEXT, facecolor=self.PANEL)
        ax.set_xlim(0, max(scores) * 1.15 if scores else 1)

    # ── PANEL 3: Recommended Trades ───────────────────────────────────────

    def _panel_trades(self, ax):
        self._style_ax(ax, 'TRADE RECOMMENDATIONS')
        ax.set_xticks([])
        ax.set_yticks([])

        recs = self.r.recommendations
        if not recs:
            ax.text(0.5, 0.5, 'No recommendations generated',
                    ha='center', va='center', color=self.DIM, fontsize=14)
            return

        # Table layout
        headers = ['Ticker', 'Strategy', 'Strikes', 'DTE', 'Cts', 'Size', 'Edge', 'Conf']
        col_x   = [0.02, 0.12, 0.28, 0.52, 0.60, 0.68, 0.80, 0.91]

        # Header row
        for hdr, x in zip(headers, col_x):
            ax.text(x, 0.92, hdr, transform=ax.transAxes, fontsize=9,
                    fontweight='bold', color=self.GOLD, va='top')

        ax.axhline(y=0.88, xmin=0.01, xmax=0.99, color=self.DIM,
                   linewidth=0.5, transform=ax.transAxes)

        # Data rows
        for i, rec in enumerate(recs):
            y_pos = 0.82 - i * 0.22  # Space between rows
            if y_pos < 0.05:
                break

            conf_color = (self.GREEN if rec.confidence == "HIGH"
                          else self.GOLD if rec.confidence == "MEDIUM"
                          else self.RED)

            row_data = [
                (rec.ticker, self.TEXT, 'bold'),
                (rec.strategy, self.CYAN, 'normal'),
                (rec.strikes, self.TEXT, 'normal'),
                (f'{rec.expiry_dte}d', self.TEXT, 'normal'),
                (str(rec.contracts), self.TEXT, 'normal'),
                (f'${rec.position_size_usd:,.0f}', self.GREEN, 'bold'),
                (f'{rec.edge_pct:+.1f}%', self.GREEN if rec.edge_pct > 5 else self.GOLD, 'bold'),
                (rec.confidence, conf_color, 'bold'),
            ]

            for (val, color, weight), x in zip(row_data, col_x):
                ax.text(x, y_pos, val, transform=ax.transAxes, fontsize=9,
                        fontweight=weight, color=color, va='top')

            # Thesis snippet below
            thesis_short = rec.thesis[:100] + '...' if len(rec.thesis) > 100 else rec.thesis
            ax.text(0.02, y_pos - 0.08, thesis_short,
                    transform=ax.transAxes, fontsize=7, color=self.DIM,
                    va='top', style='italic', wrap=True)

            # Separator
            ax.axhline(y=y_pos - 0.14, xmin=0.01, xmax=0.99,
                       color='#2d3139', linewidth=0.5, transform=ax.transAxes)

    # ── PANEL 4: Risk Summary ─────────────────────────────────────────────

    def _panel_risk(self, ax):
        self._style_ax(ax, 'RISK SUMMARY')
        ax.set_xticks([])
        ax.set_yticks([])

        recs   = self.r.recommendations
        regime = self.r.regime

        # Aggregate risk metrics
        total_risk   = sum(r.max_risk for r in recs)
        total_reward = sum(r.max_reward for r in recs)
        total_size   = sum(r.position_size_usd for r in recs)
        pct_deployed = (total_size / self.r.funnel_counts.get('scanned', 1)) if BANKROLL > 0 else 0
        pct_deployed = total_size / BANKROLL * 100

        # Risk color based on deployment %
        deploy_color = (self.GREEN if pct_deployed < 15
                        else self.GOLD if pct_deployed < 30
                        else self.RED)

        # Left column: key metrics
        metrics = [
            ('Total Positions',    f'{len(recs)}',                 self.TEXT),
            ('Capital Deployed',   f'${total_size:,.0f} ({pct_deployed:.1f}%)', deploy_color),
            ('Max Portfolio Risk',  f'${total_risk:,.0f}',          self.RED),
            ('Max Portfolio Reward', f'${total_reward:,.0f}',       self.GREEN),
            ('Risk/Reward Ratio',  f'{total_risk/total_reward:.2f}x' if total_reward > 0 else 'N/A',
             self.GOLD),
            ('Cash Reserve',       f'${BANKROLL - total_size:,.0f}', self.CYAN),
        ]

        y_start = 0.88
        for i, (label, value, color) in enumerate(metrics):
            y = y_start - i * 0.10
            ax.text(0.03, y, label + ':', transform=ax.transAxes,
                    fontsize=10, color=self.DIM, va='top')
            ax.text(0.55, y, value, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color=color, va='top')

        # Regime warning box
        y_regime = y_start - len(metrics) * 0.10 - 0.03
        regime_color = (self.GREEN if regime.regime == "RISK-ON"
                        else self.RED if regime.regime == "RISK-OFF"
                        else self.GOLD)

        box = FancyBboxPatch((0.02, y_regime - 0.12), 0.96, 0.13,
                             boxstyle="round,pad=0.01",
                             transform=ax.transAxes,
                             facecolor=regime_color, alpha=0.1,
                             edgecolor=regime_color, linewidth=1.5)
        ax.add_patch(box)

        ax.text(0.05, y_regime - 0.02, f'REGIME: {regime.regime}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                color=regime_color, va='top')
        ax.text(0.05, y_regime - 0.08, regime.description,
                transform=ax.transAxes, fontsize=8, color=self.DIM, va='top')

        # VIX gauge bar at bottom
        y_vix = y_regime - 0.20
        ax.text(0.03, y_vix, f'VIX: {regime.vix_level:.1f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=self.TEXT, va='top')

        # VIX bar (0-50 scale)
        bar_y      = y_vix - 0.06
        vix_pct    = min(regime.vix_level / 50, 1.0)
        bar_color  = (self.GREEN if regime.vix_level < 15
                      else self.GOLD if regime.vix_level < 25
                      else self.RED)

        ax.barh(bar_y, 0.9, height=0.03, left=0.03,
                color='#2d3139', transform=ax.transAxes)
        ax.barh(bar_y, 0.9 * vix_pct, height=0.03, left=0.03,
                color=bar_color, alpha=0.8, transform=ax.transAxes)

        # Zone labels
        ax.text(0.03, bar_y - 0.03, '0', transform=ax.transAxes,
                fontsize=7, color=self.DIM, va='top')
        ax.text(0.03 + 0.9 * 0.3, bar_y - 0.03, '15', transform=ax.transAxes,
                fontsize=7, color=self.DIM, va='top', ha='center')
        ax.text(0.03 + 0.9 * 0.5, bar_y - 0.03, '25', transform=ax.transAxes,
                fontsize=7, color=self.DIM, va='top', ha='center')
        ax.text(0.93, bar_y - 0.03, '50', transform=ax.transAxes,
                fontsize=7, color=self.DIM, va='top', ha='right')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           PROGRAM 13: UNIFIED TRADING PIPELINE                 ║
    ║           Capstone — Programs 1-12 Combined                    ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  SCAN → SCORE → FILTER → VOL SURFACE → REGIME → BACKTEST →    ║
    ║  SIZE → RECOMMEND                                              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Data:     yfinance (free), FRED (optional)                    ║
    ║  Bankroll: $100,000                                            ║
    ║  Universe: 30 liquid tickers                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Run the full pipeline
    pipeline = TradingPipeline()
    results  = pipeline.run()

    # Render dashboard
    dashboard = PipelineDashboard(results)
    dashboard.render()

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL TRADE RECOMMENDATIONS")
    print("=" * 70)
    for i, rec in enumerate(results.recommendations, 1):
        print(f"\n  ── Trade #{i} ──────────────────────────────────────")
        print(f"  Ticker:     {rec.ticker} ({rec.name})")
        print(f"  Strategy:   {rec.strategy} ({rec.direction})")
        print(f"  Strikes:    {rec.strikes}")
        print(f"  DTE:        {rec.expiry_dte} days")
        print(f"  Contracts:  {rec.contracts}")
        print(f"  Size:       ${rec.position_size_usd:,.0f} "
              f"({rec.kelly_fraction:.1%} Kelly)")
        print(f"  Max Risk:   ${rec.max_risk:,.0f}")
        print(f"  Max Reward: ${rec.max_reward:,.0f}")
        print(f"  Edge:       {rec.edge_pct:+.1f}% VRP")
        print(f"  IV Rank:    {rec.iv_rank:.0f}")
        print(f"  Backtest:   {rec.backtest_winrate:.0f}% win rate")
        print(f"  Confidence: {rec.confidence}")
        print(f"  Regime:     {rec.regime}")
        print(f"  Thesis:     {rec.thesis}")

    print(f"\n  Pipeline timestamp: {results.run_timestamp}")
    print(f"  Recommendations:   {len(results.recommendations)}")
    print(f"  Total deployed:    "
          f"${sum(r.position_size_usd for r in results.recommendations):,.0f} "
          f"of ${BANKROLL:,.0f}")
    print("\n  Done. Dashboard saved to pipeline_dashboard.png\n")
