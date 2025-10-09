# app.py
# =========================================================
# Crypto trading bot Backtest — Long Only (Stop Modes, Multi-Exchange)
# Fonte dati: ccxt (Binance/Bybit/OKX/Kraken/Coinbase), Binance REST o Sintetico
# Import/Export profili JSON (applicati come default al prossimo run)
# Stop mode: Percentuale | ATR | Minimo(Percentuale, ATR)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# -------------------------
# CCXT opzionale
# -------------------------
try:
    import ccxt
    HAS_CCXT = True
except Exception:
    HAS_CCXT = False

st.set_page_config(page_title="Crypto trading bot Backtest", layout="wide")

# -------------------------
# Default config (completo)
# -------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Base
    "symbol": "ETH/USDT",
    "timeframe": "1h",
    "lookback": 1000,

    # Money management
    "initial_capital": 1000.0,
    "risk_per_trade_pct": 0.02,
    "fee_pct": 0.001,
    "slippage_pct": 0.0005,

    # Stops & TPs
    "stop_loss_pct": 0.03,
    "tp_ladder": [0.06, 0.12, 0.2],
    "tp_fractions": [0.4, 0.3, 0.3],
    "tp_anticipation_pct": 0.0,     # esce prima del target (0.01 = -1%)
    "stop_mode": "Percentuale",     # Percentuale | ATR | Minimo

    # Trailing
    "trailing_start_pct": 0.10,
    "trailing_distance_pct": 0.05,
    "breakeven_stop_pct": 0.0,      # sposta SL a BE dopo X% gain

    # Strategy (base)
    "breakout_threshold_pct": 0.005,
    "ema_fast": 20,
    "ema_slow": 50,
    "trend_condition": "fast_above_slow",  # fast_above_slow | both_above | none
    "min_volume_factor": 1.3,

    # Volatilità/filtri
    "atr_period": 14,
    "atr_multiplier": 2.0,          # usato per stop ATR e/o sizing floor
    "adx_period": 14,
    "adx_min": 0.0,                  # 20-25 = trend forte (0 = disattivo)
    "atr_filter_min": 0.0,           # come % del prezzo: atr/close >= min
    "atr_filter_max": 1.0,           # come % del prezzo: atr/close <= max

    # Regole operative extra
    "cooldown_bars": 0,
    "max_trades_per_day": 9999,
    "time_stop_bars": 0,             # 0 = no time stop
    "force_close_bars": 0,           # 0 = no force close a N barre

    # Sessioni orarie (0-24); se start==end => 24h
    "session_start_h": 0,
    "session_end_h": 24,

    # Multi TF trend (approssimato moltiplicando la slow EMA)
    "use_multi_tf_trend": False,
    "higher_tf_multiplier": 4,       # es. 4 * ema_slow ~ 4H se TF=1H

    # Protezioni
    "daily_loss_limit_pct": 0.0,     # 0 = off (calcolato sul capitale iniziale)
    "equity_stop_pct": 0.0,          # stop globale sull’equity (0 = off)

    # Data source default
    "data_src": "Bybit (ccxt)",      # evitare 451 su Streamlit Cloud
}

# -------------------------
# Indicatori
# -------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=int(max(1, length)), adjust=False).mean()

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).rolling(window=int(max(1, period)), min_periods=1).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    p = int(max(1, period))
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df)
    atr_w = tr.rolling(p, min_periods=1).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(p, min_periods=1).sum() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(p, min_periods=1).sum() / atr_w
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.rolling(p, min_periods=1).mean()

# -------------------------
# Data helpers
# -------------------------
def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int = 1000, exchange_name: str = "bybit") -> Optional[pd.DataFrame]:
    """
    Scarica dati OHLCV da vari exchange via ccxt.
    exchange_name può essere: binance, bybit, okx, kraken, coinbase, ecc.
    Prova automaticamente la variante /USD se /USDT non è supportata.
    """
    if not HAS_CCXT:
        st.error("ccxt non installato. `pip install ccxt`")
        return None
    try:
        if not hasattr(ccxt, exchange_name):
            st.error(f"Exchange non supportato in ccxt: {exchange_name}")
            return None
        exchange_class = getattr(ccxt, exchange_name)
        ex = exchange_class({'enableRateLimit': True})

        tried = []
        ohlcv = None
        for sym in [symbol, symbol.replace("USDT", "USD")]:
            tried.append(sym)
            try:
                ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=int(limit))
                if ohlcv:
                    symbol = sym
                    break
            except Exception:
                continue
        if not ohlcv:
            raise RuntimeError(f"Impossibile fetch OHLCV da {exchange_name} per: {tried}")
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').astype(float)
        return df[['open','high','low','close','volume']]
    except Exception as e:
        st.error(f"Errore fetch {exchange_name} (ccxt): {e}")
        return None

def fetch_ohlcv_rest_binance(symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fallback REST pubblico Binance (potrebbe dare 451 su alcune regioni cloud)."""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.replace("/",""), "interval": interval, "limit": int(limit)}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "qav","trades","tbb","tbq","ignore"
        ])
        df = df[["open_time","open","high","low","close","volume"]]
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("timestamp").astype(float)
        return df[['open','high','low','close','volume']]
    except Exception as e:
        st.error(f"Errore fetch Binance (REST): {e}")
        return None

def synthetic_data(n: int = 1000, seed: int = 42, drift: float = 0.001, vol: float = 0.01, timeframe: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 2000.0
    closes = []
    for _ in range(n):
        ret = drift + vol * rng.standard_normal()
        price = max(10.0, price * (1.0 + ret))
        closes.append(price)
    closes = np.array(closes)
    highs = closes * (1.0 + np.maximum(0, vol * 0.5 * rng.standard_normal(n)))
    lows = closes * (1.0 - np.maximum(0, vol * 0.5 * rng.standard_normal(n)))
    opens = np.r_[closes[0], closes[:-1]]
    vols = 100 + 10 * rng.random(n)
    # mappa freq pandas
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1H", "4h": "4H", "1d": "1D"}
    idx = pd.date_range("2024-01-01", periods=n, freq=freq_map.get(timeframe, "1H"))
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=idx)

# -------------------------
# Profili (import/export) — NO rerun (applica al prossimo avvio)
# -------------------------
PROFILE_DIR = Path("configs"); PROFILE_DIR.mkdir(exist_ok=True)

def merge_defaults(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = base.copy()
    out.update({k: v for k, v in overlay.items() if v is not None})
    return out

# Applica profilo (se presente) PRIMA di creare i widget
if "pending_profile" in st.session_state and isinstance(st.session_state["pending_profile"], dict):
    DEFAULT_CONFIG = merge_defaults(DEFAULT_CONFIG, st.session_state["pending_profile"])
    del st.session_state["pending_profile"]

# -------------------------
# Backtest Engine (Long Only) — completo
# -------------------------
def run_backtest(df: pd.DataFrame, c: Dict[str, Any]) -> Dict[str, Any]:
    c = c.copy()  # difensivo contro state-leaks tra run
    df = df.copy()

    # Indicatori
    df["ema_fast"] = ema(df["close"], c["ema_fast"])
    df["ema_slow"] = ema(df["close"], c["ema_slow"])
    df["atr"] = atr(df, c["atr_period"])
    df["vol_ma"] = df["volume"].rolling(20, min_periods=1).mean()
    df["adx"] = adx(df, c["adx_period"])

    # Multi-TF trend “approssimato”
    if c.get("use_multi_tf_trend", False):
        df["ema_slow_htf"] = ema(df["close"], int(c["ema_slow"] * max(1, int(c["higher_tf_multiplier"]))))
    else:
        df["ema_slow_htf"] = df["ema_slow"]

    # Normalizza TP (con anticipo)
    tp_ladder = list(c.get("tp_ladder", [0.06, 0.12, 0.2]))
    tp_frac = list(c.get("tp_fractions", [0.4, 0.3, 0.3]))
    s = sum(tp_frac); tp_frac = [f / s for f in tp_frac] if s > 0 else [1.0 / max(1, len(tp_frac))] * max(1, len(tp_frac))
    tp_anticip = float(c.get("tp_anticipation_pct", 0.0))
    tp_factors = [(1.0 + tp) * (1.0 - tp_anticip) for tp in tp_ladder]

    # Contabilità
    cash = float(c["initial_capital"])
    fee_pct = float(c["fee_pct"])
    slip = float(c["slippage_pct"])
    risk_pct = float(c["risk_per_trade_pct"])

    trades: List[Dict[str, Any]] = []
    eq_list: List[Dict[str, Any]] = []
    position: Optional[Dict[str, Any]] = None
    pending_entry_idx: Optional[int] = None
    cooldown = 0
    trades_today = 0
    daily_pnl = 0.0
    current_day = None

    def in_session(ts: pd.Timestamp) -> bool:
        sh, eh = int(c["session_start_h"]), int(c["session_end_h"])
        if sh == eh:  # 24h
            return True
        hh = ts.hour
        if sh < eh:
            return sh <= hh < eh
        else:
            return hh >= sh or hh < eh

    n = len(df)
    for i in range(n):
        t = df.index[i]
        row = df.iloc[i]

        # reset day
        if current_day is None or t.date() != current_day:
            current_day = t.date()
            trades_today = 0
            daily_pnl = 0.0

        # Esegui pending entry all'open barra i
        if pending_entry_idx is not None and pending_entry_idx == i:
            if in_session(t) and trades_today < int(c["max_trades_per_day"]):
                entry = row["open"] * (1 + slip)

                # Stop candidates
                stop_pct = entry * (1.0 - float(c["stop_loss_pct"]))
                stop_atr = entry - float(c["atr_multiplier"]) * float(row["atr"])

                mode = c.get("stop_mode", "Percentuale")
                if mode == "Percentuale":
                    stop = stop_pct
                elif mode == "ATR":
                    stop = stop_atr
                else:  # Minimo tra i due -> stop più vicino all'entry (prezzo più alto)
                    stop = max(stop_pct, stop_atr)

                # Filtro ATR% (ambiente negoziabile)
                atr_rel = (row["atr"] / row["close"]) if row["close"] > 0 else 0.0
                if atr_rel >= float(c["atr_filter_min"]) and atr_rel <= float(c["atr_filter_max"]):
# --- Calcolo posizione e controllo capitale ---
R = max(entry - stop, 1e-9)

# Clamp risk_per_trade_pct a valori realistici (0.01% - 10%)
risk_pct = float(c["risk_per_trade_pct"])
risk_pct = min(max(risk_pct, 0.0001), 0.1)

risk_amt = cash * risk_pct

# sizing floor su ATR (opzionale): garantisce R minimo
if float(c["atr_multiplier"]) > 0:
    R = max(R, float(c["atr_multiplier"]) * float(row["atr"]))

size = risk_amt / R
notional = size * entry
fee = notional * fee_pct

# Debug info: spiega perché eventuali trade vengono saltati
if (notional + fee) > cash:
    st.warning(f"Trade skipped: notional {notional:.2f} + fee {fee:.2f} > cash {cash:.2f}")

if size > 0 and (notional + fee) <= cash:
    cash -= (notional + fee)
    position = {
        "entry_time": t,
        "entry_price": entry,
        "avg_entry": entry,
        "size": size,
        "remaining_size": size,
        "stop": stop,
        "tp_prices": [entry * f for f in tp_factors],
        "tp_fractions": tp_frac.copy(),
        "tp_taken": [False] * len(tp_frac),
        "max_price": entry,
        "trailing_active": False,
        "trail_price": None,
        "bars_open": 0
    }
    trades.append({"type": "ENTRY", "time": t, "price": entry, "size": size})
    trades_today += 1
    cooldown = int(c["cooldown_bars"])
        # Gestione posizione
        if position is not None:
            position["bars_open"] += 1
            position["max_price"] = max(position["max_price"], row["high"])

            # Breakeven
            if float(c.get("breakeven_stop_pct", 0.0)) > 0:
                be_level = position["entry_price"] * (1.0 + float(c["breakeven_stop_pct"]))
                if row["high"] >= be_level:
                    position["stop"] = max(position["stop"], position["entry_price"])

            # SL
            if row["low"] <= position["stop"]:
                px = position["stop"] * (1 - slip)
                size = position["remaining_size"]
                proceeds = size * px
                fee = proceeds * fee_pct
                cash += proceeds - fee
                pnl = (proceeds - fee) - (size * position["avg_entry"])
                trades.append({"type": "SL", "time": t, "price": px, "size": size, "pnl": pnl})
                daily_pnl += pnl
                position = None
            else:
                # TP parziali
                for j, tp_level in enumerate(position["tp_prices"]):
                    if (not position["tp_taken"][j]) and (row["high"] >= tp_level):
                        px = tp_level * (1 - slip)
                        size = min(position["remaining_size"], position["size"] * position["tp_fractions"][j])
                        if size > 1e-12:
                            proceeds = size * px
                            fee = proceeds * fee_pct
                            cash += proceeds - fee
                            pnl = (proceeds - fee) - (size * position["avg_entry"])
                            position["remaining_size"] -= size
                            position["tp_taken"][j] = True
                            trades.append({"type": "TP", "tp_index": j, "time": t, "price": px, "size": size, "pnl": pnl})
                            daily_pnl += pnl

                # Trailing
                if position is not None:
                    if (not position["trailing_active"]) and (position["max_price"] >= position["entry_price"] * (1.0 + float(c["trailing_start_pct"]))):
                        position["trailing_active"] = True
                        position["trail_price"] = position["max_price"] * (1.0 - float(c["trailing_distance_pct"]))
                    elif position["trailing_active"]:
                        position["trail_price"] = position["max_price"] * (1.0 - float(c["trailing_distance_pct"]))
                        if row["low"] <= position["trail_price"]:
                            px = position["trail_price"] * (1 - slip)
                            size = position["remaining_size"]
                            proceeds = size * px
                            fee = proceeds * fee_pct
                            cash += proceeds - fee
                            pnl = (proceeds - fee) - (size * position["avg_entry"])
                            trades.append({"type": "TRAIL", "time": t, "price": px, "size": size, "pnl": pnl})
                            daily_pnl += pnl
                            position = None

                # Time stop / Force close
                if position is not None:
                    if int(c.get("time_stop_bars", 0)) > 0 and position["bars_open"] >= int(c["time_stop_bars"]):
                        px = row["close"] * (1 - slip)
                        size = position["remaining_size"]
                        proceeds = size * px
                        fee = proceeds * fee_pct
                        cash += proceeds - fee
                        pnl = (proceeds - fee) - (size * position["avg_entry"])
                        trades.append({"type": "TIME_STOP", "time": t, "price": px, "size": size, "pnl": pnl})
                        daily_pnl += pnl
                        position = None
                    elif int(c.get("force_close_bars", 0)) > 0 and position["bars_open"] >= int(c["force_close_bars"]):
                        px = row["close"] * (1 - slip)
                        size = position["remaining_size"]
                        proceeds = size * px
                        fee = proceeds * fee_pct
                        cash += proceeds - fee
                        pnl = (proceeds - fee) - (size * position["avg_entry"])
                        trades.append({"type": "FORCE_CLOSE", "time": t, "price": px, "size": size, "pnl": pnl})
                        daily_pnl += pnl
                        position = None

                if position is not None and position["remaining_size"] <= 1e-12:
                    position = None

        # Nuove entry se flat
        if position is None and pending_entry_idx is None and i < n - 1:
            # daily loss limit
            if float(c.get("daily_loss_limit_pct", 0.0)) > 0 and daily_pnl <= -abs(float(c["daily_loss_limit_pct"]) * float(c["initial_capital"])):
                pass  # stop per la giornata
            # cooldown
            elif cooldown > 0:
                cooldown -= 1
            else:
                # condizioni segnale
                if in_session(t) and trades_today < int(c["max_trades_per_day"]):
                    prev_high = df["high"].iloc[i - 1] if i >= 1 else df["high"].iloc[i]
                    breakout = row["high"] > prev_high * (1.0 + float(c["breakout_threshold_pct"]))
                    vol_ok = row["volume"] >= df["vol_ma"].iloc[i] * float(c["min_volume_factor"])

                    tc = c.get("trend_condition", "fast_above_slow")
                    if tc == "fast_above_slow":
                        trend_ok = row["ema_fast"] > row["ema_slow"]
                    elif tc == "both_above":
                        trend_ok = (row["close"] > row["ema_fast"]) and (row["close"] > row["ema_slow"])
                    else:
                        trend_ok = True

                    adx_ok = row["adx"] >= float(c["adx_min"])

                    if c.get("use_multi_tf_trend", False):
                        trend_ok = trend_ok and (row["close"] > row["ema_slow_htf"])

                    # ATR band
                    atr_rel = (row["atr"] / row["close"]) if row["close"] > 0 else 0.0
                    atr_ok = (atr_rel >= float(c["atr_filter_min"])) and (atr_rel <= float(c["atr_filter_max"]))

                    if breakout and vol_ok and trend_ok and adx_ok and atr_ok:
                        pending_entry_idx = i + 1

        # Equity MTM
        mtm = (position["remaining_size"] * row["close"]) if position else 0.0
        eq_list.append({"time": t, "equity": cash + mtm})

    # chiusura a fine serie
    if position is not None and position["remaining_size"] > 0:
        last = df.iloc[-1]
        px = last["close"] * (1 - slip)
        size = position["remaining_size"]
        proceeds = size * px
        fee = proceeds * fee_pct
        cash += proceeds - fee
        pnl = (proceeds - fee) - (size * position["avg_entry"])
        trades.append({"type": "CLOSE_AT_END", "time": df.index[-1], "price": px, "size": size, "pnl": pnl})
        position = None

    eq_df = pd.DataFrame(eq_list).set_index("time") if eq_list else pd.DataFrame([], columns=["equity"]).set_index(pd.Index([]))
    return {
        "trades": trades,
        "equity_curve": eq_df,
        "final_capital": cash,
        "initial_capital": float(c["initial_capital"]),
        "df": df
    }

# -------------------------
# UI — Sidebar
# -------------------------
st.title("Crypto trading bot Backtest")

with st.sidebar:
    st.header("Data Source")

    data_src_options = ["Binance (ccxt)", "Bybit (ccxt)", "OKX (ccxt)", "Kraken (ccxt)",
                        "Coinbase (ccxt)", "Binance (REST)", "Synthetic Uptrend", "Synthetic Downtrend"]
    try:
        default_index = data_src_options.index(DEFAULT_CONFIG.get("data_src", "Bybit (ccxt)"))
    except ValueError:
        default_index = 1
    data_src = st.selectbox("Fonte dati", data_src_options, index=default_index)

    # Import profilo (JSON) — niente rerun: applica al prossimo ciclo
    st.subheader("Profili")
    up = st.file_uploader("Carica profilo (.json)", type=["json"])
    if up is not None:
        try:
            prof = json.load(up)
            st.session_state["pending_profile"] = prof
            st.success("Profilo caricato: i valori verranno proposti come default alla prossima esecuzione. Premi 'Run backtest'.")
        except Exception as e:
            st.error(f"JSON non valido: {e}")

    # Base
    st.subheader("Base")
    symbol = st.text_input("Symbol", DEFAULT_CONFIG["symbol"])
    timeframe_list = ["1m","5m","15m","30m","1h","4h","1d"]
    try:
        tf_index = timeframe_list.index(DEFAULT_CONFIG["timeframe"])
    except ValueError:
        tf_index = 4  # 1h
    timeframe = st.selectbox("Timeframe", timeframe_list, index=tf_index)
    lookback = st.number_input("Candles (lookback)", value=int(DEFAULT_CONFIG["lookback"]), min_value=200, max_value=5000, step=100)

    # Money management
    st.subheader("Money management")
    initial_capital = st.number_input("Initial capital (USD)", value=float(DEFAULT_CONFIG["initial_capital"]), step=100.0)
    risk_per_trade_pct = st.number_input("Risk per trade (%)", value=DEFAULT_CONFIG["risk_per_trade_pct"]*100.0)/100.0
    fee_pct = st.number_input("Fee (%)", value=DEFAULT_CONFIG["fee_pct"]*100.0)/100.0
    slippage_pct = st.number_input("Slippage (%)", value=DEFAULT_CONFIG["slippage_pct"]*100.0)/100.0

    # Stops & TPs
    st.subheader("Stops & TPs")
    stop_mode = st.selectbox("Stop mode", ["Percentuale","ATR","Minimo"], index=["Percentuale","ATR","Minimo"].index(DEFAULT_CONFIG["stop_mode"]))
    stop_loss_pct = st.number_input("Stop loss (%)", value=DEFAULT_CONFIG["stop_loss_pct"]*100.0)/100.0
    tp_ladder_str = st.text_input("TP ladder (comma)", ",".join(map(str, DEFAULT_CONFIG["tp_ladder"])))
    tp_fractions_str = st.text_input("TP fractions (sum=1)", ",".join(map(str, DEFAULT_CONFIG["tp_fractions"])))
    tp_anticipation_pct = st.number_input("TP anticipation (%)", value=DEFAULT_CONFIG["tp_anticipation_pct"]*100.0)/100.0

    # Trailing
    st.subheader("Trailing")
    trailing_start_pct = st.number_input("Trailing start (%)", value=DEFAULT_CONFIG["trailing_start_pct"]*100.0)/100.0
    trailing_distance_pct = st.number_input("Trailing distance (%)", value=DEFAULT_CONFIG["trailing_distance_pct"]*100.0)/100.0
    breakeven_stop_pct = st.number_input("Breakeven stop after gain (%)", value=DEFAULT_CONFIG["breakeven_stop_pct"]*100.0)/100.0

    # Strategy
    st.subheader("Strategy")
    breakout_threshold_pct = st.number_input("Breakout threshold (%)", value=DEFAULT_CONFIG["breakout_threshold_pct"]*100.0)/100.0
    ema_fast = st.number_input("EMA fast", value=int(DEFAULT_CONFIG["ema_fast"]), min_value=1, max_value=500)
    ema_slow = st.number_input("EMA slow", value=int(DEFAULT_CONFIG["ema_slow"]), min_value=1, max_value=1000)
    trend_condition = st.selectbox("Trend condition", ["fast_above_slow","both_above","none"],
                                   index=["fast_above_slow","both_above","none"].index(DEFAULT_CONFIG["trend_condition"]))
    min_volume_factor = st.number_input("Min volume factor", value=float(DEFAULT_CONFIG["min_volume_factor"]), min_value=0.1)

    # Volatilità/filtri
    st.subheader("Volatilità / Filtri")
    atr_period = st.number_input("ATR period", value=int(DEFAULT_CONFIG["atr_period"]), min_value=1)
    atr_multiplier = st.number_input("ATR multiplier (stop/sizing)", value=float(DEFAULT_CONFIG["atr_multiplier"]), min_value=0.0)
    adx_period = st.number_input("ADX period", value=int(DEFAULT_CONFIG["adx_period"]), min_value=1)
    adx_min = st.number_input("ADX min", value=float(DEFAULT_CONFIG["adx_min"]), min_value=0.0)
    atr_filter_min = st.number_input("ATR% min (atr/close)", value=float(DEFAULT_CONFIG["atr_filter_min"]), min_value=0.0, max_value=1.0)
    atr_filter_max = st.number_input("ATR% max (atr/close)", value=float(DEFAULT_CONFIG["atr_filter_max"]), min_value=0.0, max_value=1.0)

    # Regole operative
    st.subheader("Regole operative")
    cooldown_bars = st.number_input("Cooldown bars", value=int(DEFAULT_CONFIG["cooldown_bars"]), min_value=0)
    max_trades_per_day = st.number_input("Max trades per day", value=int(DEFAULT_CONFIG["max_trades_per_day"]), min_value=1)
    time_stop_bars = st.number_input("Time stop bars (0=off)", value=int(DEFAULT_CONFIG["time_stop_bars"]), min_value=0)
    force_close_bars = st.number_input("Force close bars (0=off)", value=int(DEFAULT_CONFIG["force_close_bars"]), min_value=0)

    # Sessioni
    st.subheader("Sessione")
    session_start_h = st.number_input("Start hour (0-24)", value=int(DEFAULT_CONFIG["session_start_h"]), min_value=0, max_value=24)
    session_end_h = st.number_input("End hour (0-24)", value=int(DEFAULT_CONFIG["session_end_h"]), min_value=0, max_value=24)

    # Multi TF
    st.subheader("Multi-TF Trend")
    use_multi_tf_trend = st.checkbox("Use multi-TF trend confirm", value=bool(DEFAULT_CONFIG["use_multi_tf_trend"]))
    higher_tf_multiplier = st.number_input("Higher TF multiplier", value=int(DEFAULT_CONFIG["higher_tf_multiplier"]), min_value=1)

    # Protezioni
    st.subheader("Protezioni")
    daily_loss_limit_pct = st.number_input("Daily loss limit (%)", value=float(DEFAULT_CONFIG["daily_loss_limit_pct"]), min_value=0.0)
    equity_stop_pct = st.number_input("Equity stop (%)", value=float(DEFAULT_CONFIG["equity_stop_pct"]), min_value=0.0)

    # Compose cfg
    try:
        tp_ladder = [float(x.strip()) for x in tp_ladder_str.split(",") if x.strip()!=""]
        tp_fractions = [float(x.strip()) for x in tp_fractions_str.split(",") if x.strip()!=""]
        s = sum(tp_fractions)
        if s <= 0:
            tp_fractions = [1.0/len(tp_fractions)]*len(tp_fractions) if tp_fractions else [1.0]
        else:
            tp_fractions = [f/s for f in tp_fractions]
    except Exception:
        st.warning("Errore parsing TP / Fractions: uso default.")
        tp_ladder = DEFAULT_CONFIG["tp_ladder"]
        tp_fractions = DEFAULT_CONFIG["tp_fractions"]

    cfg: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback": int(lookback),
        "initial_capital": float(initial_capital),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "fee_pct": float(fee_pct),
        "slippage_pct": float(slippage_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "stop_mode": stop_mode,
        "tp_ladder": tp_ladder,
        "tp_fractions": tp_fractions,
        "tp_anticipation_pct": float(tp_anticipation_pct),
        "trailing_start_pct": float(trailing_start_pct),
        "trailing_distance_pct": float(trailing_distance_pct),
        "breakeven_stop_pct": float(breakeven_stop_pct),
        "breakout_threshold_pct": float(breakout_threshold_pct),
        "ema_fast": int(ema_fast),
        "ema_slow": int(ema_slow),
        "trend_condition": trend_condition,
        "min_volume_factor": float(min_volume_factor),
        "atr_period": int(atr_period),
        "atr_multiplier": float(atr_multiplier),
        "adx_period": int(adx_period),
        "adx_min": float(adx_min),
        "atr_filter_min": float(atr_filter_min),
        "atr_filter_max": float(atr_filter_max),
        "cooldown_bars": int(cooldown_bars),
        "max_trades_per_day": int(max_trades_per_day),
        "time_stop_bars": int(time_stop_bars),
        "force_close_bars": int(force_close_bars),
        "session_start_h": int(session_start_h),
        "session_end_h": int(session_end_h),
        "use_multi_tf_trend": bool(use_multi_tf_trend),
        "higher_tf_multiplier": int(higher_tf_multiplier),
        "daily_loss_limit_pct": float(daily_loss_limit_pct),
        "equity_stop_pct": float(equity_stop_pct),
        "data_src": data_src,
    }

    # Export profilo
    st.download_button("⬇️ Scarica profilo JSON", data=json.dumps(cfg, indent=2), file_name="profile.json", mime="application/json")

    run_bt = st.button("Run backtest")

# -------------------------
# Main run
# -------------------------
if run_bt:
    with st.spinner("Fetching OHLCV..."):
        if "ccxt" in cfg["data_src"]:
            exchange_name = cfg["data_src"].split()[0].lower()  # es: "Bybit (ccxt)" -> "bybit"
            df = fetch_ohlcv_ccxt(cfg["symbol"], cfg["timeframe"], cfg["lookback"], exchange_name=exchange_name)
        elif cfg["data_src"] == "Binance (REST)":
            df = fetch_ohlcv_rest_binance(cfg["symbol"], cfg["timeframe"], cfg["lookback"])
        elif cfg["data_src"] == "Synthetic Uptrend":
            df = synthetic_data(n=cfg["lookback"], drift=+0.001, vol=0.01, timeframe=cfg["timeframe"])
        else:
            df = synthetic_data(n=cfg["lookback"], drift=-0.001, vol=0.01, timeframe=cfg["timeframe"])

    if df is None or df.empty or len(df) < max(cfg["ema_slow"]*2, 50):
        st.error("Dati insufficienti per il backtest.")
    else:
        st.success(f"Fetched {len(df)} candles — running backtest…")
        res = run_backtest(df, cfg)

        trades = res["trades"]
        equity = res["equity_curve"]
        final_cap = res["final_capital"]
        initial_cap = res["initial_capital"]
        df_ind = res["df"]

        pnl = final_cap - initial_cap
        roi = (pnl / initial_cap * 100.0) if initial_cap else 0.0
        n_entries = sum(1 for t in trades if t["type"] == "ENTRY")

        c1, c2, c3 = st.columns(3)
        c1.metric("Initial", f"{initial_cap:,.2f} USD")
        c2.metric("Final", f"{final_cap:,.2f} USD", delta=f"{pnl:,.2f} USD ({roi:.2f}%)")
        c3.metric("Entries", n_entries)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_ind.index, open=df_ind["open"], high=df_ind["high"], low=df_ind["low"], close=df_ind["close"], name="Price"
        ))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema_fast"], mode="lines", name=f"EMA{cfg['ema_fast']}", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ema_slow"], mode="lines", name=f"EMA{cfg['ema_slow']}", line=dict(width=1)))

        entry_x, entry_y, exit_x, exit_y, entry_txt, exit_txt = [], [], [], [], [], []
        for t in trades:
            if t["type"] == "ENTRY":
                entry_x.append(t["time"]); entry_y.append(t["price"])
                entry_txt.append(f"ENTRY\nprice={t['price']:.2f}\nsize={t.get('size',0):.6f}")
            elif t["type"] in ["TP","SL","TRAIL","TIME_STOP","FORCE_CLOSE","CLOSE_AT_END"]:
                exit_x.append(t["time"]); exit_y.append(t["price"])
                exit_txt.append(f"{t['type']}\nprice={t['price']:.2f}\npnl={t.get('pnl',0.0):.2f}")
        if entry_x:
            fig.add_trace(go.Scatter(x=entry_x, y=entry_y, mode="markers", name="Entry",
                                     marker_symbol="triangle-up", marker_color="green", marker_size=10,
                                     text=entry_txt, hovertemplate="%{text}<extra></extra>"))
        if exit_x:
            fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exit",
                                     marker_symbol="x", marker_color="red", marker_size=8,
                                     text=exit_txt, hovertemplate="%{text}<extra></extra>"))

        fig.update_layout(xaxis_rangeslider_visible=False, height=600, title=f"{cfg['symbol']} — Backtest")
        st.plotly_chart(fig, use_container_width=True)

        # Equity curve
        eq_fig = go.Figure()
        if not equity.empty:
            eq_fig.add_trace(go.Scatter(x=equity.index, y=equity["equity"], mode="lines", name="Equity"))
            peak = equity["equity"].cummax()
            dd = (equity["equity"] - peak) / peak
            st.write(f"Max drawdown: {100.0*dd.min():.2f}%")
        eq_fig.update_layout(height=250, margin=dict(t=10))
        st.plotly_chart(eq_fig, use_container_width=True)

        # Trade log
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            tdf["time"] = pd.to_datetime(tdf["time"])
            st.subheader("Trade Log")
            st.dataframe(tdf.sort_values("time").reset_index(drop=True))
        else:
            st.info("Nessun trade eseguito con i parametri correnti.")

