# run_mixed_portfolio.py
from __future__ import annotations
import os
from pathlib import Path
import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Cargar .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# ---------- Helpers descarga ----------
SESSION = requests.Session()

def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def fetch_yahoo(symbol: str, dest: Path, start: str = "2022-01-01"):
    """Descarga diaria de Yahoo y deja columnas estándar planas."""
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, progress=False, auto_adjust=False, threads=True)

        if df is None or df.empty:
            raise ValueError("Yahoo vacío")

        # Si viene MultiIndex (p. ej. ('Close','GLD')), selecciona el nivel del ticker y aplana
        if isinstance(df.columns, pd.MultiIndex):
            # niveles: 0 = campo ('Open','High','Low','Close','Adj Close','Volume'), 1 = ticker
            tickers_lvl = [str(t) for t in df.columns.get_level_values(1).unique()]
            # intenta selección exacta (case-insensitive)
            chosen = None
            for t in tickers_lvl:
                if t.lower() == symbol.lower():
                    chosen = t
                    break
            if chosen is not None:
                df = df.xs(key=chosen, level=1, axis=1, drop_level=True)
            else:
                # si no encuentra, quita el nivel del ticker (coge la primera columna de cada campo)
                df = df.droplevel(1, axis=1)

        # Ahora df debe tener columnas tipo: Open, High, Low, Close, Adj Close, Volume
        df = df.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        # Asegura tipos y orden
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("date").dropna(subset=["date", "close"])

        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        return True, dest
    except Exception as e:
        return False, f"[Yahoo] {symbol} -> {e}"


def fetch_twelvedata(symbol: str, dest: Path, start: str = "2022-01-01"):
    key = os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVEDATA_KEY")
    if not key:
        return False, "No TWELVEDATA_API_KEY en .env"
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 5000,
        "apikey": key,
        "start_date": start,
        "order": "ASC",
    }
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    obj = r.json()
    if "values" not in obj:
        return False, f"TwelveData sin 'values': {obj}"
    vals = obj["values"]
    if not vals:
        return False, "TwelveData vacío"
    df = pd.DataFrame(vals)
    # normaliza nombres
    df = df.rename(columns={"datetime":"date"})
    _save_parquet(df, dest)
    return True, dest

def fetch_alphavantage(symbol: str, dest: Path):
    key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_KEY")
    if not key:
        return False, "No ALPHAVANTAGE_API_KEY en .env"
    url = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":symbol,"outputsize":"full","apikey":key}
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    obj = r.json()
    key_ts = "Time Series (Daily)"
    if key_ts not in obj:
        return False, f"AlphaVantage sin {key_ts}: {obj}"
    records = []
    for d, row in obj[key_ts].items():
        rec = {
            "date": d,
            "open": row.get("1. open"),
            "high": row.get("2. high"),
            "low": row.get("3. low"),
            "close": row.get("4. close"),
            "adj_close": row.get("5. adjusted close"),
            "volume": row.get("6. volume")
        }
        records.append(rec)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    _save_parquet(df, dest)
    return True, dest

def fetch_marketstack(symbol: str, dest: Path, start: str = "2022-01-01"):
    key = os.getenv("MARKETSTACK_API_KEY") or os.getenv("MARKETSTACK_KEY")
    if not key:
        return False, "No MARKETSTACK_API_KEY en .env"
    url = "http://api.marketstack.com/v1/eod"
    # Marketstack pagina; pedimos todo con loop
    params = {"access_key": key, "symbols": symbol, "limit": 1000, "date_from": start, "sort": "ASC"}
    all_rows = []
    while True:
        r = SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        obj = r.json()
        data = obj.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        if not obj.get("pagination", {}).get("has_more", False):
            break
        # pagina siguiente
        off = obj["pagination"]["offset"] + obj["pagination"]["limit"]
        params["offset"] = off
        time.sleep(0.2)
    if not all_rows:
        return False, "Marketstack vacío"
    df = pd.DataFrame(all_rows)
    _save_parquet(df, dest)
    return True, dest

def load_candles(path: Path, symbol: str) -> Candles:
    return Candles.from_any(path, symbol=symbol)

# ---------- Config cartera ----------
# Mapea activo -> (símbolo, proveedor preferido, fallback_yahoo_symbol)
ASSETS = {
    "GOLD":      {"pref": ("XAUUSD", "twelvedata"), "yahoo": "GLD"},   # Oro vía GLD si no hay FX
    "BTC":       {"pref": ("BTCUSD", "twelvedata"), "yahoo": "BTC-USD"},
    "QQQ":       {"pref": ("QQQ", "yahoo"),         "yahoo": "QQQ"},
    "DEFENSE":   {"pref": ("DFEN.DE", "yahoo"),     "yahoo": "DFEN.DE"},
    "HYG":       {"pref": ("HYG", "yahoo"),         "yahoo": "HYG"},
    "EEM":       {"pref": ("EEM", "yahoo"),         "yahoo": "EEM"},
}
WEIGHTS = {
    "GOLD": 0.10,
    "BTC": 0.10,
    "QQQ": 0.35,
    "DEFENSE": 0.15,
    "HYG": 0.10,
    "EEM": 0.20,
}
START = "2022-01-01"
OUTDIR = Path("outputs/mixed_demo")

def smart_download(name: str, cfg: dict) -> Path:
    sym, prov = cfg["pref"]
    data_dir = Path("data") / prov / name
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / f"{name}_1d.parquet"

    ok = False; msg = ""
    if prov == "yahoo":
        ok, msg = fetch_yahoo(sym, dest, start=START)
    elif prov == "twelvedata":
        ok, msg = fetch_twelvedata(sym, dest, start=START)
        if not ok:
            print(f"⚠️  fallo con {name} via twelvedata: {msg}\n  → fallback Yahoo: {cfg['yahoo']}")
            ok, msg = fetch_yahoo(cfg["yahoo"], Path("data")/"yahoo"/name/f"{name}_1d.parquet", start=START)
            dest = Path("data")/"yahoo"/name/f"{name}_1d.parquet"
    elif prov == "alphavantage":
        ok, msg = fetch_alphavantage(sym, dest)
    elif prov == "marketstack":
        ok, msg = fetch_marketstack(sym, dest, start=START)
    else:
        msg = f"Proveedor no soportado: {prov}"

    if not ok:
        raise RuntimeError(f"Descarga fallida {name}: {msg}")
    return dest

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) Descarga todo
    files = {}
    for name, cfg in ASSETS.items():
        p = smart_download(name, cfg)
        files[name] = p

    # 2) Cargar Candles y recortar al START para mejorar solape
    series = {}
    for name, path in files.items():
        c = load_candles(path, symbol=name).between(start=START)
        # Para cripto (24/7), quitamos tz si viene
        df = c.frame.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        c = Candles(symbol=c.symbol, frame=df)
        series[name] = c

    # 3) Construir Portfolio
    pf = Portfolio(series=series, weights=WEIGHTS, initial_value=1.0)

    # 4) Report + Plots
    report = pf.report(freq=252, rf=0.02, mc_days=252, mc_paths=2000, seed=42)
    (OUTDIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\n📝 Guardado: {OUTDIR/'report.md'}")

    # Plots (componentes outer para ver todo; relleno corto ya lo hace la clase)
    pf.plots_report(
        normalize=True,
        show_components=True,
        show_hist=True,
        show_corr=True,
        show_mc=True,
        mc_days=252,
        mc_paths=1500,
        seed=42,
        mu_scale=1.0,
        sigma_scale=1.0,
        save_dir=str(OUTDIR),
        dpi=140,
        logy=False,
        align="outer",
        ffill_limit=3,
    )

if __name__ == "__main__":
    main()
