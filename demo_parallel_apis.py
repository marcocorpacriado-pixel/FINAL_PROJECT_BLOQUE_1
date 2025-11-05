# demo_parallel_apis.py
from __future__ import annotations
import os, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

# Carga .env con tus API keys: ALPHAVANTAGE_API_KEY, MARKETSTACK_API_KEY, TWELVEDATA_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# ---------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "finlab-demo/1.0"})
START = "2022-01-01"
OUTDIR = Path("outputs/parallel_demo")

# Utilidades
def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def _ok(msg: str):   print(f"\033[92m✓ {msg}\033[0m")
def _warn(msg: str): print(f"\033[93m⚠ {msg}\033[0m")
def _err(msg: str):  print(f"\033[91m✗ {msg}\033[0m")

# ---------------------------------------------------------------------------------------
# Descargas en paralelo (APIs REST → thread-safe)
# ---------------------------------------------------------------------------------------
def fetch_alphavantage(symbol: str, dest: Path) -> tuple[bool, str]:
    key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_KEY")
    if not key:
        return False, "No ALPHAVANTAGE_API_KEY en .env"
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full", "apikey": key}
    try:
        r = SESSION.get(url, params=params, timeout=40)
        r.raise_for_status()
        obj = r.json()
        ts = obj.get("Time Series (Daily)")
        if not ts:
            return False, f"AlphaVantage sin 'Time Series (Daily)': {obj}"
        records = []
        for d, row in ts.items():
            records.append({
                "date": d,
                "open": row.get("1. open"),
                "high": row.get("2. high"),
                "low":  row.get("3. low"),
                "close":row.get("4. close"),
                "adj_close": row.get("5. adjusted close"),
                "volume": row.get("6. volume"),
            })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        _save_parquet(df, dest)
        return True, str(dest)
    except Exception as e:
        return False, f"[AlphaVantage] {symbol}: {e}"

def fetch_marketstack(symbol: str, dest: Path, start: str = START) -> tuple[bool, str]:
    key = os.getenv("MARKETSTACK_API_KEY") or os.getenv("MARKETSTACK_KEY")
    if not key:
        return False, "No MARKETSTACK_API_KEY en .env"
    url = "http://api.marketstack.com/v1/eod"
    params = {"access_key": key, "symbols": symbol, "limit": 1000, "date_from": start, "sort": "ASC"}
    try:
        all_rows = []
        while True:
            r = SESSION.get(url, params=params, timeout=40)
            r.raise_for_status()
            obj = r.json()
            data = obj.get("data", [])
            if not data:
                break
            all_rows.extend(data)
            pg = obj.get("pagination", {})
            if not pg.get("has_more", False):
                break
            params["offset"] = pg.get("offset", 0) + pg.get("limit", 1000)
            time.sleep(0.2)
        if not all_rows:
            return False, "Marketstack vacío"
        df = pd.DataFrame(all_rows)
        _save_parquet(df, dest)
        return True, str(dest)
    except Exception as e:
        return False, f"[Marketstack] {symbol}: {e}"

def fetch_twelvedata(symbol: str, dest: Path, start: str = START) -> tuple[bool, str]:
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
    try:
        r = SESSION.get(url, params=params, timeout=40)
        r.raise_for_status()
        obj = r.json()
        if "values" not in obj:
            return False, f"TwelveData sin 'values': {obj}"
        vals = obj["values"]
        if not vals:
            return False, f"TwelveData vacío para {symbol}"
        df = pd.DataFrame(vals).rename(columns={"datetime": "date"})
        _save_parquet(df, dest)
        return True, str(dest)
    except Exception as e:
        return False, f"[TwelveData] {symbol}: {e}"

# ---------------------------------------------------------------------------------------
# Demo: descargas en paralelo y pipeline completo
# ---------------------------------------------------------------------------------------
def download_parallel() -> dict[str, Path]:
    """
    Descarga en paralelo:
      - Alpha Vantage: AAPL, MSFT
      - Marketstack:   TSLA, NVDA
      - TwelveData:    BTCUSD, XAUUSD  (márgenes no equity → genial para mezclas)
    Sin Yahoo aquí (para mostrar paralelismo 100% seguro).
    """
    jobs = {
        # nombre lógico : (proveedor, símbolo, ruta parquet)
        "AAPL": ("alphavantage", "AAPL", Path("data/alphavantage/AAPL/AAPL_1d.parquet")),
        "MSFT": ("alphavantage", "MSFT", Path("data/alphavantage/MSFT/MSFT_1d.parquet")),
        "TSLA": ("marketstack",  "TSLA", Path("data/marketstack/TSLA/TSLA_1d.parquet")),
        "NVDA": ("marketstack",  "NVDA", Path("data/marketstack/NVDA/NVDA_1d.parquet")),
        "BTC":  ("twelvedata",   "BTCUSD", Path("data/twelvedata/BTC/BTC_1d.parquet")),
        "GOLD": ("twelvedata",   "XAUUSD", Path("data/twelvedata/GOLD/GOLD_1d.parquet")),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    print("↓ Descargando en paralelo (AlphaVantage, Marketstack, TwelveData)…")

    def _dispatch(name: str, prov: str, sym: str, dest: Path):
        if prov == "alphavantage":
            return name, *fetch_alphavantage(sym, dest)
        elif prov == "marketstack":
            return name, *fetch_marketstack(sym, dest, start=START)
        elif prov == "twelvedata":
            return name, *fetch_twelvedata(sym, dest, start=START)
        else:
            return name, False, f"Proveedor no soportado: {prov}"

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_dispatch, n, p, s, d) for n, (p, s, d) in jobs.items()]
        for fut in as_completed(futs):
            name, ok, msg = fut.result()
            if ok:
                _ok(f"{name}: {msg}")
                results[name] = jobs[name][2]
            else:
                _err(f"{name}: {msg}")

    if not results:
        raise RuntimeError("No se descargó ningún activo. Revisa tus API keys en .env")
    return results

def main():
    files = download_parallel()

    # Cargar Candles normalizando fechas (tz-naive) y recortando a START
    series: dict[str, Candles] = {}
    for name, path in files.items():
        c = Candles.from_any(path, symbol=name).between(start=START)
        df = c.frame.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if hasattr(df["date"], "dt") and getattr(df["date"].dt, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        series[name] = Candles(symbol=name, frame=df)

    # Pesos de ejemplo (re-normaliza por si falló algún activo)
    default_weights = {
        "AAPL": 0.20,
        "MSFT": 0.20,
        "TSLA": 0.15,
        "NVDA": 0.15,
        "BTC":  0.15,
        "GOLD": 0.15,
    }
    total = sum(default_weights.get(k, 0.0) for k in series.keys())
    weights = {k: default_weights[k] / total for k in series.keys() if k in default_weights}

    pf = Portfolio(series=series, weights=weights, initial_value=1.0)
    _ok("Cartera construida correctamente (con activos descargados en paralelo)")

    # Matriz de correlaciones (+ warning)
    try:
        corr = pf.assets_corr_matrix()
        print("\n📊 Matriz de correlaciones:\n", corr.round(3), "\n")
        warn = pf.max_correlation_warning(threshold=0.5)
        if warn:
            _warn(warn)
        else:
            _ok("Correlaciones bajo el umbral 0.5")
    except Exception as e:
        _err(f"No se pudo calcular la matriz de correlaciones: {e}")

    # Monte Carlo (comparativa métodos) + Report
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("\n⏳ Ejecutando simulaciones Monte Carlo (GBM, Cholesky, Bootstrap, Cópula)…")
    report_mc = pf.monte_carlo_overview(
        days=252,
        n_paths=1500,           # paths por método en TU implementación
        seed=42,
        freq=252,
        rf=0.02,
        show_paths_per_method=25,
        alpha_band=(5, 95),
        title="Monte Carlo — Comparativa de métodos (descargas paralelas: AV/Marketstack/TwelveData)",
        save_path=str(OUTDIR / "mc_comparison_parallel.png"),
    )
    _ok(f"Gráfico Monte Carlo guardado en {OUTDIR/'mc_comparison_parallel.png'}")

    report = pf.report(freq=252, rf=0.02, mc_days=252, mc_paths=1500, seed=42)
    (OUTDIR / "Report.md").write_text(report, encoding="utf-8")
    _ok(f"Report.md guardado en {OUTDIR/'Report.md'}")

    print("\nResumen Monte Carlo (medias por método):\n")
    print(report_mc)

    print("\n✅ DEMO COMPLETADA. Archivos en:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
