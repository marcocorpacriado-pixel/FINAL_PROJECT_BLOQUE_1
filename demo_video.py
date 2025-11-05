# demo_video.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# 1) Cargar .env para las API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

SESSION = requests.Session()
OUTDIR = Path("outputs/demo_video")
START = "2022-01-01"

# ---------- Utilidades ----------
def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def _ok(msg: str):   print(f"\033[92m✓ {msg}\033[0m")
def _warn(msg: str): print(f"\033[93m⚠ {msg}\033[0m")
def _err(msg: str):  print(f"\033[91m✗ {msg}\033[0m")

# ---------- Descargas ----------
def fetch_yahoo(symbol: str, dest: Path, start: str = START) -> tuple[bool, str]:
    """
    Descarga UN solo símbolo de Yahoo y guarda parquet con columnas planas.
    (Secuencial; sin ThreadPoolExecutor)
    """
    try:
        df = yf.download(symbol, start=start, progress=False, auto_adjust=False, threads=True)
        if df is None or df.empty:
            return False, f"[Yahoo] {symbol} vacío"

        # Por robustez: si viniera MultiIndex, intenta extraer el nivel del ticker exacto
        if isinstance(df.columns, pd.MultiIndex):
            lvl1 = [str(x) for x in df.columns.get_level_values(1)]
            if symbol in lvl1:
                df = df.xs(symbol, level=1, axis=1, drop_level=True)
            else:
                # último recurso (no debería ocurrir cuando pedimos un único símbolo)
                df = df.droplevel(1, axis=1)

        df = df.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("date").dropna(subset=["date", "close"])

        _save_parquet(df, dest)
        return True, str(dest)
    except Exception as e:
        return False, f"[Yahoo] {symbol}: {e}"

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
        r = SESSION.get(url, params=params, timeout=30)
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

# ---------- Descarga secuencial ----------
def download_demo_data() -> dict[str, Path]:
    """
    Yahoo (secuencial): NVDA, CCJ, PLTR
    TwelveData (secuencial, con fallback): IBEX (intenta varios aliases; si falla, Yahoo ^IBEX)
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    # 1) Yahoo secuencial
    yahoo_list = [
        ("NVDA", Path("data/yahoo/NVDA/NVDA_1d.parquet")),
        ("CCJ",  Path("data/yahoo/CCJ/CCJ_1d.parquet")),
        ("PLTR", Path("data/yahoo/PLTR/PLTR_1d.parquet")),
    ]
    print("↓ Descargando Yahoo Finance (secuencial): NVDA, CCJ, PLTR…")
    for sym, dest in yahoo_list:
        ok, msg = fetch_yahoo(sym, dest, start=START)
        if ok:
            _ok(f"{sym}: descargado → {msg}")
            files[sym] = dest
        else:
            _err(f"{sym}: {msg}")

    # 2) IBEX vía TwelveData (intentos secuenciales)
    print("↓ Descargando IBEX 35 vía TwelveData (con fallback a Yahoo ^IBEX)…")
    td_candidates = ["IBEX", "IBEX35", "ES35", "ES-35"]
    td_ok = False
    for s in td_candidates:
        dest = Path("data/twelvedata/IBEX/IBEX_1d.parquet")
        ok, msg = fetch_twelvedata(s, dest, start=START)
        if ok:
            _ok(f"IBEX (TwelveData:{s}) → {msg}")
            files["IBEX"] = dest
            td_ok = True
            break
        else:
            _warn(f"IBEX (TwelveData:{s}) falló: {msg}")

    if not td_ok:
        ok, msg = fetch_yahoo("^IBEX", Path("data/yahoo/IBEX/IBEX_1d.parquet"), start=START)
        if ok:
            _ok(f"IBEX fallback Yahoo (^IBEX) → {msg}")
            files["IBEX"] = Path("data/yahoo/IBEX/IBEX_1d.parquet")
        else:
            _err(f"IBEX no disponible: {msg}")

    return files

# ---------- Construcción y demo de cartera ----------
def main():
    files = download_demo_data()

    needed = {"NVDA", "CCJ", "PLTR", "IBEX"}
    missing = needed - set(files.keys())
    if missing:
        _warn(f"Faltan datos para: {', '.join(sorted(missing))}. Continúo con lo disponible.")

    # 2) Cargar Candles (normalizando fechas a tz-naive)
    series: dict[str, Candles] = {}
    for name, path in files.items():
        c = Candles.from_any(path, symbol=name).between(start=START)
        df = c.frame.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if hasattr(df["date"], "dt") and getattr(df["date"].dt, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        series[name] = Candles(symbol=name, frame=df)

    # 3) Pesos (suman 1)
    weights = {"NVDA": 0.35, "CCJ": 0.20, "PLTR": 0.25, "IBEX": 0.20}
    total_avail = sum(weights[k] for k in weights if k in series)
    weights = {k: v / total_avail for k, v in weights.items() if k in series}

    # 4) Portfolio
    pf = Portfolio(series=series, weights=weights, initial_value=1.0)
    _ok("Cartera construida correctamente")

    # 5) Correlaciones + aviso
    try:
        corr = pf.assets_corr_matrix()
        print("\n📊 Matriz de correlaciones:\n", corr.round(3), "\n")
        warn = pf.max_correlation_warning(threshold=0.5)
        if warn:
            _warn(warn)
        else:
            _ok("Correlaciones bajo umbral 0.5")
    except Exception as e:
        _err(f"No se pudo calcular la matriz de correlaciones: {e}")

    # 6) Monte Carlo (bandas + trayectorias) + Report
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n⏳ Ejecutando simulaciones Monte Carlo (GBM, Cholesky, Bootstrap, Cópula)…")
    report_mc = pf.monte_carlo_overview(
        days=252,
        n_paths=1500,              # paths por método
        seed=42,
        freq=252,
        rf=0.02,
        show_paths_per_method=30,  # para el vídeo
        alpha_band=(5, 95),
        title="Monte Carlo — Comparativa de métodos (bandas + trayectorias por método)",
        save_path=str(OUTDIR / "mc_comparison.png"),
    )
    _ok(f"Gráfico Monte Carlo guardado en {OUTDIR/'mc_comparison.png'}")

    report = pf.report(freq=252, rf=0.02, mc_days=252, mc_paths=1500, seed=42)
    (OUTDIR / "Report.md").write_text(report, encoding="utf-8")
    _ok(f"Report.md guardado en {OUTDIR/'Report.md'}")

    print("\nResumen Monte Carlo (medias por método):\n")
    print(report_mc)
    print("\n✅ DEMO COMPLETADA. Archivos en:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
