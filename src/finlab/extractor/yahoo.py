




# src/finlab/extractor/yahoo.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf
from .io_utils import save_timeseries
from ..cache import cache

def fetch_prices_yahoo(
    symbol: str,
    outdir: Path,
    *,
    interval: str = "1d",        # "1m","5m","15m","1h","1d","1wk","1mo"
    start: str | None = None,    # "YYYY-MM-DD"
    end: str | None = None,      # "YYYY-MM-DD"
    fmt: str = "parquet",
    auto_adjust: bool = False,
    use_cache: bool = True,
) -> Path:
    """
    Descarga históricos desde Yahoo Finance para un símbolo.
    Estandariza: date, open, high, low, close, (adj_close), (volume)
    Guarda en CSV/Parquet usando utilitario común.
    """
     # 1) Verificar cache primero
    if use_cache:
        cached_data = cache.get(symbol, "yahoo", interval=interval, start=start, end=end, auto_adjust=auto_adjust)
        if cached_data is not None:
            print(f"✅ {symbol}: Usando datos en cache")
            # Guardar el dato cacheado en el formato solicitado
            safe = symbol.replace("/", "_").replace("-", "_").replace(".", "_").replace("^", "").upper()
            dest_dir = outdir / "yahoo" / safe
            dest_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{safe}_{interval}"
            path = save_timeseries(cached_data, dest_dir, base_name=base_name, fmt=fmt)
            return path

    # 2) Si no hay cache, descargar normalmente (tu código original)
    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
    # 1) Descargar con Ticker().history (más estable para un solo símbolo)
    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)

    if df is None or df.empty:
        raise RuntimeError(f"Sin datos para '{symbol}' en Yahoo Finance (interval={interval}, start={start}, end={end}).")

    # 2) Asegurar DataFrame 2D y mover índice fecha a columna
    df = df.reset_index()

    # 3) Normalizar nombres a minúsculas
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 4) Renombrar 'date'/'datetime' y 'adj close'
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    # En daily suele venir como 'date'; en intradía a veces 'datetime'

    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})

    # 5) Tipos correctos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open","high","low","close","adj_close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6) Si falta adj_close, usar close como fallback
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # 7) Selección segura de columnas disponibles (en orden estándar)
    cols = [c for c in ["date","open","high","low","close","adj_close","volume"] if c in df.columns]
    if "date" not in cols or "close" not in cols:
        raise RuntimeError(f"Estructura inesperada tras normalizar Yahoo para '{symbol}'. Columnas: {df.columns.tolist()}")

    df_std = (
        df[cols]
        .dropna(subset=["date","close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 8) Carpeta y guardado
    safe = symbol.replace("/", "_").replace("-", "_").replace(".", "_").replace("^", "").upper()
    dest_dir = outdir / "yahoo" / safe
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{safe}_{interval}"
    path = save_timeseries(df_std, dest_dir, base_name=base_name, fmt=fmt)
    return path
