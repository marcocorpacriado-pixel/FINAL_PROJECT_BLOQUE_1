
from pathlib import Path
import os, time, requests, pandas as pd
from .io_utils import save_timeseries
from ..cache import cache
_PREMIUM_HINTS = (
    "premium endpoint",
    "premium plan",
)

def _is_premium_message(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    msg = (data.get("Information") or data.get("Note") or data.get("Error Message") or "") .lower()
    return any(h in msg for h in _PREMIUM_HINTS)

def _fetch_raw(symbol: str, function: str, outputsize: str, api_key: str) -> dict:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_prices_alphavantage(
    symbol: str,
    outdir: Path,
    start: str | None = None,
    end: str | None = None,
    *,
    adjusted: bool = False,      # si da premium, haremos fallback automático
    outputsize: str = "compact", # "compact" más seguro en free
    fmt: str = "csv",
    use_cache : bool = True,
) -> Path:
    """
    Descarga precios diarios desde Alpha Vantage.
    Si 'adjusted=True' y el endpoint es premium, hace fallback a DAILY normal.
    """
     # 1) Verificar cache primero
    if use_cache:
        cached_data = cache.get(symbol, "alphavantage", start=start, end=end, adjusted=adjusted, outputsize=outputsize)
        if cached_data is not None:
            print(f"✅ {symbol}: Usando datos en cache (AlphaVantage)")
            safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
            symbol_dir = outdir / "alphavantage" / "prices" / safe_symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{safe_symbol}"
            out = save_timeseries(cached_data, symbol_dir, base_name=base_name, fmt=fmt)
            return out
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Falta ALPHAVANTAGE_API_KEY en .env")

    # 1) Intenta adjusted si lo pidió el usuario
    data = {}
    used_function = None
    if adjusted:
        used_function = "TIME_SERIES_DAILY_ADJUSTED"
        data = _fetch_raw(symbol, used_function, outputsize, api_key)
        if _is_premium_message(data):
            # fallback automático a DAILY
            used_function = "TIME_SERIES_DAILY"
            data = _fetch_raw(symbol, used_function, outputsize, api_key)

    else:
        used_function = "TIME_SERIES_DAILY"
        data = _fetch_raw(symbol, used_function, outputsize, api_key)

    # Mensajes típicos (rate limit, etc.)
    if isinstance(data, dict) and ("Information" in data or "Note" in data or "Error Message" in data):
        msg = data.get("Information") or data.get("Note") or data.get("Error Message")
        raise RuntimeError(f"Alpha Vantage dijo: {msg}")

    # Extrae serie
    key = "Time Series (Daily)"
    ts = data.get(key)
    if ts is None:
        raise RuntimeError(f"Respuesta inesperada: {str(data)[:200]}")

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index.name = "date"
    df = df.rename(columns=str.lower).reset_index()

    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",   # sólo si ADJUSTED
        "5. volume": "volume",              # DAILY
        "6. volume": "volume",              # ADJUSTED
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open","high","low","close","volume","adj_close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date")

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    cols = [c for c in ["date","open","high","low","close","adj_close","volume"] if c in df.columns]
    df_std = df[cols].dropna(subset=["date","close"])

    safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
    # misma estructura que te dejé
    symbol_dir = outdir / "alphavantage" / "prices" / safe_symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{safe_symbol}"
    out = save_timeseries(df_std, symbol_dir, base_name=base_name, fmt=fmt)

    # Respeta límites free
    time.sleep(12)
    return out






