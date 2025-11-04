from pathlib import Path
import os, time, requests, pandas as pd
from .io_utils import save_timeseries
from ..cache import cache

def _request_marketstack(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_prices_marketstack(
    symbol: str,
    outdir: Path,
    start: str | None = None,   
    end: str | None = None,     
    *,
    fmt: str = "csv",
    use_cache:bool = True,           
) -> Path:
    """
    Descarga precios EOD desde MarketStack (free tier).
    Estandariza columnas y guarda en CSV o Parquet.
    """
    # 1) Verificar cache primero
    if use_cache:
        cached_data = cache.get(symbol, "marketstack", start=start, end=end)
        if cached_data is not None:
            print(f"✅ {symbol}: Usando datos en cache (MarketStack)")
            safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
            symbol_dir = outdir / "marketstack" / safe_symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{safe_symbol}"
            out = save_timeseries(cached_data, symbol_dir, base_name=base_name, fmt=fmt)
            return out
        
    api_key = os.getenv("MARKETSTACK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Falta MARKETSTACK_API_KEY en .env")

    params = {"access_key": api_key, "symbols": symbol, "limit": 1000}

    # Intento http (clásico free)
    url_http = "http://api.marketstack.com/v1/eod"
    data = _request_marketstack(url_http, params)
    items = data.get("data")

    # Fallback https si vacío
    if not items:
        url_https = "https://api.marketstack.com/v1/eod"
        try:
            data_https = _request_marketstack(url_https, params)
            items = data_https.get("data")
            data = data_https
        except Exception:
            pass

    # Errores explícitos
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"MarketStack error: {data['error']}")

    if not items:
        raise RuntimeError(
            f"Sin datos de MarketStack para '{symbol}' (start={start}, end={end}). "
            f"Respuesta: {str(data)[:200]}"
        )

    df = pd.DataFrame(items)
    needed = {"date", "close"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"Faltan columnas esperadas. Columnas: {df.columns.tolist()}")

    # Tipos y orden
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    # Renombrado estándar (si existen)
    df_std = df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "adj_close": "adj_close",
    })
    cols = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df_std.columns]
    df_std = df_std[["date"] + [c for c in cols if c != "date"]].dropna(subset=["date", "close"])

    safe_symbol = symbol.replace("/", "_").replace("\\", "_").upper()
    symbol_dir = outdir / "marketstack" / safe_symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{safe_symbol}"
    out = save_timeseries(df_std, symbol_dir, base_name=base_name, fmt=fmt)

    time.sleep(1.2)
    return out


