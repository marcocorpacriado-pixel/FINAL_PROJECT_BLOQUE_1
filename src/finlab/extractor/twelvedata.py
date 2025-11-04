# src/finlab/extractor/twelvedata.py
from pathlib import Path
import os, time, requests, pandas as pd
from .io_utils import save_timeseries
from ..cache import cache

def fetch_prices_twelvedata(
    symbol: str,
    outdir: Path,
    interval: str = "1day",
    start: str | None = None,
    end: str | None = None,
    *,
    fmt: str = "csv",
    use_cache: bool = True,
) -> Path:
    """
    Descarga datos desde la API gratuita de TwelveData.
    Compatible con bolsa, forex y criptos.
    Permite guardar en CSV o Parquet (fmt).
    """
    # 1) Verificar cache primero
    if use_cache:
        cached_data = cache.get(symbol, "twelvedata", interval=interval, start=start, end=end)
        if cached_data is not None:
            print(f"✅ {symbol}: Usando datos en cache (TwelveData)")
            safe_symbol = symbol.replace("/", "_").replace("\\", "_")
            symbol_dir = outdir / safe_symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_timeseries(cached_data, symbol_dir, base_name=f"{safe_symbol}_{interval}", fmt=fmt)
            return out_path
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Falta TWELVEDATA_API_KEY en el archivo .env")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": 5000,  # máximo en el plan gratuito
        "format": "JSON",
    }

    if start: params["start_date"] = start
    if end:   params["end_date"] = end

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "values" not in data:
        raise RuntimeError(f"❌ Respuesta inesperada de TwelveData: {str(data)[:200]}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df = df.rename(columns=str.lower)

    # Normaliza el símbolo para carpetas/archivos
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    symbol_dir = outdir / safe_symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Asegura esquema estándar básico (date, open, high, low, close, volume)
    df_std = df.rename(columns={
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })
    # Ordena columnas si existen
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df_std.columns]
    df_std = df_std[cols]

    # Guarda usando el utilitario (CSV o Parquet)
    out_path = save_timeseries(df_std, symbol_dir, base_name=f"{safe_symbol}_{interval}", fmt=fmt)

    time.sleep(1.2)
    return out_path
