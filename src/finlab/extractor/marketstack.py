from pathlib import Path
import os
import requests
import pandas as pd
import time


def _request_marketstack(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_prices_marketstack(
    symbol: str,
    outdir: Path,
    start: str | None = None,   # YYYY-MM-DD
    end: str | None = None      # YYYY-MM-DD
) -> Path:
    """
    Descarga precios EOD desde MarketStack (free tier).
    Guarda CSV en outdir/<SYMBOL>/<SYMBOL>.csv

    Maneja casos de lista vacía y prueba http -> https si viene sin datos.
    """
    api_key = os.getenv("MARKETSTACK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Falta MARKETSTACK_API_KEY en .env")

    params = {
        "access_key": api_key,
        "symbols": symbol,
        "limit": 1000
    }

    # 1) Primero intenta con http (free tier clásico)
    url_http = "http://api.marketstack.com/v1/eod"
    data = _request_marketstack(url_http, params)

    items = data.get("data")
    # 2) Si items es None o vacío, intenta con https por si el proveedor ha cambiado algo
    if not items:
        url_https = "https://api.marketstack.com/v1/eod"
        try:
            data_https = _request_marketstack(url_https, params)
            items = data_https.get("data")
            data = data_https  # para mensajes de error, mantener lo último
        except Exception:
            # si https también falla, seguimos con lo que haya
            pass

    # 3) Errores explícitos
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"MarketStack error: {data['error']}")

    # 4) Sin datos (lista vacía)
    if not items:
        raise RuntimeError(
            f"Sin datos de MarketStack para symbol='{symbol}' "
            f"en rango start={start} end={end}. "
            f"Prueba sin fechas o con otro símbolo. Respuesta: {str(data)[:200]}"
        )

    # 5) Construir DataFrame
    df = pd.DataFrame(items)
    # Validar columnas esperadas
    needed = {"date", "close"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"Faltan columnas esperadas en MarketStack. Columnas: {df.columns.tolist()}")

    # Orden por fecha ascendente
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Filtrado local por fechas (opcional)
   # --- Filtrado local por fechas (opcional, manejando zona horaria) ---
    if "date" in df.columns:
    # Asegura que las fechas sean timezone-naive (sin UTC)
        if pd.api.types.is_datetime64tz_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_convert(None)

        if start:
         start_dt = pd.to_datetime(start)
         df = df[df["date"] >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            df = df[df["date"] <= end_dt]


    # 6) Guardar
    outdir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")
    symbol_dir = outdir / safe_symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    out = symbol_dir / f"{safe_symbol}.csv"
    df.to_csv(out, index=False)

    # Respetar un poco rate limit
    time.sleep(1.2)
    return out

