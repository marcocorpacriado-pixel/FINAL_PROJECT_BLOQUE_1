from pathlib import Path
import os, time, requests, pandas as pd

def fetch_prices_alphavantage(
    symbol: str,
    outdir: Path,
    start: str | None = None,
    end: str | None = None,
    *,
    adjusted: bool = False,     # usar DAILY (free) por defecto
    outputsize: str = "compact" # "compact" (≈100 últimos) es lo más seguro en free
) -> Path:
    """
    Descarga precios diarios desde Alpha Vantage en modo compatible con el plan gratuito.
    - adjusted=False usa TIME_SERIES_DAILY (free).
    - outputsize="compact" evita respuestas premium/rate limit.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Falta ALPHAVANTAGE_API_KEY en .env")

    function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"

    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,  # "compact" ~100 datos; "full" puede disparar límites
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Errores típicos del free tier
    if isinstance(data, dict) and ("Information" in data or "Note" in data or "Error Message" in data):
        msg = data.get("Information") or data.get("Note") or data.get("Error Message")
        raise RuntimeError(f"Alpha Vantage dijo: {msg}")

    # Estructura según endpoint
    key = "Time Series (Daily)" if function == "TIME_SERIES_DAILY" else "Time Series (Daily)"
    ts = data.get(key)
    if ts is None:
        raise RuntimeError(f"Respuesta inesperada: {str(data)[:200]}")

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index.name = "date"
    df = df.sort_index()

    # Normalizamos nombres si vienen ajustados o no
    rename_map = {
        "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
        "5. adjusted close": "adj_close", "5. volume": "volume", "6. volume": "volume",
        "7. dividend amount": "dividend", "8. split coefficient": "split"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Recorte opcional por fecha (índice es string YYYY-MM-DD)
    if start: df = df[df.index >= start]
    if end:   df = df[df.index <= end]

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{symbol.upper()}.csv"
    df.to_csv(out, index=True)

    # Respeta límites gratuitos
    time.sleep(12)  # free tier suele permitir 5 req/min -> ~12s por seguridad
    return out

