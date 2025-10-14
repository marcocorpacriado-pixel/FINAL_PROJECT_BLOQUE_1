from pathlib import Path
import os, time, requests, pandas as pd

def fetch_prices_twelvedata(
    symbol: str,
    outdir: Path,
    interval: str = "1day",
    start: str | None = None,
    end: str | None = None
) -> Path:
    """
    Descarga datos desde la API gratuita de TwelveData.
    Compatible con bolsa, forex y criptos.
    """
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

      # --- Crear carpeta base ---
    outdir.mkdir(parents=True, exist_ok=True)

    # Normaliza el símbolo (evita caracteres problemáticos como "/")
    safe_symbol = symbol.replace("/", "_").replace("\\", "_")

    # Crea la subcarpeta específica del símbolo
    symbol_dir = outdir / safe_symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Guarda el CSV dentro de esa subcarpeta
    out = symbol_dir / f"{safe_symbol}_{interval}.csv"
    df.to_csv(out, index=False)

    time.sleep(1.2)
    return out

