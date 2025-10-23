# src/finlab/extractor/io_utils.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_timeseries(
    df: pd.DataFrame,
    outdir: Path,
    base_name: str,
    fmt: str = "csv",
) -> Path:
    """
    Guarda un DataFrame de series temporales en CSV o Parquet.
    - outdir: carpeta base donde guardar.
    - base_name: nombre de archivo sin extensión (p.ej. 'AAPL_1day').
    - fmt: 'csv' o 'parquet' (por defecto 'csv').
    Devuelve la ruta final.
    """
    fmt = (fmt or "csv").lower()
    _ensure_dir(outdir)

    # Normaliza tipos mínimos: fecha a string ISO si hace falta
    df_out = df.copy()
    if "date" in df_out.columns and not pd.api.types.is_string_dtype(df_out["date"]):
        # Mantener ISO legible; Parquet guarda tipos nativos igualmente
        try:
            df_out["date"] = pd.to_datetime(df_out["date"])
        except Exception:
            pass

    if fmt == "parquet":
        path = outdir / f"{base_name}.parquet"
        # Requiere pyarrow o fastparquet instalado
        df_out.to_parquet(path, index=False)
        return path
    elif fmt == "csv":
        path = outdir / f"{base_name}.csv"
        df_out.to_csv(path, index=False)
        return path
    else:
        raise ValueError("Formato no soportado. Usa 'csv' o 'parquet'.")
