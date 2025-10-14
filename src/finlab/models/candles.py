from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np  # usar np en vez de pd.np

# Mapeos de columnas por proveedor → columnas estándar
_COL_MAPS: List[Dict[str, str]] = [
    # AlphaVantage (TIME_SERIES_DAILY o ADJUSTED)
    {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume",
        "7. dividend amount": "dividend",
        "8. split coefficient": "split",
    },
    # TwelveData
    {
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
    # MarketStack
    {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "adj_close": "adj_close",
    },
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # estandariza nombres (lower, strip)
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # aplica el primer mapeo que encaje
    for m in _COL_MAPS:
        if any(col in df.columns for col in m.keys()):
            df = df.rename(columns={src.lower(): dst for src, dst in m.items() if src.lower() in df.columns})
            break

    # si aún no hay 'date' pero el índice parece fecha (caso AlphaVantage)
    if "date" not in df.columns and isinstance(df.index, pd.Index):
        try:
            _ = pd.to_datetime(df.index)
            df = df.rename_axis("date").reset_index()
        except Exception:
            pass

    # columnas mínimas
    needed = {"date", "open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Faltan columnas mínimas {needed}. Columnas disponibles: {df.columns.tolist()}")

    # tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ordenar y limpiar NaN de fecha/cierre
    df = df.sort_values("date").dropna(subset=["date", "close"]).reset_index(drop=True)
    return df


@dataclass
class Candles:
    symbol: str
    frame: pd.DataFrame  # columnas estándar: date, open, high, low, close, (volume, adj_close, ...)

    @classmethod
    def from_csv(cls, path: Path, symbol: Optional[str] = None) -> "Candles":
        """Carga un CSV de cualquier proveedor y lo normaliza al esquema estándar."""
        df = pd.read_csv(path)
        df = _normalize_columns(df)
        # Si no pasaron símbolo, infiere del nombre de archivo/carpeta
        sym = symbol or Path(path).stem.split("_")[0].upper()
        return cls(symbol=sym, frame=df)

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve una copia del DataFrame normalizado."""
        return self.frame.copy()

    def to_records(self) -> list[dict[str, Any]]:
        """Lista de dicts (útil para serializar)."""
        return self.frame.to_dict(orient="records")

    def closes(self) -> pd.Series:
        return self.frame["close"]

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.frame.head(n)

    # ---------- utilidades añadidas ----------
    def between(self, start: Optional[str] = None, end: Optional[str] = None) -> "Candles":
        """Devuelve una nueva Candles filtrada por fechas (inclusive)."""
        df = self.frame.copy()
        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]
        return Candles(symbol=self.symbol, frame=df.reset_index(drop=True))

    def resample(self, rule: str = "W") -> "Candles":
        """
        Re-muestrea OHLC. Regla típica: 'W' (semanal), 'M' (mensual), 'D' (diaria).
        - open: primer valor del periodo
        - high: máximo
        - low:  mínimo
        - close: último valor del periodo
        - volume: suma (si existe)
        """
        df = self.frame.copy().set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
        return Candles(symbol=self.symbol, frame=out)

    def log_returns(self) -> pd.Series:
        """Retornos logarítmicos a partir de 'close' (limpios y sin inf)."""
        c = pd.to_numeric(self.frame["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        r = np.log(c / c.shift(1))
        return r.dropna()


    def plot(self, title: Optional[str] = None):
        """Gráfico simple del 'close'."""
        import matplotlib.pyplot as plt
        df = self.frame
        plt.figure(figsize=(10, 4))
        plt.plot(df["date"], df["close"], linewidth=1.5)
        plt.title(title or f"{self.symbol} — Close")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def clean(self,
              drop_duplicates: bool = True,
              sort: bool = True,
              fill_method: str | None = None,  # "ffill" o "bfill" o None
              clip_outliers: tuple[float, float] | None = None  # p.ej. (0.01, 0.99)
              ) -> "Candles":
        """Limpia la serie: dupes, orden, relleno opcional y recorte de atípicos en 'close'."""
        df = self.frame.copy()

        # quitar duplicados por fecha
        if drop_duplicates:
            df = df.drop_duplicates(subset=["date"], keep="last")

        # ordenar por fecha
        if sort:
            df = df.sort_values("date")

        # outliers (en close) si se pide
        if clip_outliers is not None and "close" in df.columns:
            lo_q, hi_q = clip_outliers
            lo, hi = df["close"].quantile([lo_q, hi_q])
            df["close"] = df["close"].clip(lower=lo, upper=hi)

        # rellenar huecos de OHLC si se pide
        if fill_method in {"ffill", "bfill"}:
            df = df.set_index("date")
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].fillna(method=fill_method)
            df = df.reset_index()

        # quitar filas sin 'close' o fecha
        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
        return Candles(symbol=self.symbol, frame=df)

    def to_business_days(self, fill: bool = True) -> "Candles":
        """Re-muestrea a días laborables ('B') y rellena huecos si fill=True."""
        df = self.frame.copy().set_index("date").sort_index()
        df = df.asfreq("B")  # calendario de días laborables
        if fill:
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        df = df.dropna(subset=["close"]).reset_index()
        return Candles(symbol=self.symbol, frame=df)

    def validate(self, strict: bool = True) -> "Candles":
        """
        Verifica que la serie tenga como mínimo: date, open, high, low, close.
        - strict=True: lanza ValueError si faltan columnas o hay 'close' no numérico.
        - strict=False: intenta convertir y deja NaN si no se puede.
        """
        df = self.frame
        needed = {"date", "open", "high", "low", "close"}
        missing = needed.difference(df.columns)
        if missing:
            if strict:
                raise ValueError(f"Faltan columnas {missing} en {self.symbol}")
            # si no es estricto, no hacemos magia: devolvemos igual
            return self

        # tipos
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # quitar filas inválidas
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return Candles(symbol=self.symbol, frame=df)
