from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union 
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64tz_dtype


def _normalize_dates_no_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'date' a datetime, si trae tz -> UTC y quita tz; normaliza a día."""
    df = df.copy()
    d = pd.to_datetime(df["date"], errors="coerce")
    # si hay tz, unifica a UTC y quita tz
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    df["date"] = d.dt.normalize()
    return df


# Mapeos de columnas por proveedor → columnas estándar
_COL_MAPS: List[Dict[str, str]] = [
    # AlphaVantage
    {
        "1. open": "open",
        "2. high": "high", 
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume",
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
    # Yahoo Finance
    {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume",
    },
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for m in _COL_MAPS:
        if any(col in df.columns for col in m.keys()):
            df = df.rename(columns={src.lower(): dst for src, dst in m.items() if src.lower() in df.columns})
            break

    if "date" not in df.columns and isinstance(df.index, pd.Index):
        try:
            _ = pd.to_datetime(df.index)
            df = df.rename_axis("date").reset_index()
        except Exception:
            pass

    needed = {"date", "open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Faltan columnas mínimas {needed}. Columnas disponibles: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").dropna(subset=["date", "close"]).reset_index(drop=True)
    return df

@dataclass
class Candles:
    symbol: str
    frame: pd.DataFrame

    @classmethod
    def from_parquet(cls, path: Path, symbol: Optional[str] = None) -> "Candles":
        df = pd.read_parquet(path)
        df = _normalize_columns(df)
        sym = symbol or Path(path).stem.split("_")[0].upper()
        return cls(symbol=sym, frame=df)

    @classmethod
    def from_any(cls, path: Path, symbol: Optional[str] = None) -> "Candles":
        ext = path.suffix.lower()
        if ext == ".parquet":
            return cls.from_parquet(path, symbol=symbol)
        elif ext == ".csv":
            return cls.from_csv(path, symbol=symbol)
        else:
            raise ValueError(f"Formato no soportado: {ext}")

    @classmethod
    def from_csv(cls, path: Path, symbol: Optional[str] = None) -> "Candles":
        df = pd.read_csv(path)
        df = _normalize_columns(df)
        sym = symbol or Path(path).stem.split("_")[0].upper()
        return cls(symbol=sym, frame=df)

    def to_dataframe(self) -> pd.DataFrame:
        return self.frame.copy()

    def to_records(self) -> list[dict[str, Any]]:
        return self.frame.to_dict(orient="records")

    def closes(self) -> pd.Series:
        return self.frame["close"]

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.frame.head(n)

    def between(self, start: Optional[str] = None, end: Optional[str] = None) -> "Candles":
        df = self.frame.copy()
        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]
        return Candles(symbol=self.symbol, frame=df.reset_index(drop=True))

    def resample(self, rule: str = "W") -> "Candles":
        df = self.frame.copy().set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
        return Candles(symbol=self.symbol, frame=out)

    def log_returns(self, use_adj_close: bool = True) -> pd.Series:
        """Retornos logarítmicos indexados por FECHA (día), sin zona horaria."""
        df = self.frame.copy()

        price_col = "adj_close" if (use_adj_close and "adj_close" in df.columns) else "close"

        dates = pd.to_datetime(df["date"], errors="coerce")
        # si viene con tz, pásalo a UTC y quita tz
        if is_datetime64tz_dtype(dates):
            dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
        # normaliza a fecha (00:00)
        dates = dates.dt.normalize()

        close = pd.to_numeric(df[price_col], errors="coerce")

        s = pd.Series(close.values, index=dates, name="close").dropna()
        # por si hay duplicados de día, quédate con el último y ordena
        s = s[~s.index.duplicated(keep="last")].sort_index()

        r = np.log(s / s.shift(1))
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        r.index.name = "date"
        return r


    def plot(self, title: Optional[str] = None):
        import matplotlib.pyplot as plt
        df = self.frame
        plt.figure(figsize=(10, 4))
        plt.plot(df["date"], df["close"], linewidth=1.5)
        plt.title(title or f"{self.symbol} — Close")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    def clean(
    self,
    drop_duplicates: bool = True,
    sort: bool = True,
    fill_method: str | None = None,
    clip_outliers: tuple[float, float] | None = None,
) -> "Candles":
        df = self.frame.copy()

        # normaliza fechas y tz SIEMPRE aquí
        df = _normalize_dates_no_tz(df)

        if drop_duplicates:
            df = df.drop_duplicates(subset=["date"], keep="last")
        if sort:
            df = df.sort_values("date")

        if clip_outliers is not None and "close" in df.columns:
            lo_q, hi_q = clip_outliers
            lo, hi = df["close"].quantile([lo_q, hi_q])
            df["close"] = df["close"].clip(lower=lo, upper=hi)

        if fill_method in {"ffill", "bfill"}:
            df = df.set_index("date")
            cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
            if fill_method == "ffill":
                df[cols] = df[cols].ffill()
            else:
                df[cols] = df[cols].bfill()
            df = df.reset_index()

        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
        return Candles(symbol=self.symbol, frame=df)


    # ===========================
    #  MÉTODO SIMPLIFICADO
    # ===========================
    def to_business_days(
        self,
        fill_method: str = "previous_close",
        normalize_timezone: bool = True,
        target_timezone: Optional[str] = "UTC",
        calendar: str = "weekends",
        safe_ohlc: bool = True,
        prefer_adj_close: bool = True,
    ) -> "Candles":
        """
        Versión simplificada: por defecto NO rellena datos (fill_method="none")
        """
        # 1. Copiar y validar datos
        df = self.frame.copy()
        if df.empty:
            return Candles(symbol=self.symbol, frame=df)

        if "date" not in df.columns or "close" not in df.columns:
            raise ValueError(f"Faltan columnas 'date' o 'close' en {self.symbol}")

        # 2. Limpiar fechas
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

        # 3. Normalizar timezone si es necesario
        if normalize_timezone and is_datetime64tz_dtype(df["date"]):
            tz = target_timezone or "UTC"
            df["date"] = df["date"].dt.tz_convert(tz).dt.tz_localize(None)

        # 4. Elegir columna de precio
        price_col = "adj_close" if (prefer_adj_close and "adj_close" in df.columns) else "close"

        # 5. Crear calendario de días hábiles
        start, end = df["date"].min(), df["date"].max()
        business_dates = self._get_business_calendar(start, end, calendar)

        # 6. Reindexar a días hábiles
        base = df.set_index("date").sort_index()
        aligned = base.reindex(business_dates)

        # 7. Aplicar relleno SOLO si se solicita explícitamente
        if fill_method == "none":
            filled = aligned
        elif fill_method == "previous_close":
            # Solo forward-fill del precio principal
            filled = aligned.copy()
            filled[price_col] = filled[price_col].ffill()
            
            # Sincronizar close/adj_close si existen ambos
            if price_col == "adj_close" and "close" in filled.columns:
                filled["close"] = filled["close"].fillna(filled["adj_close"])
            elif price_col == "close" and "adj_close" in filled.columns:
                filled["adj_close"] = filled["adj_close"].fillna(filled["close"])
                
        else:
            # Para otros métodos, simplemente dropear NaNs
            filled = aligned

        # 8. Limpiar resultados
        filled = filled.dropna(subset=[price_col])
        
        filled.index.name = "date"
        out = filled.reset_index()
        return Candles(symbol=self.symbol, frame=out)

    def _get_business_calendar(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        calendar: str = "weekends"
    ) -> pd.DatetimeIndex:
        """Calendario simple de días hábiles."""
        if calendar.upper() == "NYSE":
            try:
                import pandas_market_calendars as mcal
                nyse = mcal.get_calendar("NYSE")
                days = nyse.valid_days(start_date=start, end_date=end)
                return pd.DatetimeIndex(days.tz_convert("UTC").tz_localize(None))
            except Exception:
                return pd.date_range(start=start, end=end, freq="B")
        else:
            return pd.date_range(start=start, end=end, freq="B")

    def detect_asset_frequency(self) -> str:
        """
        Detecta automáticamente la frecuencia de trading del activo.
        Returns: 'business_days' | '24_7' | 'unknown'
        """
        if self.frame.empty:
            return 'unknown'
        
        df = self.frame.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calcular diferencias entre días consecutivos
        date_diffs = df['date'].diff().dt.days
        
        if len(date_diffs) < 5:
            return 'unknown'
        
        # Analizar patrones de frecuencia
        weekday_distribution = df['date'].dt.dayofweek.value_counts(normalize=True)
        weekend_ratio = weekday_distribution.get(5, 0) + weekday_distribution.get(6, 0)
        
        # Si hay datos en fin de semana > 10%, es 24/7
        if weekend_ratio > 0.1:
            return '24_7'
        
        # Si la mayoría de diferencias son 1-3 días, es business_days
        typical_business_gaps = date_diffs.between(1, 3).mean()
        if typical_business_gaps > 0.8:
            return 'business_days'
        
        return 'unknown'

    # Helpers eliminados por simplicidad (no se usan en el modo simplificado)
    def _returns_based_fill(self, df: pd.DataFrame, price_col: str, safe_ohlc: bool) -> pd.DataFrame:
        return df
    
    def _previous_close_fill(self, df: pd.DataFrame, price_col: str, safe_ohlc: bool) -> pd.DataFrame:
        return df
    
    def _linear_fill(self, df: pd.DataFrame, price_col: str, safe_ohlc: bool) -> pd.DataFrame:
        return df
    
    def _sync_price_columns(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        return df
    
    def _apply_safe_ohlc(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        return df
    
    def _enforce_ohlc_invariants(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        return df
    
    def _select_price_col(self, df: pd.DataFrame, prefer_adj_close: bool = True) -> str:
        if prefer_adj_close and "adj_close" in df.columns:
            return "adj_close"
        return "close"


