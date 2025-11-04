from pathlib import Path
import pandas as pd
import time
from typing import Optional, Any
import hashlib
import pickle
from .config import config

class DataCache:
    """Sistema de cache para datos financieros"""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, symbol: str, provider: str, **params) -> str:
        """Genera clave única para el cache"""
        param_str = str(sorted(params.items()))
        key_str = f"{provider}_{symbol}_{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Verifica si el cache sigue siendo válido"""
        if not cache_file.exists():
            return False
        
        cache_age = time.time() - cache_file.stat().st_mtime
        return cache_age < (self.ttl_hours * 3600)
    
    def get(self, symbol: str, provider: str, **params) -> Optional[pd.DataFrame]:
        """Obtiene datos del cache si existen y son válidos"""
        if not config.CACHE_ENABLED:
            return None
            
        cache_key = self._get_cache_key(symbol, provider, **params)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if self._is_cache_valid(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"⚠️  Error leyendo cache: {e}")
                return None
        return None
    
    def set(self, symbol: str, provider: str, data: pd.DataFrame, **params) -> None:
        """Guarda datos en el cache"""
        if not config.CACHE_ENABLED:
            return
            
        cache_key = self._get_cache_key(symbol, provider, **params)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            data.to_parquet(cache_file)
        except Exception as e:
            print(f"⚠️  Error guardando cache: {e}")

# Instancia global del cache
cache = DataCache()