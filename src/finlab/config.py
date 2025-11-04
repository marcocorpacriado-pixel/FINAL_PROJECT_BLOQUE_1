from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Límites de simulaciones
    MAX_SIMULATION_PATHS: int = 10000
    MAX_SIMULATION_DAYS: int = 1000
    
    # Directorios
    DEFAULT_DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path(".cache")
    
    # Configuración de APIs
    REQUEST_TIMEOUT: int = 30
    YAHOO_RETRIES: int = 3
    
    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_HOURS: int = 24
    
    # Rendimiento
    MAX_WORKERS: int = 5
    CHUNK_SIZE: int = 1000

# Instancia global de configuración
config = Config()