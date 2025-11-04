from pathlib import Path
from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# ============================================================
# CONFIGURACIÓN DE LA CARTERA
# ============================================================

series = {
    "GOLD": Candles.from_parquet(Path("data/yahoo/GOLD/GOLD_1d.parquet"), symbol="GOLD"),
    "BTC": Candles.from_parquet(Path("data/yahoo/BTC/BTC_1d.parquet"), symbol="BTC"),
    "QQQ": Candles.from_parquet(Path("data/yahoo/QQQ/QQQ_1d.parquet"), symbol="QQQ"),
    "DEFENSE": Candles.from_parquet(Path("data/yahoo/DEFENSE/DEFENSE_1d.parquet"), symbol="DEFENSE"),
    "HYG": Candles.from_parquet(Path("data/yahoo/HYG/HYG_1d.parquet"), symbol="HYG"),
    "EEM": Candles.from_parquet(Path("data/yahoo/EEM/EEM_1d.parquet"), symbol="EEM"),
}

weights = {
    "GOLD": 0.10,
    "BTC": 0.10,
    "QQQ": 0.35,
    "DEFENSE": 0.15,
    "HYG": 0.10,
    "EEM": 0.20,
}

# Crear cartera
pf = Portfolio(series=series, weights=weights)

# ============================================================
# SIMULACIÓN MONTE CARLO COMPARATIVA
# ============================================================

report = pf.monte_carlo_overview(
    days=252,                   # 1 año simulado
    n_paths=2000,               # número total de trayectorias por método
    seed=42,                    # reproducibilidad
    freq=252,                   # frecuencia anual (días de mercado)
    rf=0.00,                    # tipo libre de riesgo
    show_paths_per_method=50,   # nº de trayectorias visibles por método
    alpha_band=(5, 95),         # bandas 5%-95%
    save_path="outputs/mc_compare/mc_methods_bands.png"
)

print("\n=== INFORME MONTE CARLO ===")
print(report)
