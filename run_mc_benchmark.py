# run_mc_benchmark.py
from pathlib import Path
from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# --- Carga rápida de una cartera de ejemplo (ajusta a tus rutas locales)
series = {
    "SPY": Candles.from_parquet(Path("data/yahoo/SPY/SPY_1d.parquet"), symbol="SPY"),
    "GLD": Candles.from_parquet(Path("data/yahoo/GLD/GLD_1d.parquet"), symbol="GLD"),
    "BTC": Candles.from_parquet(Path("data/yahoo/BTC/BTC_1d.parquet"), symbol="BTC"),
}
weights = {"SPY": 0.50, "GLD": 0.30, "BTC": 0.20}
pf = Portfolio(series=series, weights=weights)

# Carpeta de salida para la imagen del README
save_path = Path("docs/assets/mc_benchmark_all_methods.png")
save_path.parent.mkdir(parents=True, exist_ok=True)

# --- Benchmark Monte Carlo: 252 días, 5000 paths por método
report = pf.monte_carlo_overview(
    days=252,
    n_paths=5000,                # <- tu función usa n_paths
    seed=42,
    freq=252,
    rf=0.02,
    show_paths_per_method=0,     # 0 para no sobrecargar el gráfico del README
    title="Benchmark Monte Carlo (GBM vs Cholesky vs Cópula vs Bootstrap)",
    save_path=str(save_path),
)

print("\n===== RESUMEN (inclúyelo en README si quieres) =====\n")
print(report)


