from pathlib import Path
import sys

# Añadimos la ruta a /src para poder importar finlab.*
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from finlab.models.candles import Candles   # ✅ cambio aquí
from finlab.models.portfolio import Portfolio

# === Carga los datos ===
btc = Candles.from_any(Path("data/twelvedata/BTC_USD/BTC_USD_1day.parquet"))
eth = Candles.from_any(Path("data/twelvedata/ETH_USD/ETH_USD_1day.parquet"))

# Limpieza básica + días laborables
btc = btc.clean(fill_method="ffill").to_business_days(fill=True)
eth = eth.clean(fill_method="ffill").to_business_days(fill=True)

# === Crea la cartera ===
pf = Portfolio(series={"BTC": btc, "ETH": eth}, weights={"BTC": 0.5, "ETH": 0.5})

# === Prueba los 4 métodos de Monte Carlo ===
for method in ["gbm", "cholesky", "copula", "bootstrap"]:
    print(f"\n--- Método Monte Carlo: {method.upper()} ---")
    paths = pf.simulate(
        days=252,
        n_paths=500,
        seed=123,
        mc_method=method,
        halflife_days=90,   # más peso a los últimos 3 meses
        block_len=10
    )
    sm = pf.summarize_paths(paths)
    print(f"Final esperado: {sm['end_mean']:.4f} | p5={sm['end_p5']:.4f} | p95={sm['end_p95']:.4f}")
