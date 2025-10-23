# run_demo_portfolio.py
from pathlib import Path
from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# --- rutas a tus datos (ajusta si usas otros símbolos) ---
BTC = Path("data/twelvedata/BTC_USD/BTC_USD_1day.parquet")
ETH = Path("data/twelvedata/ETH_USD/ETH_USD_1day.parquet")

# comprobación rápida
assert BTC.exists(), f"No se encuentra {BTC}"
assert ETH.exists(), f"No se encuentra {ETH}"

# --- carga y limpieza ---
btc = Candles.from_any(BTC).clean(fill_method="ffill").to_business_days(fill=True)
eth = Candles.from_any(ETH).clean(fill_method="ffill").to_business_days(fill=True)

# --- definición de cartera ---
weights = {"BTC": 0.6, "ETH": 0.4}
pf = Portfolio(series={"BTC": btc, "ETH": eth}, weights=weights, initial_value=1.0)

# --- generar informe markdown ---
md = pf.report(freq=252, rf=0.0, mc_days=252, mc_paths=2000, seed=123)
out_md = Path("reports/portfolio.md")
out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text(md, encoding="utf-8")
print("\n=== Portfolio Report (resumen) ===\n")
print(md)

# --- generar gráficos ---
img_dir = Path("reports/imgs")
pf.plots_report(save_dir=str(img_dir), mc_days=252, mc_paths=1000, seed=123, logy=True, align="inner")
print(f"\n✅ Gráficos guardados en: {img_dir.resolve()}")

# Comparativa rápida de métodos con recencia (halflife 60 días)
for method in ["gbm", "cholesky", "copula", "bootstrap"]:
    paths = pf.simulate(days=252, n_paths=1000, seed=123, mc_method=method, block_len=10, halflife_days=60)
    sm = pf.summarize_paths(paths)
    print(f"[{method}] hl=60d  mean_end={sm['end_mean']:.4f} | p5={sm['end_p5']:.4f} | p95={sm['end_p95']:.4f}")



##########################################

from pathlib import Path
from finlab.domain.candles import Candles
from finlab.models.portfolio import Portfolio

# === Carga los datos ===
btc = Candles.from_any(Path("data/twelvedata/BTC_USD/BTC_USD_1day.parquet"))
eth = Candles.from_any(Path("data/twelvedata/ETH_USD/ETH_USD_1day.parquet"))

btc = btc.clean().to_business_days()
eth = eth.clean().to_business_days()

# === Crea la cartera ===
pf = Portfolio(components={"BTC": btc, "ETH": eth}, weights={"BTC": 0.5, "ETH": 0.5})

# === Prueba los 4 métodos ===
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
