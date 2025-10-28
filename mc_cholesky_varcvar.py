# scripts/mc_cholesky_varcvar.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# --- Config ---
inputs = [
    Path("data/twelvedata/SPY/SPY_1day.parquet"),
    Path("data/twelvedata/XAU_USD/XAU_USD_1day.parquet"),
    Path("data/twelvedata/BTC_USD/BTC_USD_1day.parquet"),
]
weights = {"SPY": 0.5, "Oro": 0.3, "BTC": 0.2}
days = 252
n_paths = 10_000
seed = 123
initial_value = 1.0
halflife_days = 90   # pondera más lo reciente (opcional)
outdir = Path("outputs/demo_macro_cholesky")
outdir.mkdir(parents=True, exist_ok=True)

# --- Cargar y limpiar series ---
series = {}
for p in inputs:
    c = Candles.from_any(p).clean(fill_method="ffill").to_business_days(fill=True)
    # Normaliza nombres bonitos
    stem = p.stem.split("_")[0]
    name = {"SPY": "SPY", "XAU": "Oro", "BTC": "BTC"}.get(stem, stem)
    series[name] = c

# Asegurar que el orden de weights coincide con series
series = {k: series[k] for k in ["SPY","Oro","BTC"]}

# --- Cartera ---
port = Portfolio(series=series, weights=weights, initial_value=initial_value)

# --- Simulación Monte Carlo (CHOLESKY) ---
paths = port.simulate(
    days=days,
    n_paths=n_paths,
    seed=seed,
    mc_method="cholesky",
    halflife_days=halflife_days,
)

# --- Métricas de resultado terminal ---
terminal_vals = paths[:, -1]
terminal_ret = terminal_vals / initial_value - 1.0
loss = -terminal_ret  # pérdidas positivas

def var_cvar(loss_array, alpha=0.95):
    loss_array = np.asarray(loss_array)
    var = float(np.quantile(loss_array, alpha))
    cvar = float(loss_array[loss_array >= var].mean()) if np.any(loss_array >= var) else var
    return var, cvar

levels = [0.95, 0.99]
metrics = []
for a in levels:
    v, cv = var_cvar(loss, alpha=a)
    metrics.append((a, v, cv))

p5 = float(np.percentile(terminal_vals, 5))
p95 = float(np.percentile(terminal_vals, 95))
mean_end = float(terminal_vals.mean())

# --- Guardar resumen Markdown ---
md_lines = [
    "# Monte Carlo (Cholesky) — VaR/CVaR sobre horizonte 252 días",
    "",
    f"- Trayectorias: **{n_paths}**",
    f"- Días simulados: **{days}**",
    f"- Semivida (EWMA): **{halflife_days}** días",
    "",
    "## Resultados terminales",
    f"- Valor final esperado: **{mean_end:.4f}**",
    f"- Banda 5–95% del valor final: **[{p5:.4f}, {p95:.4f}]**",
    "",
    "## Riesgo por Monte Carlo (pérdida sobre valor final)",
]
for a, v, cv in metrics:
    md_lines.append(f"- **Nivel {int(a*100)}%** → VaR: **{v:.4f}**, CVaR: **{cv:.4f}**")

(outdir / "varcvar_mc.md").write_text("\n".join(md_lines), encoding="utf-8")

# --- Gráficos ---
plt.figure(figsize=(8,4.5))
plt.hist(terminal_ret, bins=60, alpha=0.9)
plt.title("Distribución del retorno terminal (252 días) — Cholesky")
plt.xlabel("Retorno")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "terminal_returns_hist.png", bbox_inches="tight")

plt.figure(figsize=(8,4.5))
plt.hist(loss, bins=60, alpha=0.9)
for a, v, cv in metrics:
    plt.axvline(v, linestyle="--", label=f"VaR {int(a*100)}%: {v:.4f}")
    plt.axvline(cv, linestyle=":", label=f"CVaR {int(a*100)}%: {cv:.4f}")
plt.title("Distribución de pérdidas (loss = -retorno terminal) — Cholesky")
plt.xlabel("Pérdida")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / "loss_hist_with_varcvar.png", bbox_inches="tight")

print("✅ Hecho.")
print(f"→ {outdir / 'varcvar_mc.md'}")
print(f"→ {outdir / 'terminal_returns_hist.png'}")
print(f"→ {outdir / 'loss_hist_with_varcvar.png'}")
