# scripts/mc_compare_all.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# ---------- Config ----------
INPUTS = [
    Path("data/twelvedata/SPY/SPY_1day.parquet"),
    Path("data/twelvedata/XAU_USD/XAU_USD_1day.parquet"),
    Path("data/twelvedata/BTC_USD/BTC_USD_1day.parquet"),
]
WEIGHTS = {"SPY": 0.5, "Oro": 0.3, "BTC": 0.2}
DAYS = 252
N_PATHS = 5000
SEED = 123
INITIAL_VALUE = 1.0
HALFLIFE_DAYS = 90
BLOCK_LEN = 10

OUTDIR = Path("outputs/mc_benchmark")
OUTDIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("gbm", {"mc_method": "gbm"}),
    ("cholesky", {"mc_method": "cholesky", "halflife_days": HALFLIFE_DAYS}),
    ("copula", {"mc_method": "copula", "halflife_days": HALFLIFE_DAYS}),
    ("bootstrap", {"mc_method": "bootstrap", "block_len": BLOCK_LEN, "halflife_days": HALFLIFE_DAYS}),
]

# ---------- Carga y cartera ----------
series = {}
for p in INPUTS:
    c = Candles.from_any(p).clean(fill_method="ffill").to_business_days(fill=True)
    stem = p.stem.split("_")[0]
    name = {"SPY": "SPY", "XAU": "Oro", "BTC": "BTC"}.get(stem, stem)
    series[name] = c

series = {k: series[k] for k in ["SPY", "Oro", "BTC"]}  # orden fijo
port = Portfolio(series=series, weights=WEIGHTS, initial_value=INITIAL_VALUE)

# ---------- Ejecuta métodos ----------
results = {}
for label, kwargs in METHODS:
    paths = port.simulate(days=DAYS, n_paths=N_PATHS, seed=SEED, **kwargs)
    results[label] = paths

# ---------- Funciones auxiliares ----------
def summarize(paths: np.ndarray):
    mean = paths.mean(axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    end = paths[:, -1]
    return {
        "mean": mean, "p5": p5, "p95": p95,
        "end_mean": float(end.mean()),
        "end_p5": float(np.percentile(end, 5)),
        "end_p95": float(np.percentile(end, 95)),
    }

summaries = {k: summarize(v) for k, v in results.items()}

# ---------- Plot 1: bandas comparadas ----------
plt.figure(figsize=(10, 6))
t = np.arange(DAYS)
for label, sm in summaries.items():
    plt.plot(t, sm["mean"], linewidth=2, label=f"{label.capitalize()} — media")
    plt.fill_between(t, sm["p5"], sm["p95"], alpha=0.12)
plt.title("Monte Carlo — comparación de bandas 5–95% y medias (cartera)")
plt.xlabel("Días")
plt.ylabel("Valor de la cartera")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "mc_compare_bands.png", bbox_inches="tight")

# ---------- Plot 2: distribuciones del valor final ----------
plt.figure(figsize=(10, 6))
bins = 80
for label, paths in results.items():
    endvals = paths[:, -1]
    plt.hist(endvals, bins=bins, alpha=0.4, density=True, label=label.capitalize())
plt.title("Distribución del valor final — 252 días")
plt.xlabel("Valor final")
plt.ylabel("Densidad")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "mc_compare_terminal.png", bbox_inches="tight")

# ---------- Tabla Markdown con métricas terminales ----------
rows = []
for label, sm in summaries.items():
    rows.append({
        "Metodo": label,
        "End_mean": sm["end_mean"],
        "End_p5": sm["end_p5"],
        "End_p95": sm["end_p95"],
    })
df = pd.DataFrame(rows).sort_values("Metodo")
df.to_csv(OUTDIR / "mc_summary_terminal.csv", index=False)

md = ["# Resumen Monte Carlo — Comparativa métodos",
      "",
      "| Método | Valor final esperado | p5 | p95 |",
      "|:--|--:|--:|--:|"]
for _, r in df.iterrows():
    md.append(f"| {str(r['Metodo']).capitalize()} | {r['End_mean']:.4f} | {r['End_p5']:.4f} | {r['End_p95']:.4f} |")
(Path(OUTDIR) / "mc_summary_terminal.md").write_text("\n".join(md), encoding="utf-8")

print("✅ Hecho.")
print(f"→ {OUTDIR / 'mc_compare_bands.png'}")
print(f"→ {OUTDIR / 'mc_compare_terminal.png'}")
print(f"→ {OUTDIR / 'mc_summary_terminal.md'}")
