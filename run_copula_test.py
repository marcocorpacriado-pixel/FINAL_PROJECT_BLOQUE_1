# run_copula_test.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

OUTDIR = Path("outputs/mixed_demo")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Usa los parquet ya descargados por tu script anterior (run_mixed_portfolio.py)
DATA = {
    "GOLD":    Path("data/yahoo/GOLD/GOLD_1d.parquet"),
    "BTC":     Path("data/yahoo/BTC/BTC_1d.parquet"),
    "QQQ":     Path("data/yahoo/QQQ/QQQ_1d.parquet"),
    "DEFENSE": Path("data/yahoo/DEFENSE/DEFENSE_1d.parquet"),
    "HYG":     Path("data/yahoo/HYG/HYG_1d.parquet"),
    "EEM":     Path("data/yahoo/EEM/EEM_1d.parquet"),
}

WEIGHTS = {
    "GOLD": 0.10,
    "BTC": 0.10,
    "QQQ": 0.35,
    "DEFENSE": 0.15,
    "HYG": 0.10,
    "EEM": 0.20,
}

START = "2022-01-01"
DAYS = 252
N_PATHS = 50   # 50 para ver trayectorias sin saturar
SEED = 42

def load_series():
    series = {}
    for name, p in DATA.items():
        if not p.exists():
            raise FileNotFoundError(f"No existe el fichero esperado: {p}. Ejecuta antes run_mixed_portfolio.py para descargar.")
        c = Candles.from_any(p, symbol=name).between(start=START)
        # normaliza fecha a tz-naive (por si BTC trae tz)
        df = c.frame.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        series[name] = Candles(symbol=name, frame=df)
    return series

def main():
    series = load_series()
    pf = Portfolio(series=series, weights=WEIGHTS, initial_value=1.0)

    # 1) Cópula Gaussiana
    cop_paths = pf.simulate(days=DAYS, n_paths=N_PATHS, seed=SEED, mc_method="copula")
    print("✅ Copula OK:", cop_paths.shape)

    # 2) (Opcional) también GBM / Cholesky / Bootstrap para comparar rápido
    gbm_paths  = pf.simulate(days=DAYS, n_paths=N_PATHS, seed=SEED, mc_method="gbm")
    chol_paths = pf.simulate(days=DAYS, n_paths=N_PATHS, seed=SEED, mc_method="cholesky")
    boot_paths = pf.simulate(days=DAYS, n_paths=N_PATHS, seed=SEED, mc_method="bootstrap")

    # 3) Gráfico con bandas 5–95% + algunas trayectorias por método
    def band(ax, paths, label):
        mean = paths.mean(axis=0)
        p5   = np.percentile(paths, 5, axis=0)
        p95  = np.percentile(paths, 95, axis=0)
        t = np.arange(paths.shape[1])
        ax.fill_between(t, p5, p95, alpha=0.12, label=f"{label} (banda 5–95%)")
        ax.plot(t, mean, lw=1.8, label=f"{label} (media)")
        # 8 trayectorias para no saturar
        for i in range(min(8, paths.shape[0])):
            ax.plot(t, paths[i], alpha=0.25, linewidth=0.8)

    fig, ax = plt.subplots(figsize=(11,6))
    band(ax, gbm_paths,  "GBM")
    band(ax, chol_paths, "Cholesky")
    band(ax, boot_paths, "Bootstrap")
    band(ax, cop_paths,  "Copula")

    ax.set_title("Monte Carlo — Comparativa métodos (50 paths por método)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    out_png = OUTDIR / "mc_compare_50paths.png"
    plt.savefig(out_png, dpi=140)
    print(f"📈 Guardado: {out_png}")

    # 4) Resumen de cada método (end value mean/p5/p95)
    def summarize(paths):
        end = paths[:, -1]
        return {
            "end_mean": float(end.mean()),
            "end_p5": float(np.percentile(end, 5)),
            "end_p95": float(np.percentile(end, 95)),
        }

    summary = {
        "GBM": summarize(gbm_paths),
        "Cholesky": summarize(chol_paths),
        "Bootstrap": summarize(boot_paths),
        "Copula": summarize(cop_paths),
    }

    # Guarda resumen y también imprime
    import json
    out_json = OUTDIR / "mc_compare_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("📝 Resumen final (valor al día 252):")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
