# run_plot_mc.py
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Agregar la carpeta src/ al path ---
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

# === Carga datos (ajusta rutas si las tuyas son distintas) ===
btc = Candles.from_any(Path("data/twelvedata/BTC_USD/BTC_USD_1day.parquet")).clean().to_business_days()
eth = Candles.from_any(Path("data/twelvedata/ETH_USD/ETH_USD_1day.parquet")).clean().to_business_days()

pf = Portfolio(series={"BTC": btc, "ETH": eth}, weights={"BTC": 0.5, "ETH": 0.5})

# === Parámetros comunes de simulación ===
DAYS = 252
N_PATHS = 600
SEED = 123
HALFLIFE = 90
BLOCK_LEN = 10  # solo afecta al método bootstrap

methods = ["gbm", "cholesky", "copula", "bootstrap"]
outdir = Path("outputs/mc")
outdir.mkdir(parents=True, exist_ok=True)


def summarize_and_print(name, paths):
    sm = pf.summarize_paths(paths)
    print(f"{name:>10s} | fin esp={sm['end_mean']:.4f} | p5={sm['end_p5']:.4f} | p95={sm['end_p95']:.4f}")
    return sm


def plot_paths(name, paths):
    sm = pf.summarize_paths(paths)
    t = np.arange(paths.shape[1])

    plt.figure(figsize=(10, 5))
    plt.fill_between(t, sm["p_low"], sm["p_high"], alpha=0.2, label=f"{name.upper()} banda 5–95%")
    plt.plot(t, sm["mean"], lw=2, label=f"{name.upper()} media")
    for i in range(min(20, paths.shape[0])):
        plt.plot(t, paths[i], alpha=0.25, linewidth=0.7)

    plt.title(f"MC — {name.upper()} | fin esp {sm['end_mean']:.4f} | p5 {sm['end_p5']:.4f} | p95 {sm['end_p95']:.4f}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"mc_{name}_paths.png", bbox_inches="tight")


def plot_terminal_hist(name, paths):
    end_vals = paths[:, -1]
    plt.figure(figsize=(7, 4))
    plt.hist(end_vals, bins=40, alpha=0.85)
    plt.title(f"Distribución del valor final — {name.upper()}")
    plt.xlabel("Valor final")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"mc_{name}_terminal_hist.png", bbox_inches="tight")


def compare_methods_bands(method_list, days, n_paths, seed, halflife_days, block_len):
    """
    Dibuja en una sola figura:
      - Las bandas 5–95% de cada método (relleno semitransparente)
      - La media de cada método (línea)
    """
    plt.figure(figsize=(11.5, 6))
    t = np.arange(days)

    # Para la leyenda clara, guardamos handles y labels manualmente
    lines_for_legend = []
    labels_for_legend = []

    for m in method_list:
        paths = pf.simulate(
            days=days,
            n_paths=n_paths,
            seed=seed,
            mc_method=m,
            halflife_days=halflife_days,
            block_len=block_len,
        )
        sm = pf.summarize_paths(paths)

        # bandas
        plt.fill_between(t, sm["p_low"], sm["p_high"], alpha=0.12, label=f"{m.upper()} banda 5–95%")

        # media (guardamos el handle para leyenda bonita)
        (line_handle,) = plt.plot(t, sm["mean"], lw=2, label=f"{m.upper()} media")
        lines_for_legend.append(line_handle)
        labels_for_legend.append(f"{m.upper()} media")

    plt.title("Comparativa Monte Carlo — Bandas y medias por método")
    plt.grid(alpha=0.3)
    # Leyenda: primero todas las medias (líneas), luego dejamos que matplotlib añada rellenos
    leg1 = plt.legend(lines_for_legend, labels_for_legend, loc="upper left")
    plt.gca().add_artist(leg1)
    plt.legend(loc="upper right")  # para las bandas
    plt.tight_layout()
    plt.savefig(outdir / "mc_compare_bands.png", bbox_inches="tight")


def compare_methods_terminal_density(method_list, days, n_paths, seed, halflife_days, block_len):
    """
    Superpone histogramas normalizados (densidades) del valor final de los 4 métodos.
    """
    plt.figure(figsize=(8.8, 5.2))
    for m in method_list:
        paths = pf.simulate(
            days=days,
            n_paths=n_paths,
            seed=seed,
            mc_method=m,
            halflife_days=halflife_days,
            block_len=block_len,
        )
        end_vals = paths[:, -1]
        plt.hist(end_vals, bins=60, density=True, alpha=0.35, label=m)

    plt.title("Comparativa densidad del valor final — métodos")
    plt.xlabel("Valor final")
    plt.ylabel("Densidad")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mc_compare_terminal_all.png", bbox_inches="tight")


def main():
    # Report de la cartera (sin MC dentro del report para no duplicar)
    print(pf.report(mc_days=None))

    # Gráficos individuales y resúmenes por método
    for m in methods:
        paths = pf.simulate(
            days=DAYS,
            n_paths=N_PATHS,
            seed=SEED,
            mc_method=m,
            halflife_days=HALFLIFE,
            block_len=BLOCK_LEN,
        )
        summarize_and_print(m, paths)
        plot_paths(m, paths)
        plot_terminal_hist(m, paths)

    # Comparativas en una figura
    compare_methods_bands(methods, DAYS, N_PATHS, SEED, HALFLIFE, BLOCK_LEN)
    compare_methods_terminal_density(methods, DAYS, N_PATHS, SEED, HALFLIFE, BLOCK_LEN)

    print(f"\n✅ Gráficos guardados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()


