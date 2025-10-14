from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finlab.models.candles import Candles


@dataclass
class Portfolio:
    series: Dict[str, Candles]
    weights: Dict[str, float]
    initial_value: float = 1.0

    _returns: Optional[pd.Series] = field(init=False, default=None)

    # ----------------------- construcción -----------------------
    def __post_init__(self):
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Los pesos deben ser > 0")
        self.weights = {k: v / total for k, v in self.weights.items()}

    # ----------------------- limpieza/alineación ----------------
    def _aligned_returns(self) -> pd.DataFrame:
        """
        Limpia cada serie, la valida, remuestrea a días laborables,
        calcula retornos log y alinea por intersección de fechas.
        """
        frames = []
        for sym, c in self.series.items():
            cs = c.validate(strict=True).clean(fill_method="ffill").to_business_days(fill=True)
            r = cs.log_returns().rename(sym)
            if r.empty:
                raise ValueError(f"Serie vacía tras limpieza: {sym}")
            frames.append(r)
        df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        if df.shape[0] < 2:
            raise ValueError("No hay suficiente histórico tras limpiar y alinear.")
        return df

    def portfolio_returns(self) -> pd.Series:
        """Retornos log ponderados de la cartera (rebalanceo diario)."""
        df = self._aligned_returns()
        w = np.array([self.weights[s] for s in df.columns], dtype=float)
        port = df.dot(w)
        self._returns = port
        return port

    def stats(self, freq: int = 252, rf: float = 0.0) -> dict:
        """Media, desviación y Sharpe anualizado (con limpieza previa)."""
        r = self.portfolio_returns().dropna()
        if r.empty:
            return {"mean": np.nan, "std": np.nan, "sharpe": np.nan}
        mean = r.mean() * freq
        std = r.std() * np.sqrt(freq)
        sharpe = (mean - rf) / std if std and not np.isnan(std) else np.nan
        return {"mean": mean, "std": std, "sharpe": sharpe}

    # ----------------------- utilidades de correlación/retornos ----------------
    def _asset_returns_matrix(self) -> pd.DataFrame:
        """Matriz de retornos log limpios y alineados por activo (cols = símbolos)."""
        frames, syms = [], []
        for sym, c in self.series.items():
            rr = c.validate(strict=True).clean(fill_method="ffill").to_business_days(fill=True).log_returns().rename(sym)
            frames.append(rr); syms.append(sym)
        df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        if df.shape[0] < 2:
            raise ValueError("No hay suficiente histórico para análisis de componentes.")
        return df

    def assets_corr_matrix(self) -> pd.DataFrame:
        """Matriz de correlación entre retornos de los activos."""
        return self._asset_returns_matrix().corr()

    def max_correlation_warning(self, threshold: float = 0.5) -> str | None:
        """Devuelve un aviso si la correlación máxima (en valor absoluto) supera el umbral."""
        corr = self.assets_corr_matrix().values
        # ignorar diagonal
        mask = ~np.eye(corr.shape[0], dtype=bool)
        max_abs = float(np.abs(corr[mask]).max()) if corr.size > 1 else 0.0
        if max_abs > threshold:
            return f"⚠️ Correlación máxima |ρ|={max_abs:.2f} > {threshold:.2f}. Revisa la diversificación."
        return None

    # ----------------------- métricas de riesgo: VaR y CVaR --------------------
    @staticmethod
    def _var_cvar_from_returns(r: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
        """
        VaR/CVaR diarios a nivel 'alpha' sobre pérdidas (loss = -r).
        Devuelve números positivos (porcentaje en términos de 'loss').
        """
        r = pd.Series(r).dropna()
        if r.empty:
            return np.nan, np.nan
        loss = -r.values
        var = float(np.quantile(loss, alpha))
        cvar = float(loss[loss >= var].mean()) if np.any(loss >= var) else var
        return var, cvar

    # ----------------------- Monte Carlo (cartera) ---------------
    def simulate(
        self,
        days: int = 252,
        n_paths: int = 1000,
        seed: Optional[int] = None,
        freq: int = 252,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
        mu_override: Optional[float] = None,
        sigma_override: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simula la evolución del VALOR de la cartera como un GBM único.
        Parámetros maleables:
        - days, n_paths, seed
        - mu_scale / sigma_scale: escalado de drift y vol estimados
        - mu_override / sigma_override: si se pasan, sustituyen a la estimación histórica
        """
        r = self.portfolio_returns().dropna()
        dt = 1 / freq

        mu_hat = r.mean()          # retorno log medio por paso
        sig_hat = r.std()          # vol por paso
        mu = mu_override if mu_override is not None else mu_hat * mu_scale
        sigma = sigma_override if sigma_override is not None else sig_hat * sigma_scale

        rng = np.random.default_rng(seed)
        paths = np.zeros((n_paths, days))
        paths[:, 0] = self.initial_value
        for t in range(1, days):
            z = rng.standard_normal(n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        return paths

    def simulate_components(
        self,
        days: int = 252,
        n_paths: int = 1000,
        seed: Optional[int] = None,
        freq: int = 252,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Simula cada activo por separado (GBM univariado) y compone una cartera con
        rebalanceo diario (pesos fijos). Devuelve (comps, port_paths).
        """
        # construir matriz de retornos limpios por activo
        frames = []
        syms = []
        for sym, c in self.series.items():
            cs = c.validate(strict=True).clean(fill_method="ffill").to_business_days(fill=True)
            r = cs.log_returns().rename(sym)
            frames.append(r); syms.append(sym)
        df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        if df.shape[0] < 2:
            raise ValueError("No hay suficiente histórico para simular componentes.")

        dt = 1 / freq
        mu_vec = df.mean(axis=0).values * mu_scale
        sig_vec = df.std(axis=0).values * sigma_scale

        rng = np.random.default_rng(seed)
        comps: Dict[str, np.ndarray] = {}
        for i, sym in enumerate(syms):
            paths = np.zeros((n_paths, days))
            paths[:, 0] = 1.0
            for t in range(1, days):
                z = rng.standard_normal(n_paths)
                paths[:, t] = paths[:, t-1] * np.exp((mu_vec[i] - 0.5 * sig_vec[i]**2)*dt + sig_vec[i]*np.sqrt(dt)*z)
            comps[sym] = paths

        # combinar con rebalanceo diario
        port = np.zeros((n_paths, days))
        port[:, 0] = self.initial_value
        w = np.array([self.weights[s] for s in syms], dtype=float)

        for t in range(1, days):
            # rentabilidades diarias de cada activo
            d_rets = []
            for sym in syms:
                s = comps[sym]
                d_rets.append(s[:, t] / s[:, t-1] - 1.0)
            d_rets = np.stack(d_rets, axis=1)  # (n_paths, n_assets)
            port[:, t] = port[:, t-1] * (1.0 + (d_rets * w).sum(axis=1))

        return comps, port
    
    def random_portfolios(
        self,
        n: int = 1000,
        alpha_dirichlet: float = 1.0,
        alpha_var: float = 0.95,
        freq: int = 252,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Genera 'n' carteras con pesos aleatorios ~ Dirichlet(alpha_dirichlet),
        calcula métricas históricas (media, vol, Sharpe) y riesgo (VaR/CVaR diarios).
        Devuelve un DataFrame con una columna por métrica y una columna 'weights'
        que contiene un dict {simbolo: peso}.
        """
        df_ret = self._asset_returns_matrix()  # (T x Nassets)
        syms = list(df_ret.columns)
        rng = np.random.default_rng(seed)

        rows = []
        for _ in range(n):
            # pesos aleatorios
            w = rng.dirichlet([alpha_dirichlet] * len(syms))
            wmap = {s: float(wi) for s, wi in zip(syms, w)}
            # retornos de cartera (rebalanceo diario)
            rp = df_ret.dot(w)

            # métricas (históricas)
            mu = float(rp.mean()) * freq
            sd = float(rp.std()) * (freq ** 0.5)
            sharpe = (mu / sd) if sd > 0 else np.nan

            # riesgo (diario)
            var, cvar = self._var_cvar_from_returns(rp, alpha=alpha_var)

            rows.append({
                "mean_ann": mu,
                "std_ann": sd,
                "sharpe": sharpe,
                "VaR": var,
                "CVaR": cvar,
                "weights": wmap,
            })

        return pd.DataFrame(rows)


    # ----------------------- Resúmenes/plots ---------------------
    @staticmethod
    def summarize_paths(paths: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> dict:
        """Calcula bandas percentiles y media a lo largo del tiempo."""
        mean = paths.mean(axis=0)
        p_low = np.percentile(paths, q_low, axis=0)
        p_high = np.percentile(paths, q_high, axis=0)
        end_vals = paths[:, -1]
        return {
            "mean": mean,
            "p_low": p_low,
            "p_high": p_high,
            "end_mean": float(end_vals.mean()),
            "end_p5": float(np.percentile(end_vals, 5)),
            "end_p95": float(np.percentile(end_vals, 95)),
        }

    def plot_simulation(self, paths: np.ndarray, title: str = "Simulación Monte Carlo — Cartera"):
        """Visualiza las trayectorias con banda 5–95% y la media."""
        summary = self.summarize_paths(paths)
        t = np.arange(paths.shape[1])

        plt.figure(figsize=(10, 5))
        # bandeado
        plt.fill_between(t, summary["p_low"], summary["p_high"], alpha=0.2, label="Banda 5–95%")
        # media
        plt.plot(t, summary["mean"], lw=2, label="Media")
        # algunas trayectorias
        for i in range(min(20, paths.shape[0])):
            plt.plot(t, paths[i], alpha=0.25, linewidth=0.7)

        plt.title(title + f"\nFin esperado={summary['end_mean']:.4f} | p5={summary['end_p5']:.4f} | p95={summary['end_p95']:.4f}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def report(
        self,
        freq: int = 252,
        rf: float = 0.0,
        include_warnings: bool = True,
        include_components: bool = True,
        mc_days: int | None = 252,
        mc_paths: int = 2000,
        seed: int | None = 123,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> str:
        """
        Informe Markdown:
        - Pesos, métricas (anualizadas), warnings (opcional)
        - Resumen Monte Carlo (opcional, si mc_days no es None)
        - Resumen componentes (opcional)
        """
        # métricas históricas
        try:
            st = self.stats(freq=freq, rf=rf)
            r = self.portfolio_returns().dropna()
            nobs = len(r)
        except Exception as e:
            return f"# Portfolio Report\n\n**Error preparando métricas:** `{e}`"

        def _max_drawdown(x: pd.Series) -> float:
            eq = (x.cumsum()).pipe(np.exp)  # log-returns -> equity
            peak = np.maximum.accumulate(eq.values)
            dd = (eq.values - peak) / peak
            return float(dd.min())

        mdd = _max_drawdown(r)

        lines = [
            "# Portfolio Report",
            "",
            "## Pesos",
            *[f"- **{k}**: {v:.2%}" for k, v in self.weights.items()],
            "",
            "## Métricas históricas (anualizadas)",
            f"- Observaciones: **{nobs}**",
            f"- Rentabilidad media: **{st['mean']*100:.2f}%**",
            f"- Volatilidad: **{st['std']*100:.2f}%**",
            f"- Sharpe: **{st['sharpe'] if not np.isnan(st['sharpe']) else 'N/A'}**",
            f"- Máx. drawdown (desde retornos log): **{mdd*100:.2f}%**",
        ]

        # Warnings
        if include_warnings:
            warns: list[str] = []
            if nobs < 100:
                warns.append(f"- ⚠️ Histórico corto: sólo **{nobs}** días.")
            if any(v <= 0 for v in self.weights.values()):
                warns.append("- ⚠️ Pesos no positivos detectados.")
            srs = []
            for sym, c in self.series.items():
                df = c.frame
                if df["close"].isna().mean() > 0.05:
                    srs.append(sym)
            if srs:
                warns.append(f"- ⚠️ Series con >5% NaN en 'close': {', '.join(srs)}")
            if warns:
                lines += ["", "## Advertencias", *warns]

        # Componentes (resumen)
        if include_components:
            lines += ["", "## Componentes"]
            for sym, c in self.series.items():
                try:
                    rr = c.clean(fill_method="ffill").to_business_days().log_returns()
                    mu_i = rr.mean() * freq
                    sd_i = rr.std() * np.sqrt(freq)
                    lines.append(f"- **{sym}**: media {mu_i*100:.2f}%, vol {sd_i*100:.2f}%")
                except Exception:
                    lines.append(f"- **{sym}**: (no disponible tras limpieza)")

        # Monte Carlo (resumen numérico)
        if mc_days is not None and mc_days > 1:
            try:
                paths = self.simulate(
                    days=mc_days, n_paths=mc_paths, seed=seed,
                    mu_scale=mu_scale, sigma_scale=sigma_scale, freq=freq
                )
                sm = self.summarize_paths(paths)
                lines += [
                    "",
                    "## Monte Carlo (GBM cartera)",
                    f"- Días: **{mc_days}**, trayectorias: **{mc_paths}**",
                    f"- Valor final esperado: **{sm['end_mean']:.4f}**",
                    f"- Banda 5–95% final: **[{sm['end_p5']:.4f}, {sm['end_p95']:.4f}]**",
                ]
            except Exception as e:
                lines += ["", f"## Monte Carlo", f"- Error al simular: `{e}`"]

        return "\n".join(lines)
    def plots_report(
        self,
        normalize: bool = True,
        show_components: bool = True,
        show_hist: bool = True,
        show_corr: bool = True,
        show_mc: bool = True,
        mc_days: int = 252,
        mc_paths: int = 1000,
        seed: int | None = 123,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
        save_dir: str | None = None,
        dpi: int = 130,
    ):
        """
        Genera un informe visual con:
          1) Cierres normalizados de componentes
          2) Histograma de retornos de la cartera
          3) Matriz de correlaciones entre activos
          4) Banda Monte Carlo (GBM cartera)

        Si save_dir se indica, guarda PNGs allí.
        """
        import os
        import matplotlib.pyplot as plt

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1) Cierres normalizados
        if show_components:
            plt.close('all')
            plt.figure(figsize=(11, 5), dpi=dpi)
            for sym, c in self.series.items():
                df = c.clean(fill_method="ffill").to_business_days().to_dataframe()
                if df.empty:
                    continue
                y = df["close"].astype(float)
                if normalize and len(y) > 0:
                    y = y / y.iloc[0]
                plt.plot(df["date"], y, label=sym)
            plt.title("Componentes — cierres normalizados" if normalize else "Componentes — cierres")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "components.png"), bbox_inches="tight")

        # 2) Histograma de retornos de cartera
        if show_hist:
            plt.figure(figsize=(8, 4), dpi=dpi)
            r = self.portfolio_returns().dropna()
            plt.hist(r.values, bins=40, alpha=0.8)
            plt.title("Histograma de retornos (log, diarios)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "hist_returns.png"), bbox_inches="tight")

               # 3) Correlaciones entre activos (heatmap con anotaciones)
        if show_corr:
            df_ret = self._asset_returns_matrix()
            syms = list(df_ret.columns)
            corr = df_ret.corr()

            plt.figure(figsize=(6.5, 5.5), dpi=dpi)
            im = plt.imshow(corr.values, interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(syms)), syms, rotation=45, ha='right')
            plt.yticks(range(len(syms)), syms)
            plt.title("Matriz de correlaciones")

            # anotaciones con el coeficiente
            for i in range(len(syms)):
                for j in range(len(syms)):
                    plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=9)

            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "correlations.png"), bbox_inches="tight")

            # Aviso de correlación alta (no rompe, sólo informa)
            warn = self.max_correlation_warning(threshold=0.5)
            if warn:
                print(warn)


        # 4) Monte Carlo banda 5–95%
        if show_mc:
            paths = self.simulate(
                days=mc_days, n_paths=mc_paths, seed=seed,
                mu_scale=mu_scale, sigma_scale=sigma_scale
            )
            plt.figure(figsize=(10, 5), dpi=dpi)
            sm = self.summarize_paths(paths)
            t = np.arange(paths.shape[1])
            plt.fill_between(t, sm["p_low"], sm["p_high"], alpha=0.2, label="Banda 5–95%")
            plt.plot(t, sm["mean"], lw=2, label="Media")
            for i in range(min(15, paths.shape[0])):
                plt.plot(t, paths[i], alpha=0.25, linewidth=0.7)
            plt.title(f"Monte Carlo — fin esp. {sm['end_mean']:.4f} | p5 {sm['end_p5']:.4f} | p95 {sm['end_p95']:.4f}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "montecarlo.png"), bbox_inches="tight")


    def random_portfolios_summary(
        self,
        n: int = 1000,
        alpha_dirichlet: float = 1.0,
        alpha_var: float = 0.95,
        freq: int = 252,
        seed: int | None = 123,
        topk: int = 3,
    ) -> dict:
        """
        Ejecuta random_portfolios() y muestra:
         - Top/bottom por CVaR (menor es mejor)
         - Top/bottom por Sharpe (mayor es mejor)
        Devuelve un dict con los DataFrames de selección.
        """
        res = self.random_portfolios(n=n, alpha_dirichlet=alpha_dirichlet, alpha_var=alpha_var, freq=freq, seed=seed)

        # ordenar por CVaR (ascendente: menor pérdida esperada en cola = mejor)
        best_cvar = res.nsmallest(topk, "CVaR").copy()
        worst_cvar = res.nlargest(topk, "CVaR").copy()

        # ordenar por Sharpe (descendente)
        best_sharpe = res.nlargest(topk, "sharpe").copy()
        worst_sharpe = res.nsmallest(topk, "sharpe").copy()

        def _fmt_row(row):
            ws = ", ".join(f"{k}:{v:.2%}" for k, v in row["weights"].items())
            return f"Sharpe={row['sharpe']:.3f} | VaR={row['VaR']:.4f} | CVaR={row['CVaR']:.4f} | Weights: {ws}"

        print("\n🏆 Top por CVaR (mejores, menor riesgo de cola):")
        for _, r in best_cvar.iterrows():
            print("  -", _fmt_row(r))

        print("\n🚨 Peores por CVaR (mayor riesgo de cola):")
        for _, r in worst_cvar.iterrows():
            print("  -", _fmt_row(r))

        print("\n⭐ Top por Sharpe (mejores):")
        for _, r in best_sharpe.iterrows():
            print("  -", _fmt_row(r))

        print("\n⬇️  Peores por Sharpe:")
        for _, r in worst_sharpe.iterrows():
            print("  -", _fmt_row(r))

        return {
            "all": res,
            "best_cvar": best_cvar,
            "worst_cvar": worst_cvar,
            "best_sharpe": best_sharpe,
            "worst_sharpe": worst_sharpe,
        }
