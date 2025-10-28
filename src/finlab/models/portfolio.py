from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from finlab.models.candles import Candles

# ====================== HELPERS DE MÓDULO (NO dentro de la clase) ======================

def _exp_weights(T: int, halflife_days: float | None = None, lam: float | None = None) -> np.ndarray:
    """
    Pesos exponenciales normalizados w_t ∝ exp(-λ * age_t), donde age=0 es el dato más reciente.
    - Si halflife_days está dado, λ = ln(2) / halflife_days.
    - Si lam está dado, se usa lam directamente.
    - Si ambos son None, pesos iguales.
    Devuelve vector de tamaño T para series ordenadas ascendente por fecha (pasado -> presente).
    """
    if T <= 0:
        return np.array([])
    if halflife_days is None and lam is None:
        return np.ones(T) / T
    if lam is None:
        lam = np.log(2.0) / float(halflife_days)
    ages = np.arange(T - 1, -1, -1)  # [T-1, ..., 1, 0]  (0 = más reciente)
    w = np.exp(-lam * ages)
    s = w.sum()
    return w / s if s > 0 else np.ones(T) / T


def _weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Media y desviación típica ponderadas (ddof=0)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m = float((w * x).sum())
    var = float((w * (x - m) ** 2).sum())
    return m, np.sqrt(var)


def _weighted_cov(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """
    Covarianza ponderada (filas = tiempo ascendente, columnas = activos).
    C = (Xc^T diag(w) Xc) con sum(w)=1 y Xc centrada por medias ponderadas.
    """
    X = df.values.astype(float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m = (w[:, None] * X).sum(axis=0, keepdims=True)
    Xc = X - m
    return (Xc.T * w) @ Xc


def _cov_to_corr(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(C), 1e-18, None))
    Dinv = np.diag(1.0 / d)
    return Dinv @ C @ Dinv


# =================================== CLASE PORTFOLIO ===================================

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
        """Limpia, valida, remuestrea a B-days, calcula log-returns y alinea por intersección."""
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
        """Retornos log ponderados (rebalanceo diario)."""
        df = self._aligned_returns()
        w = np.array([self.weights[s] for s in df.columns], dtype=float)
        port = df.dot(w)
        self._returns = port
        return port

    def stats(self, freq: int = 252, rf: float = 0.04) -> dict:
        """Media, desviación y Sharpe anualizado (Sharpe con exceso vs. rf)."""
        r = self.portfolio_returns().dropna()
        if r.empty:
            return {"mean": np.nan, "std": np.nan, "sharpe": np.nan}
        mean = r.mean() * freq
        std = r.std() * np.sqrt(freq)
        sharpe = (mean - rf) / std if std and not np.isnan(std) else np.nan
        return {"mean": mean, "std": std, "sharpe": sharpe}

    # ----------------------- correlación / retornos -------------
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
        return self._asset_returns_matrix().corr()

    def max_correlation_warning(self, threshold: float = 0.5) -> Optional[str]:
        """Aviso si la correlación máxima (|ρ|) supera el umbral."""
        corr = self.assets_corr_matrix().values
        mask = ~np.eye(corr.shape[0], dtype=bool)  # ignora diagonal
        max_abs = float(np.abs(corr[mask]).max()) if corr.size > 1 else 0.0
        if max_abs > threshold:
            return f"⚠️ Correlación máxima |ρ|={max_abs:.2f} > {threshold:.2f}. Revisa la diversificación."
        return None

    # ----------------------- VaR / CVaR -------------------------
    @staticmethod
    def _var_cvar_from_returns(r: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
        """VaR/CVaR diarios a nivel 'alpha' sobre pérdidas (loss = -r). Devuelve valores positivos."""
        r = pd.Series(r).dropna()
        if r.empty:
            return np.nan, np.nan
        loss = -r.values
        var = float(np.quantile(loss, alpha))
        cvar = float(loss[loss >= var].mean()) if np.any(loss >= var) else var
        return var, cvar

    # ----------------------- Monte Carlo (multi-método) ---------
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
        *,
        mc_method: str = "gbm",            # "gbm" | "cholesky" | "copula" | "bootstrap"
        block_len: int = 10,               # bootstrap por bloques
        halflife_days: float | None = None,
        lam: float | None = None,          # alternativa directa a halflife: λ
    ) -> np.ndarray:
        """
        Simulación Monte Carlo del valor de la cartera con opción de ponderación exponencial de la historia.
        - gbm: GBM univariado con stats (μ,σ) de la cartera (media/vol log diarias), ponderadas (EWMA).
        - cholesky: componentes correlacionados (Cholesky) con μ y covarianza ponderadas.
        - copula: Iman–Conover con correlación objetivo ponderada y márgenes empíricas (bootstrap ponderado).
        - bootstrap: (block) bootstrap conjunto ponderado (preserva co-movimientos).
        Ponderación exponencial: w_t ∝ exp(-λ * age_t), age_t=0 dato más reciente. λ=ln(2)/halflife_days.
        """
        mc_method = mc_method.lower()

        if mc_method == "gbm":
            # Retornos log de la cartera
            r = self.portfolio_returns().dropna().values  # ordenados por fecha ascendente
            if r.size < 2:
                raise ValueError("Histórico insuficiente para GBM.")
            w = _exp_weights(len(r), halflife_days=halflife_days, lam=lam)
            mu_hat, sig_hat = _weighted_mean_std(r, w)
            mu = mu_override if mu_override is not None else mu_hat * mu_scale
            sigma = sigma_override if sigma_override is not None else sig_hat * sigma_scale
            dt = 1.0 / float(freq)

            rng = np.random.default_rng(seed)
            paths = np.zeros((n_paths, days))
            paths[:, 0] = self.initial_value
            for t in range(1, days):
                z = rng.standard_normal(n_paths)
                paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            return paths

        elif mc_method == "cholesky":
            # Construye matriz (T x N) de retornos log limpios y alineados
            df = self._asset_returns_matrix()  # indexado por fecha asc, columnas = símbolos
            if df.shape[0] < 2:
                raise ValueError("Histórico insuficiente para Cholesky.")

            # μ y Σ ponderados (EWMA)
            w = _exp_weights(len(df), halflife_days=halflife_days, lam=lam)
            mu_vec = (w[:, None] * df.values).sum(axis=0) * mu_scale
            Sigma = _weighted_cov(df, w) * (sigma_scale ** 2)

            # Cholesky con jitter si fuera necesario
            L = None
            jitter = 0.0
            for _ in range(6):
                try:
                    L = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0]) * jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-12 if jitter == 0.0 else jitter * 10
            if L is None:
                # fallback: diagonal (independiente) si no es PD
                L = np.diag(np.sqrt(np.clip(np.diag(Sigma), 1e-18, None)))

            rng = np.random.default_rng(seed)
            syms = list(df.columns)
            n_assets = len(syms)
            wport = np.array([self.weights[s] for s in syms], dtype=float)

            # simula incrementos log conjuntos con μ,Σ y compón el portfolio con rebalanceo
            paths = np.zeros((n_paths, days))
            paths[:, 0] = self.initial_value
            for p in range(n_paths):
                v = self.initial_value
                for t in range(1, days):
                    z = rng.standard_normal(n_assets)
                    inc = mu_vec + L @ z  # ~ N(μ,Σ)
                    r_s = np.exp(inc) - 1.0
                    v = v * (1.0 + (r_s * wport).sum())
                    paths[p, t] = v
            return paths

        elif mc_method == "copula":
            return self._simulate_copula(
                days=days, n_paths=n_paths, seed=seed,
                halflife_days=halflife_days, lam=lam
            )

        elif mc_method == "bootstrap":
            return self._simulate_bootstrap(
                days=days, n_paths=n_paths, seed=seed,
                block_len=block_len, halflife_days=halflife_days, lam=lam
            )

        else:
            raise ValueError("mc_method debe ser uno de: gbm | cholesky | copula | bootstrap")

    # ----------------------- Métodos auxiliares de simulación -------------------
    def _simulate_bootstrap(
        self,
        days: int,
        n_paths: int,
        seed: Optional[int],
        block_len: int = 10,
        halflife_days: float | None = None,
        lam: float | None = None,
    ) -> np.ndarray:
        """
        (Block) bootstrap conjunto ponderado por recencia:
        remuestrea filas de log-returns alineados con prob. ∝ recencia; block_len>1 preserva dependencia temporal.
        """
        df = self._asset_returns_matrix()  # (T x N)
        T, N = df.shape
        if T < 10:
            raise ValueError("Histórico insuficiente para bootstrap.")

        rng = np.random.default_rng(seed)
        syms = list(df.columns)
        wport = np.array([self.weights[s] for s in syms], dtype=float)

        # probabilidades por recencia (filas)
        w = _exp_weights(T, halflife_days=halflife_days, lam=lam)
        L = max(1, int(block_len))

        paths = np.zeros((n_paths, days))
        paths[:, 0] = self.initial_value

        if L == 1:
            # i.i.d. bootstrap ponderado
            for p in range(n_paths):
                v = self.initial_value
                for t in range(1, days):
                    k = rng.choice(T, p=w)
                    r_log = df.iloc[k].values
                    r_s = np.exp(r_log) - 1.0
                    v = v * (1.0 + (r_s * wport).sum())
                    paths[p, t] = v
            return paths

        # block bootstrap: elige inicios ponderados por la prob. del primer elemento del bloque
        start_probs = w[: T - L + 1].copy()
        start_probs = start_probs / start_probs.sum()

        for p in range(n_paths):
            v = self.initial_value
            t = 1
            while t < days:
                start = int(rng.choice(T - L + 1, p=start_probs))
                blk = df.iloc[start : start + L].values  # (L x N)
                for row in blk:
                    if t >= days:
                        break
                    r_s = np.exp(row) - 1.0
                    v = v * (1.0 + (r_s * wport).sum())
                    paths[p, t] = v
                    t += 1
        return paths

    def _simulate_copula(
        self,
        days: int,
        n_paths: int,
        seed: Optional[int],
        halflife_days: float | None = None,
        lam: float | None = None,
    ) -> np.ndarray:
        """
        Cópula gaussiana (Iman–Conover) con recencia:
          - Corr objetivo por covarianza ponderada (EWMA).
          - Márgenes empíricas vía bootstrap ponderado por recencia y reordenadas por ranks de Z correlacionado.
        """
        df = self._asset_returns_matrix()  # (T x N)
        T, N = df.shape
        if T < 50:
            raise ValueError("Histórico insuficiente para cópula (mín ~50 días).")

        rng = np.random.default_rng(seed)
        syms = list(df.columns)
        wport = np.array([self.weights[s] for s in syms], dtype=float)

        # correlación objetivo ponderada
        w = _exp_weights(T, halflife_days=halflife_days, lam=lam)
        Cw = _weighted_cov(df, w)
        Corr = _cov_to_corr(Cw)

        # Cholesky con jitter
        L = None
        jitter = 0.0
        for _ in range(6):
            try:
                L = np.linalg.cholesky(Corr + np.eye(N) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10
        if L is None:
            L = np.eye(N)

        # listas de log-returns por activo
        Xcols = [df[c].values for c in syms]
        p_rows = w  # prob. de bootstrap por fila (recencia)

        paths = np.zeros((n_paths, days))
        paths[:, 0] = self.initial_value

        for p in range(n_paths):
            # normales correlacionadas (days-1 x N)
            Z = rng.standard_normal(size=(days - 1, N)) @ L.T

            # simulación de log-returns preservando márgenes y ranks
            sim_log = np.zeros_like(Z)
            for i in range(N):
                samp_idx = rng.choice(T, size=days - 1, replace=True, p=p_rows)
                samp = np.sort(Xcols[i][samp_idx])  # ordena marginal
                order = np.argsort(Z[:, i])        # ranks target
                sim_log[order, i] = samp           # asigna por ranks (Iman–Conover)

            # convertir a retorno simple y componer
            sim_rets = np.exp(sim_log) - 1.0  # (days-1 x N)
            v = self.initial_value
            path = [v]
            for t in range(days - 1):
                r_day = (sim_rets[t] * wport).sum()
                v = v * (1.0 + r_day)
                path.append(v)
            paths[p, :] = np.array(path)

        return paths

    # ----------------------- Monte Carlo por componentes (antiguo) ------------
    def simulate_components(
        self,
        days: int = 252,
        n_paths: int = 1000,
        seed: Optional[int] = None,
        freq: int = 252,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
        correlated: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Simula cada activo y compone la cartera con rebalanceo diario (pesos fijos).
        Si correlated=True, usa Cholesky sobre la covarianza diaria de retornos log.
        Devuelve (comps, port_paths).
        """
        # (T x N) retornos diarios limpios y alineados
        frames, syms = [], []
        for sym, c in self.series.items():
            cs = c.validate(strict=True).clean(fill_method="ffill").to_business_days(fill=True)
            r = cs.log_returns().rename(sym)
            frames.append(r); syms.append(sym)
        df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        if df.shape[0] < 2:
            raise ValueError("No hay suficiente histórico para simular componentes.")

        # medias y covarianza diarias (por paso)
        mu_vec = (df.mean(axis=0).values) * mu_scale          # (N,)
        Sigma = (df.cov().values) * (sigma_scale ** 2)        # (N,N)

        # Cholesky (con jitter si hace falta) o modo independiente
        L = None
        if correlated and Sigma.size > 1:
            jitter = 0.0
            for _ in range(6):
                try:
                    L = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0]) * jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-12 if jitter == 0.0 else jitter * 10
            if L is None:
                correlated = False

        rng = np.random.default_rng(seed)

        # Simulación por activo
        n_assets = len(syms)
        comps: Dict[str, np.ndarray] = {sym: np.zeros((n_paths, days)) for sym in syms}
        for sym in syms:
            comps[sym][:, 0] = 1.0

        for t in range(1, days):
            if correlated and n_assets > 1:
                eps = rng.standard_normal((n_paths, n_assets))   # (P, N)
                shocks = eps @ L.T                               # (P, N) ~ N(0, Sigma)
                inc = mu_vec + shocks                            # incrementos log diarios
            else:
                inc = np.zeros((n_paths, n_assets))
                for i in range(n_assets):
                    z = rng.standard_normal(n_paths) * np.sqrt(Sigma[i, i])
                    inc[:, i] = mu_vec[i] + z

            for i, sym in enumerate(syms):
                comps[sym][:, t] = comps[sym][:, t-1] * np.exp(inc[:, i])

        # Combinar en cartera con rebalanceo diario (pesos fijos)
        port = np.zeros((n_paths, days))
        port[:, 0] = self.initial_value
        w = np.array([self.weights[s] for s in syms], dtype=float)

        for t in range(1, days):
            d_rets = []
            for sym in syms:
                s = comps[sym]
                d_rets.append(s[:, t] / s[:, t-1] - 1.0)
            d_rets = np.stack(d_rets, axis=1)                   # (P, N)
            port[:, t] = port[:, t-1] * (1.0 + (d_rets * w).sum(axis=1))

        return comps, port

    # ----------------------- Barrido de carteras aleatorias --------------------
    def random_portfolios(
        self,
        n: int = 1000,
        alpha_dirichlet: float = 1.0,
        alpha_var: float = 0.95,
        freq: int = 252,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Genera 'n' carteras con pesos aleatorios ~ Dirichlet(alpha_dirichlet),
        calcula métricas históricas (μ, σ, Sharpe) y riesgo (VaR/CVaR diarios).
        """
        df_ret = self._asset_returns_matrix()
        syms = list(df_ret.columns)
        rng = np.random.default_rng(seed)

        rows = []
        for _ in range(n):
            w = rng.dirichlet([alpha_dirichlet] * len(syms))
            wmap = {s: float(wi) for s, wi in zip(syms, w)}
            rp = df_ret.dot(w)

            mu = float(rp.mean()) * freq
            sd = float(rp.std()) * (freq ** 0.5)
            sharpe = (mu / sd) if sd > 0 else np.nan

            var, cvar = self._var_cvar_from_returns(rp, alpha=alpha_var)
            rows.append({"mean_ann": mu, "std_ann": sd, "sharpe": sharpe, "VaR": var, "CVaR": cvar, "weights": wmap})

        return pd.DataFrame(rows)

    # ----------------------- Resúmenes / plots ------------------
    @staticmethod
    def summarize_paths(paths: np.ndarray, q_low: float = 5.0, q_high: float = 95.0) -> dict:
        """Bandas percentiles y media a lo largo del tiempo + resumen final."""
        mean = paths.mean(axis=0)
        p_low = np.percentile(paths, q_low, axis=0)
        p_high = np.percentile(paths, q_high, axis=0)
        end_vals = paths[:, -1]
        return {
            "mean": mean, "p_low": p_low, "p_high": p_high,
            "end_mean": float(end_vals.mean()),
            "end_p5": float(np.percentile(end_vals, 5)),
            "end_p95": float(np.percentile(end_vals, 95)),
        }

    def plot_simulation(self, paths: np.ndarray, title: str = "Simulación Monte Carlo — Cartera"):
        """Visualiza trayectorias, media y banda 5–95%."""
        summary = self.summarize_paths(paths)
        t = np.arange(paths.shape[1])

        plt.figure(figsize=(10, 5))
        plt.fill_between(t, summary["p_low"], summary["p_high"], alpha=0.2, label="Banda 5–95%")
        plt.plot(t, summary["mean"], lw=2, label="Media")
        for i in range(min(20, paths.shape[0])):
            plt.plot(t, paths[i], alpha=0.25, linewidth=0.7)
        plt.title(title + f"\nFin esperado={summary['end_mean']:.4f} | p5={summary['end_p5']:.4f} | p95={summary['end_p95']:.4f}")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    def report(
        self,
        freq: int = 252,
        rf: float = 0.04,
        include_warnings: bool = True,
        include_components: bool = True,
        mc_days: Optional[int] = 252,
        mc_paths: int = 2000,
        seed: Optional[int] = 123,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
    ) -> str:
        """Informe Markdown: pesos, métricas anualizadas, avisos y resumen Monte Carlo."""
        try:
            st = self.stats(freq=freq, rf=rf)
            r = self.portfolio_returns().dropna()
            nobs = len(r)
            start_eff = r.index.min()
            end_eff = r.index.max()
                    # --- CAGR esperado (desde media log anualizada) y CAGR histórico (realizado) ---
            mean_ann = st.get("mean", np.nan)  # μ log anualizado
            cagr_expected = float(np.exp(mean_ann) - 1.0) if np.isfinite(mean_ann) else np.nan

            # Equity a partir de retornos log: eq_t = exp(sum_{τ≤t} r_τ)
            eq = np.exp(r.cumsum())
            # Años efectivos de la muestra
            years = (pd.to_datetime(end_eff) - pd.to_datetime(start_eff)).days / 365.25 if nobs > 1 else np.nan
            # Crecimiento total (multiplicador); eq.iloc[0] ≈ 1.0 por construcción
            growth = float(eq.iloc[-1] / eq.iloc[0]) if len(eq) >= 2 else np.nan
            cagr_real = float(growth ** (1.0 / years) - 1.0) if (np.isfinite(growth) and growth > 0 and np.isfinite(years) and years > 0) else np.nan

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
            "## Rango de datos efectivo",
            f"- Desde: **{pd.to_datetime(start_eff).date()}**",
            f"- Hasta: **{pd.to_datetime(end_eff).date()}**",
            "",
            "## Pesos",
            *[f"- **{k}**: {v:.2%}" for k, v in self.weights.items()],
            "",
            "## Métricas históricas (anualizadas)",
            f"- Observaciones: **{nobs}**",
            f"- Rentabilidad media: **{st['mean']*100:.2f}%**",
            f"- Volatilidad: **{st['std']*100:.2f}%**",
            f"- Sharpe (rf={rf*100:.2f}%): **{st['sharpe'] if not np.isnan(st['sharpe']) else 'N/A'}**",
            f"- Máx. drawdown (desde retornos log): **{mdd*100:.2f}%**",
             f"- CAGR esperado (desde μ log): **{cagr_expected*100:.2f}%**",
            f"- CAGR histórico (realizado): **{(cagr_real*100 if np.isfinite(cagr_real) else float('nan')):.2f}%**",

        ]

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
                lines += ["", "## Monte Carlo", f"- Error al simular: `{e}`"]

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
        seed: Optional[int] = 123,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
        save_dir: Optional[str] = None,
        dpi: int = 130,
        logy: bool = True,
        align: str = "inner",
    ):
        """
        Visual report:
          1) Componentes (cierres normalizados; intersección temporal si align='inner')
          2) Histograma de retornos de cartera
          3) Matriz de correlaciones (con anotaciones)
          4) Banda Monte Carlo (GBM cartera)
        """
        import os
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1) Componentes
        if show_components:
            # Cargar frames limpios y a B-days
            frames = {}
            for sym, c in self.series.items():
                df = c.clean(fill_method="ffill").to_business_days().to_dataframe()[["date", "close"]].copy()
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                frames[sym] = df.dropna(subset=["date", "close"])

            # Alinear por intersección de fechas (periodo común) si align="inner"
            common_idx = None
            if align.lower() == "inner":
                for df in frames.values():
                    idx = pd.DatetimeIndex(df["date"])
                    common_idx = idx if common_idx is None else common_idx.intersection(idx)
                # Recortar cada serie al tramo común
                if common_idx is not None and len(common_idx) > 0:
                    for sym in list(frames.keys()):
                        df = frames[sym]
                        frames[sym] = df[df["date"].isin(common_idx)]

            plt.close("all")
            plt.figure(figsize=(11, 5), dpi=dpi)
            for sym, df in frames.items():
                if df.empty:
                    continue
                y = df["close"].astype(float)
                if normalize and len(y) > 0:
                    y = y / y.iloc[0]  # normaliza desde el primer dato disponible de CADA serie (tras el recorte)
                plt.plot(df["date"], y, label=sym)

            ttl = "Componentes — cierres normalizados" if normalize else "Componentes — cierres"
            if align.lower() == "inner" and common_idx is not None and len(common_idx) > 0:
                ttl += f" (intersección {common_idx.min().date()} → {common_idx.max().date()})"
            plt.title(ttl)
            plt.grid(alpha=0.3)
            plt.legend()
            if logy:
                plt.yscale("log")
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "components.png"), bbox_inches="tight")

        # 2) Histograma de retornos de cartera
        if show_hist:
            plt.figure(figsize=(8, 4), dpi=dpi)
            r = self.portfolio_returns().dropna()
            plt.hist(r.values, bins=40, alpha=0.8)
            plt.title("Histograma de retornos (log, diarios)")
            plt.grid(alpha=0.3); plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "hist_returns.png"), bbox_inches="tight")

        # 3) Correlaciones (heatmap anotado) + warning si |ρ|max>0.5
        if show_corr:
            df_ret = self._asset_returns_matrix()
            syms = list(df_ret.columns)
            corr = df_ret.corr()
            plt.figure(figsize=(6.5, 5.5), dpi=dpi)
            im = plt.imshow(corr.values, interpolation="nearest", aspect="auto", vmin=-1, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(syms)), syms, rotation=45, ha="right")
            plt.yticks(range(len(syms)), syms)
            plt.title("Matriz de correlaciones")
            for i in range(len(syms)):
                for j in range(len(syms)):
                    plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "correlations.png"), bbox_inches="tight")
            warn = self.max_correlation_warning(threshold=0.5)
            if warn:
                print(warn)

        # 4) Monte Carlo (GBM cartera)
        if show_mc:
            paths = self.simulate(days=mc_days, n_paths=mc_paths, seed=seed, mu_scale=mu_scale, sigma_scale=sigma_scale)
            plt.figure(figsize=(10, 5), dpi=dpi)
            sm = self.summarize_paths(paths)
            t = np.arange(paths.shape[1])
            plt.fill_between(t, sm["p_low"], sm["p_high"], alpha=0.2, label="Banda 5–95%")
            plt.plot(t, sm["mean"], lw=2, label="Media")
            for i in range(min(15, paths.shape[0])):
                plt.plot(t, paths[i], alpha=0.25, linewidth=0.7)
            plt.title(f"Monte Carlo — fin esp. {sm['end_mean']:.4f} | p5 {sm['end_p5']:.4f} | p95 {sm['end_p95']:.4f}")
            plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "montecarlo.png"), bbox_inches="tight")

    # ----------------------- Resumen Dirichlet (top/bottom) ----
    def random_portfolios_summary(
        self,
        n: int = 1000,
        alpha_dirichlet: float = 1.0,
        alpha_var: float = 0.95,
        freq: int = 252,
        seed: Optional[int] = 123,
        topk: int = 3,
    ) -> dict:
        """Imprime top/bottom por CVaR y Sharpe, y devuelve dict con los DataFrames."""
        res = self.random_portfolios(n=n, alpha_dirichlet=alpha_dirichlet, alpha_var=alpha_var, freq=freq, seed=seed)
        best_cvar = res.nsmallest(topk, "CVaR").copy()
        worst_cvar = res.nlargest(topk, "CVaR").copy()
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
