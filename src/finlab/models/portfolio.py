from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import logging
import warnings

from finlab.models.candles import Candles

# ====================== HELPERS DE MÓDULO ======================

def _exp_weights(T: int, halflife_days: float | None = None, lam: float | None = None) -> np.ndarray:
    if T <= 0:
        return np.array([])
    if halflife_days is None and lam is None:
        return np.ones(T) / T
    if lam is None:
        lam = np.log(2.0) / float(halflife_days)
    ages = np.arange(T - 1, -1, -1)
    w = np.exp(-lam * ages)
    s = w.sum()
    return w / s if s > 0 else np.ones(T) / T

def _weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m = float((w * x).sum())
    var = float((w * (x - m) ** 2).sum())
    return m, np.sqrt(var)

def _weighted_cov(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
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
    logger: Optional[logging.Logger] = field(default=None, repr=False)

    _returns: Optional[pd.Series] = field(init=False, default=None)
    _asset_returns_cache: Optional[pd.DataFrame] = field(init=False, default=None)

    def __post_init__(self):
        missing_symbols = set(self.weights.keys()) - set(self.series.keys())
        if missing_symbols:
            raise ValueError(f"Símbolos en weights no encontrados en series: {missing_symbols}")
        
        negative_weights = {k: v for k, v in self.weights.items() if v < 0}
        if negative_weights:
            raise ValueError(f"Pesos negativos no permitidos: {negative_weights}")
        
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Los pesos deben ser > 0")
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    def _log_warning(self, message: str):
        if self.logger:
            self.logger.warning(message)
        else:
            warnings.warn(message, UserWarning, stacklevel=3)

    # ----------------------- ALINEACIÓN INTELIGENTE CORREGIDA -----------------------
    def _aligned_returns(self) -> pd.DataFrame:
        """Alinea series inteligentemente - usa business days con relleno para crypto."""
        frames = []
        all_frequencies = {}
        
        for sym, c in self.series.items():
            # Detectar frecuencia de cada activo
            freq = c.detect_asset_frequency()
            all_frequencies[sym] = freq
            
            # Para todos los activos, usar business days con relleno apropiado
            cs = c.to_business_days(fill_method="none")  # días laborables, cero relleno

            if freq == '24_7':
                # Crypto: business days con relleno para mantener continuidad
                cs_processed = cs.to_business_days(fill_method="previous_close")
            else:
                # Acciones: business days sin relleno (ya tienen gaps naturales)
                cs_processed = cs.to_business_days(fill_method="none")
            
            r = cs_processed.log_returns().rename(sym)
            if r.empty:
                raise ValueError(f"Serie vacía tras limpieza: {sym}")
            frames.append(r)
        
        # Primero intentar unión por intersección
        df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        
        # Si no hay suficiente overlap, usar unión exterior con forward fill
        if df.shape[0] < 10:
            self._log_warning(f"Poco overlap histórico ({df.shape[0]} días). Usando unión exterior.")
            df = pd.concat(frames, axis=1, join="outer")
            df = df.ffill().dropna(how="any")
            
            if df.shape[0] < 10:
                # Último recurso: usar el activo con menos datos como referencia
                min_length = min(len(f) for f in frames)
                if min_length > 0:
                    df = pd.concat([f.iloc[-min_length:] for f in frames], axis=1, join="inner")
        
        if df.shape[0] < 2:
            raise ValueError(f"No hay suficiente histórico común. Días disponibles: {df.shape[0]}")
            
        return df

    def portfolio_returns(self) -> pd.Series:
        """Retornos log ponderados (rebalanceo diario)."""
        df = self._aligned_returns()
        w = np.array([self.weights[s] for s in df.columns], dtype=float)
        port = df.dot(w)
        self._returns = port
        return port

    def stats(self, freq: int = 252, rf: float = 0.04) -> dict:
        """Métricas anualizadas."""
        r = self.portfolio_returns().dropna()
        if r.empty:
            return {
                "mean": np.nan, "std": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "skewness": np.nan, "kurtosis": np.nan,
                "var_95": np.nan, "cvar_95": np.nan
            }
        
        mean = r.mean() * freq
        std = r.std() * np.sqrt(freq)
        sharpe = (mean - rf) / std if std > 0 else np.nan
        
        downside_returns = r[r < 0]
        downside_std = downside_returns.std() * np.sqrt(freq) if len(downside_returns) > 1 else 0
        sortino = (mean - rf) / downside_std if downside_std > 0 else np.nan
        
        skew = float(r.skew())
        kurt = float(r.kurtosis())
        
        var_95, cvar_95 = self._var_cvar_from_returns(r, 0.95)
        
        return {
            "mean": mean, "std": std, "sharpe": sharpe,
            "sortino": sortino, "skewness": skew, "kurtosis": kurt,
            "var_95": var_95, "cvar_95": cvar_95
        }
    
    def detect_frequency_mismatch(self) -> Dict[str, List[str]]:
        """
        Detecta si hay mezcla de activos con diferentes frecuencias de trading.
        """
        frequencies = {}
        warnings = []
        
        for sym, candle in self.series.items():
            freq = candle.detect_asset_frequency()
            frequencies[sym] = freq
            
            if freq == 'unknown':
                warnings.append(f"⚠️  No se pudo detectar frecuencia para {sym}")
        
        unique_freqs = set(frequencies.values())
        if len(unique_freqs) > 1 and 'unknown' not in unique_freqs:
            freq_groups = {}
            for sym, freq in frequencies.items():
                freq_groups.setdefault(freq, []).append(sym)
            
            warnings.append("🔔 MEZCLA DE FRECUENCIAS DETECTADA:")
            for freq, symbols in freq_groups.items():
                warnings.append(f"   - {freq}: {', '.join(symbols)}")
            warnings.append("   Se usarán días laborables con relleno para crypto")
        
        return {'frequencies': frequencies, 'warnings': warnings}

    def _asset_returns_matrix(self) -> pd.DataFrame:
        """Matriz de retornos log alineados."""
        if hasattr(self, '_asset_returns_cache') and self._asset_returns_cache is not None:
            return self._asset_returns_cache
            
        frames, syms = [], []
        for sym, c in self.series.items():
            # Usar el mismo método de alineación que _aligned_returns
            freq = c.detect_asset_frequency()
            cs = c.clean(fill_method="ffill")
            
            if freq == '24_7':
                cs_processed = cs.to_business_days(fill_method="previous_close")
            else:
                cs_processed = cs.to_business_days(fill_method="none")
                
            rr = cs_processed.log_returns().rename(sym)  


            frames.append(rr)
            syms.append(sym)
        
        df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
        
        # Fallback si hay poco overlap
        if df.shape[0] < 10:
            df = pd.concat(frames, axis=1, join="outer")
            df = df.ffill().dropna(how="any")
        
        if df.shape[0] < 2:
            raise ValueError("No hay suficiente histórico común para análisis.")
        
        self._asset_returns_cache = df
        return df

    def assets_corr_matrix(self) -> pd.DataFrame:
        return self._asset_returns_matrix().corr()

    def max_correlation_warning(self, threshold: float = 0.5) -> Optional[str]:
        corr = self.assets_corr_matrix().values
        mask = ~np.eye(corr.shape[0], dtype=bool)
        max_abs = float(np.abs(corr[mask]).max()) if corr.size > 1 else 0.0
        if max_abs > threshold:
            return f"⚠️ Correlación máxima |ρ|={max_abs:.2f} > {threshold:.2f}. Revisa la diversificación."
        return None

    @staticmethod
    def _var_cvar_from_returns(r: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
        r = pd.Series(r).dropna()
        if r.empty:
            return np.nan, np.nan
        loss = -r.values
        var = float(np.quantile(loss, alpha))
        cvar = float(loss[loss >= var].mean()) if np.any(loss >= var) else var
        return var, cvar

    # ----------------------- Métricas de Rolling -----------------
    def rolling_stats(self, window: int = 63, freq: int = 252) -> pd.DataFrame:
        returns = self.portfolio_returns()
        
        rolling_mean = returns.rolling(window).mean() * freq
        rolling_std = returns.rolling(window).std() * np.sqrt(freq)
        rolling_sharpe = rolling_mean / rolling_std
        
        return pd.DataFrame({
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_sharpe': rolling_sharpe
        }).dropna()

    # ----------------------- Frontera Eficiente -----------------
    def efficient_frontier(self, n_portfolios: int = 1000, freq: int = 252) -> pd.DataFrame:
        df_ret = self._asset_returns_matrix()
        returns = df_ret.values
        mean_returns = returns.mean(axis=0) * freq
        cov_matrix = np.cov(returns.T) * freq
        
        results = []
        n_assets = len(mean_returns)
        
        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else np.nan
            
            results.append({
                'weights': weights,
                'return': portfolio_return,
                'std': portfolio_std,
                'sharpe': sharpe
            })
        
        return pd.DataFrame(results)

    # ----------------------- Monte Carlo SIMPLIFICADO -----------------
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
        """
        MAX_PATHS = 10000
        MAX_DAYS = 1000

        if days <= 0:
            raise ValueError("days debe ser > 0")
        if n_paths <= 0:
            raise ValueError("n_paths debe ser > 0")
        if n_paths > MAX_PATHS:
            raise ValueError(f"n_paths no puede exceder {MAX_PATHS}")
        if days > MAX_DAYS:
            raise ValueError(f"days no puede exceder {MAX_DAYS}")

        mc_method = mc_method.lower()

        # Valida que tenemos al menos algo de histórico común
        df_test = self._asset_returns_matrix()
        if df_test.shape[0] < 2:
            raise ValueError("Histórico insuficiente para simular.")

        if mc_method == "gbm":
            # Retornos log de la cartera
            r = self.portfolio_returns().dropna().values
            if r.size < 2:
                raise ValueError("Histórico insuficiente para GBM.")
            # pesos exponenciales (opcional)
            w = _exp_weights(len(r), halflife_days=halflife_days, lam=lam)
            mu_hat, sig_hat = _weighted_mean_std(r, w)
            m = (mu_override if mu_override is not None else mu_hat) * mu_scale
            s = (sigma_override if sigma_override is not None else sig_hat) * sigma_scale
            

            rng = np.random.default_rng(seed)
            paths = np.zeros((n_paths, days))
            paths[:, 0] = self.initial_value
            for t in range(1, days):
                z = rng.standard_normal(n_paths)
                paths[:, t] = paths[:, t - 1] * np.exp(m + s*z)
            return paths

        elif mc_method == "cholesky":
            # Matriz (T x N) de retornos log limpios y alineados por intersección
            df = self._asset_returns_matrix()
            if df.shape[0] < 2:
                raise ValueError("Histórico insuficiente para Cholesky.")

            # μ y Σ ponderados (EWMA)
            w = _exp_weights(len(df), halflife_days=halflife_days, lam=lam)
            X = df.values.astype(float)
            mu_vec = (w[:, None] * df.values).sum(axis=0) * mu_scale
            Sigma = _weighted_cov(df, w) * (sigma_scale ** 2)

            # Cholesky con jitter si hace falta
            L = None
            jitter = 0.0
            for _ in range(6):
                try:
                    L = np.linalg.cholesky(Sigma + np.eye(Sigma.shape[0]) * jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter = 1e-12 if jitter == 0.0 else jitter * 10
            if L is None:
                # fallback: diagonal (independiente)
                L = np.diag(np.sqrt(np.clip(np.diag(Sigma), 1e-18, None)))

            rng = np.random.default_rng(seed)
            N = X.shape[1]
            wport = np.array([self.weights[s] for s in df.columns], dtype=float)

            Z = rng.standard_normal(size=(days-1,n_paths,N))

            shocks = Z @ L.T #aplicamos correlación

            inc = shocks + mu_vec[None,None, :] 
            
            r_assets = np.exp(inc) -1.0

            r_port = (r_assets * wport[None,None, : ]).sum(axis=2)

            paths = np.empty((n_paths,days),dtype = float)
            paths[:,0] = self.initial_value

            growth = np.cumprod(1.0 + r_port , axis = 0)
            paths[:, 1:] = self.initial_value * growth.T

            return paths

        elif mc_method == "bootstrap":
            return self._simulate_bootstrap(
                days=days, n_paths=n_paths, seed=seed,
                block_len=block_len, halflife_days=halflife_days, lam=lam
            )

        elif mc_method == "copula":
            return self._simulate_copula(
                days=days, n_paths=n_paths, seed=seed,
                halflife_days=halflife_days, lam=lam
            )

        else:
            raise ValueError("mc_method debe ser uno de: gbm | cholesky | copula | bootstrap")


    # ----------------------- Barrido de carteras aleatorias --------------------
    def random_portfolios(
        self,
        n: int = 1000,
        alpha_dirichlet: float = 1.0,
        alpha_var: float = 0.95,
        freq: int = 252,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
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
        summary = self.summarize_paths(paths)
        t = np.arange(paths.shape[1])

        plt.figure(figsize=(10, 5))
        plt.fill_between(t, summary["p_low"], summary["p_high"], alpha=0.2, label="Banda 5–95%")
        plt.plot(t, summary["mean"], lw=2, label="Media")
        for i in range(min(20, paths.shape[0])):
            plt.plot(t, paths[i], alpha=0.25, linewidth=0.7)
        plt.title(title + f"\nFin esperado={summary['end_mean']:.4f} | p5={summary['end_p5']:.4f} | p95={summary['end_p95']:.4f}")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    def monte_carlo_overview(
        self,
        days: int = 252,
        n_paths: int = 2000,
        seed: int = 42,
        freq: int = 252,
        rf: float = 0.0,
        show_paths_per_method: int = 50,
        alpha_band: tuple = (5, 95),
        title: str = "Simulaciones Monte Carlo — Cartera Global (bandas + trayectorias por método)",
        save_path: Optional[str] = None,
    ):
            """
            Ejecuta y compara múltiples métodos de Monte Carlo sobre la cartera:
            - GBM
            - Cholesky
            - Bootstrap
            - Copula

            Muestra bandas del 5-95% y trayectorias individuales por método.
            Devuelve un informe con métricas medias por método.
            """

            import numpy as np
            import matplotlib.pyplot as plt
            from textwrap import dedent

            methods = ["gbm", "cholesky", "bootstrap", "copula"]
            results = {}
            np.random.seed(seed)

            # --- función auxiliar para calcular métricas de simulación ---
            def _sim_metrics_from_paths(paths: np.ndarray) -> dict:
                log_ret = np.log(paths[:, 1:] / paths[:, :-1])
                mu_d = float(log_ret.mean())
                sd_d = float(log_ret.std())

                mu_ann = mu_d * freq
                sd_ann = sd_d * np.sqrt(freq)
                sharpe = (mu_ann - rf) / sd_ann if sd_ann > 0 else np.nan

                years = days / freq
                cagr = float((paths[:, -1].mean() / paths[:, 0].mean()) ** (1 / years) - 1)

                return {
                    "mu_ann": mu_ann,
                    "sd_ann": sd_ann,
                    "sharpe": sharpe,
                    "cagr": cagr,
                }

            # --- generar simulaciones ---
            for m in methods:
                try:
                    paths = self.simulate(days=days, n_paths=n_paths, mc_method=m, seed=seed)
                    results[m] = {
                        "paths": paths,
                        "metrics": _sim_metrics_from_paths(paths),
                    }
                    print(f"✅ {m.capitalize()} OK: {paths.shape}")
                except Exception as e:
                    print(f"❌ Error con método {m}: {e}")

            # --- gráfico combinado ---
            plt.figure(figsize=(11, 6))
            for m, color in zip(methods, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
                if m not in results:
                    continue

                paths = results[m]["paths"]
                mean = paths.mean(axis=0)
                p_low, p_high = np.percentile(paths, alpha_band, axis=0)

                # bandas de confianza
                plt.fill_between(
                    np.arange(days),
                    p_low,
                    p_high,
                    alpha=0.15,
                    color=color,
                )

                # media
                plt.plot(mean, color=color, lw=2, label=m.capitalize())

                # algunas trayectorias
                n_show = min(show_paths_per_method, paths.shape[0])
                for i in range(n_show):
                    plt.plot(paths[i], color=color, alpha=0.15, lw=0.6)

            plt.title(title)
            plt.xlabel("Días simulados")
            plt.ylabel("Valor de cartera")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            if save_path:
                from pathlib import Path
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=140)
                print(f"📈 Guardado gráfico en {save_path}")

            plt.show()

            # --- construir informe resumen ---
            lines = ["# Informe Monte Carlo — Comparativa de métodos\n"]
            for m in methods:
                if m not in results:
                    continue
                met = results[m]["metrics"]
                lines.append(
                    dedent(
                        f"""
                        ## {m.capitalize()}
                        - Retorno medio anualizado: {met['mu_ann']:.4f}
                        - Volatilidad anualizada:   {met['sd_ann']:.4f}
                        - Sharpe ratio:             {met['sharpe']:.3f}
                        - CAGR efectivo:            {met['cagr']:.4f}
                        """
                    )
                )

            report = "\n".join(lines)

            # Guardar informe
            if save_path:
                import os
                report_path = os.path.splitext(save_path)[0] + "_report.md"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"📝 Informe guardado en {report_path}")

            return report


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
        try:
            st = self.stats(freq=freq, rf=rf)
            r = self.portfolio_returns().dropna()
            nobs = len(r)
            start_eff = r.index.min()
            end_eff = r.index.max()
            
            # CAGR esperado y realizado
            mean_ann = st.get("mean", np.nan)
            cagr_expected = float(np.exp(mean_ann) - 1.0) if np.isfinite(mean_ann) else np.nan

            eq = np.exp(r.cumsum())
            years = (pd.to_datetime(end_eff) - pd.to_datetime(start_eff)).days / 365.25 if nobs > 1 else np.nan
            growth = float(eq.iloc[-1] / eq.iloc[0]) if len(eq) >= 2 else np.nan
            cagr_real = float(growth ** (1.0 / years) - 1.0) if (np.isfinite(growth) and growth > 0 and np.isfinite(years) and years > 0) else np.nan

        except Exception as e:
            return f"# Portfolio Report\n\n**Error preparando métricas:** `{e}`"
        
        def _max_drawdown(x: pd.Series) -> float:
            eq = (x.cumsum()).pipe(np.exp)
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
            f"- Días comunes: **{nobs}**",
            "",
            "## Pesos",
            *[f"- **{k}**: {v:.2%}" for k, v in self.weights.items()],
            "",
            "## Métricas históricas (anualizadas)",
            f"- Rentabilidad media: **{st['mean']*100:.2f}%**",
            f"- Volatilidad: **{st['std']*100:.2f}%**",
            f"- Sharpe (rf={rf*100:.2f}%): **{st['sharpe'] if not np.isnan(st['sharpe']) else 'N/A'}**",
            f"- Sortino: **{st['sortino'] if not np.isnan(st['sortino']) else 'N/A'}**",
            f"- Máx. drawdown: **{mdd*100:.2f}%**",
            f"- CAGR esperado: **{cagr_expected*100:.2f}%**",
            f"- CAGR histórico: **{(cagr_real*100 if np.isfinite(cagr_real) else float('nan')):.2f}%**",
        ]

        if include_warnings:
            warns: list[str] = []
            if nobs < 100:
                warns.append(f"- ⚠️ Histórico corto: sólo **{nobs}** días comunes.")
            
            # Detectar frecuencias mixtas
            freq_info = self.detect_frequency_mismatch()
            for warning in freq_info['warnings']:
                warns.append(f"- {warning}")
            
            # Warning de correlación
            corr_warn = self.max_correlation_warning(threshold=0.5)
            if corr_warn:
                warns.append(f"- {corr_warn}")
                
            if warns:
                lines += ["", "## Advertencias", *warns]

        if include_components:
            lines += ["", "## Componentes"]
            for sym, c in self.series.items():
                try:
                    rr = c.clean(fill_method="ffill").log_returns()
                    mu_i = rr.mean() * freq
                    sd_i = rr.std() * np.sqrt(freq)
                    freq_type = c.detect_asset_frequency()
                    lines.append(f"- **{sym}**: {mu_i*100:.2f}% media, {sd_i*100:.2f}% vol ({freq_type})")
                except Exception:
                    lines.append(f"- **{sym}**: (no disponible)")

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
                    f"- Valor final esperado: **{sm['end_mean']:.4f}**",
                    f"- Banda 5–95% final: **[{sm['end_p5']:.4f}, {sm['end_p95']:.4f}]**",
                ]
            except Exception as e:
                lines += ["", "## Monte Carlo", f"- Error al simular: `{e}`"]

        return "\n".join(lines)
    
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

        # block bootstrap: elegir inicios ponderados por la prob. del primer elemento del bloque
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
        Cópula gaussiana (Iman–Conover) con recencia.
        """
        df = self._asset_returns_matrix()
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

        # Precalcular márgenes ordenadas por activo
        Xcols_sorted = [np.sort(df[c].values) for c in syms]
        p_rows = w  # prob. de bootstrap por fila (recencia)

        paths = np.zeros((n_paths, days))
        paths[:, 0] = self.initial_value

        # Generar todo vectorizado
        all_samples = rng.choice(T, size=(n_paths, days-1), replace=True, p=p_rows)
        Z = rng.standard_normal(size=(n_paths, days-1, N))
        Z_corr = Z @ L.T

        for p in range(n_paths):
            sim_log = np.zeros((days-1, N))
            for i in range(N):
                samp = Xcols_sorted[i][all_samples[p]]
                order = np.argsort(Z_corr[p, :, i])
                sim_log[order, i] = samp
            sim_rets = np.exp(sim_log) - 1.0
            v = self.initial_value
            path = [v]
            for t in range(days - 1):
                r_day = (sim_rets[t] * wport).sum()
                v = v * (1.0 + r_day)
                path.append(v)
            paths[p, :] = np.array(path)

        return paths

    
    def plot_mc_comparison_with_bands(
    pf,
    days=252,
    n_paths=2000,
    seed=123,
    methods=("gbm", "cholesky", "bootstrap", "copula"),
    title="Comparativa Monte Carlo — Cartera",
):
        import matplotlib.pyplot as plt
        import numpy as np

        colors = {
            "gbm": "#1f77b4",        # azul
            "cholesky": "#ff7f0e",   # naranja
            "bootstrap": "#2ca02c",  # verde
            "copula": "#d62728",     # rojo
        }

        plt.figure(figsize=(12, 7))

        for m in methods:
            paths = pf.simulate(days=days, n_paths=n_paths, seed=seed, mc_method=m)
            sm = pf.summarize_paths(paths)  # usa tu helper existente

            t = np.arange(paths.shape[1])
            c = colors.get(m, None)

            # banda 5–95%
            plt.fill_between(t, sm["p_low"], sm["p_high"], alpha=0.18, label=None, color=c)
            # media
            plt.plot(t, sm["mean"], lw=2.2, label=m.capitalize(), color=c)

        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


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
        *,
        align: str = "outer",          # "outer" (unión) o "inner" (intersección)
        ffill_limit: Optional[int] = 3 # Nº máx. de días para ffill visual (None o 0 para desactivar)
    ):
            """
            Visual report sin crear días artificiales en los componentes.

            - align="outer": dibuja cada serie en toda su historia y rellena huecos cortos (festivos) con ffill limitado.
            - align="inner": recorta todas las series al tramo común (intersección de fechas).
            - ffill_limit: nº máx. de días consecutivos a rellenar sólo para el gráfico.
            """

            import os
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # ---------------- 1) Componentes ----------------
            if show_components:
                # Helper: fechas -> tz-naive normalizadas al día
                def _as_naive_daily_index(s: pd.Series) -> pd.DatetimeIndex:
                    d = pd.to_datetime(s, errors="coerce")
                    if getattr(d.dt, "tz", None) is not None:
                        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
                    return pd.DatetimeIndex(d.dt.normalize())

                # Construir series limpias (índice=fecha diaria tz-naive)
                series: Dict[str, pd.Series] = {}
                for sym, c in self.series.items():
                    df = c.clean(fill_method=None).to_dataframe()
                    if "date" not in df.columns or "close" not in df.columns:
                        continue
                    df = df[["date", "close"]].copy()
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    df = df.dropna(subset=["date", "close"]).sort_values("date")
                    idx = _as_naive_daily_index(df["date"])
                    s = pd.Series(df["close"].values, index=idx)
                    s = s[~s.index.duplicated(keep="last")].sort_index()  # por si hay duplicados de día
                    if not s.empty:
                        series[sym] = s

                if series:
                    align_mode = (align or "outer").lower()
                    aligned: Dict[str, pd.Series] = {}

                    if align_mode == "inner":
                        # Intersección de todos los índices
                        common_idx = None
                        for s in series.values():
                            common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
                        common_idx = pd.DatetimeIndex(common_idx) if common_idx is not None else None

                        for sym, s in series.items():
                            sr = s.reindex(common_idx)
                            if ffill_limit and ffill_limit > 0:
                                sr = sr.ffill(limit=int(ffill_limit))
                            aligned[sym] = sr

                    else:
                        # Unión de todos los índices
                        union_idx = None
                        for s in series.values():
                            union_idx = s.index if union_idx is None else union_idx.union(s.index)
                        union_idx = pd.DatetimeIndex(union_idx) if union_idx is not None else None

                        for sym, s in series.items():
                            sr = s.reindex(union_idx)
                            if ffill_limit and ffill_limit > 0:
                                sr = sr.ffill(limit=int(ffill_limit))
                            aligned[sym] = sr

                    # Rango temporal para el título
                    min_d, max_d = None, None
                    for s in aligned.values():
                        v = s.dropna()
                        if v.empty:
                            continue
                        mi, ma = v.index.min(), v.index.max()
                        min_d = mi if min_d is None or mi < min_d else min_d
                        max_d = ma if max_d is None or ma > max_d else max_d

                    # Plot
                    plt.close("all")
                    plt.figure(figsize=(11, 5), dpi=dpi)
                    for sym, s in aligned.items():
                        v = s.dropna()
                        if v.empty:
                            continue
                        y = v.astype(float)
                        if normalize:
                            y = y / y.iloc[0]
                        plt.plot(v.index, y.values, label=sym, linewidth=1.5)

                    ttl = "Componentes — cierres normalizados"
                    if align_mode == "inner":
                        ttl += " (inner)"
                    else:
                        ttl += " (outer)"
                    if min_d is not None and max_d is not None:
                        ttl += f"\nPeríodo común: {min_d.date()} → {max_d.date()}"
                    plt.title(ttl)
                    plt.grid(alpha=0.3)
                    plt.legend()
                    if logy:
                        plt.yscale("log")
                    plt.tight_layout()
                    if save_dir:
                        plt.savefig(os.path.join(save_dir, "components.png"), bbox_inches="tight")

            # ---------------- 2) Histograma de retornos ----------------
            if show_hist:
                plt.figure(figsize=(8, 4), dpi=dpi)
                r = self.portfolio_returns().dropna()
                plt.hist(r.values, bins=40, alpha=0.8)
                plt.title("Histograma de retornos (log, diarios)")
                plt.grid(alpha=0.3); plt.tight_layout()
                if save_dir:
                    plt.savefig(os.path.join(save_dir, "hist_returns.png"), bbox_inches="tight")

            # ---------------- 3) Correlaciones ----------------
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

            # ---------------- 4) Monte Carlo ----------------
            if show_mc:
                paths = self.simulate(days=mc_days, n_paths=mc_paths, seed=seed,
                                    mu_scale=mu_scale, sigma_scale=sigma_scale)
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



        def random_portfolios_summary(
            self,
            n: int = 1000,
            alpha_dirichlet: float = 1.0,
            alpha_var: float = 0.95,
            freq: int = 252,
            seed: Optional[int] = 123,
            topk: int = 3,
        ) -> dict:
            res = self.random_portfolios(n=n, alpha_dirichlet=alpha_dirichlet, alpha_var=alpha_var, freq=freq, seed=seed)
            best_cvar = res.nsmallest(topk, "CVaR").copy()
            worst_cvar = res.nlargest(topk, "CVaR").copy()
            best_sharpe = res.nlargest(topk, "sharpe").copy()
            worst_sharpe = res.nsmallest(topk, "sharpe").copy()

            def _fmt_row(row):
                ws = ", ".join(f"{k}:{v:.2%}" for k, v in row["weights"].items())
                return f"Sharpe={row['sharpe']:.3f} | CVaR={row['CVaR']:.4f} | Weights: {ws}"

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
    