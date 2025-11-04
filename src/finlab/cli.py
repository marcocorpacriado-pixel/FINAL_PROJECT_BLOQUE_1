from pathlib import Path
from typing import List
import typer
from dotenv import load_dotenv
from typing import List, Optional
from finlab.extractor.yahoo import fetch_prices_yahoo



# -----------------------------
# APP raíz y subgrupos
# -----------------------------
app = typer.Typer(help="Extractor y simulador financiero multi-API (base).")

fetch_app = typer.Typer(help="Descarga datos desde distintas APIs.")
app.add_typer(fetch_app, name="fetch")

simulate_app = typer.Typer(help="Simulaciones Monte Carlo")
app.add_typer(simulate_app, name="simulate")

# -----------------------------
# Callback global (dotenv)
# -----------------------------
@app.callback()
def _init():
    """Carga variables de entorno (.env) si existe."""
    load_dotenv()

# -----------------------------
# Comandos básicos
# -----------------------------
@app.command("version")
def version():
    """Muestra la versión del paquete."""
    typer.echo("finlab 0.1.0")

@app.command("hello")
def hello(name: str = typer.Argument("mundo", help="Nombre al que saludar")):
    """Comando de prueba."""
    typer.echo(f"Hola, {name}!")

# -----------------------------
# FETCH: Alphavantage / TwelveData / Marketstack / Batch
# -----------------------------
from finlab.extractor.alphavantage import fetch_prices_alphavantage
from finlab.extractor.twelvedata import fetch_prices_twelvedata
from finlab.extractor.marketstack import fetch_prices_marketstack

@fetch_app.command("alphavantage")
def fetch_alphavantage_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Símbolo bursátil, p.ej. AAPL o MSFT"),
    start: str = typer.Option(None, "--start", help="Fecha inicial (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="Fecha final (YYYY-MM-DD)"),
    outdir: Path = typer.Option(Path("data"), "--outdir", help="Carpeta base salida"),
    adjusted: bool = typer.Option(False, "--adjusted", help="Usar TIME_SERIES_DAILY_ADJUSTED"),
    outputsize: str = typer.Option("compact", "--outputsize", help="compact | full"),
    fmt: str = typer.Option("csv", "--format", help="Formato: csv | parquet"),
):
    """Descarga precios diarios desde AlphaVantage (elige CSV o Parquet)."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_alphavantage(
        symbol, outdir, start=start, end=end, adjusted=adjusted, outputsize=outputsize, fmt=fmt
    )
    typer.echo(f"✅ Archivo guardado: {path}")


@fetch_app.command("twelvedata")
def fetch_twelvedata_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Símbolo bursátil o cripto, ej: AAPL o BTC/USD"),
    interval: str = typer.Option("1day", "--interval", help="Intervalo: 1min, 5min, 1h, 1day, 1week..."),
    start: str = typer.Option(None, "--start", help="Fecha inicial YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="Fecha final YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data/twelvedata"), "--outdir", help="Carpeta destino"),
    fmt: str = typer.Option("csv", "--format", help="Formato de salida: csv | parquet"),
):
    """Descarga precios desde TwelveData (elige CSV o Parquet)."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_twelvedata(symbol, outdir, interval=interval, start=start, end=end, fmt=fmt)
    typer.echo(f"✅ Archivo guardado: {path}")


@fetch_app.command("marketstack")
def fetch_marketstack_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Símbolo bursátil, p.ej. TSLA"),
    start: str = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data/marketstack"), "--outdir", help="Carpeta destino"),
    fmt: str = typer.Option("csv", "--format", help="Formato: csv | parquet"),
):
    """Descarga precios EOD desde MarketStack (elige CSV o Parquet)."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_marketstack(symbol, outdir, start=start, end=end, fmt=fmt)
    typer.echo(f"✅ Archivo guardado: {path}")

@fetch_app.command("yahoo")
def fetch_yahoo_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Ticker Yahoo, p.ej. SPY, BTC-USD, SAP.DE, REMX"),
    interval: str = typer.Option("1d", "--interval", help="1m,5m,15m,1h,1d,1wk,1mo"),
    start: str = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data"), "--outdir"),
    fmt: str = typer.Option("parquet", "--format", help="csv | parquet"),
    auto_adjust: bool = typer.Option(False, "--auto-adjust", help="Ajustar OHLC por dividendos/splits"),
):
    """Descarga precios desde Yahoo Finance para un símbolo."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_yahoo(
        symbol, outdir, interval=interval, start=start, end=end, fmt=fmt, auto_adjust=auto_adjust
    )
    typer.echo(f"✅ Archivo guardado: {path}")


@fetch_app.command("batch")
def fetch_batch(
    provider: str = typer.Argument(..., help="Proveedor: alphavantage | twelvedata | marketstack | yahoo"),
    symbols: str = typer.Option(..., "--symbols", help="Lista separada por comas, p.ej. AAPL,MSFT,SPY"),
    start: str = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data"), "--outdir", help="Carpeta base de salida"),
    interval: str = typer.Option("1day", "--interval", help="(solo TwelveData/Yahoo) 1min, 5min, 1h, 1day, 1d, 1wk..."),
    fmt: str = typer.Option("csv", "--format", help="Formato: csv | parquet"),
    adjusted: bool = typer.Option(False, "--adjusted", help="(AlphaVantage) usar DAILY_ADJUSTED"),
    outputsize: str = typer.Option("compact", "--outputsize", help="(AlphaVantage) compact | full"),
    max_workers: int = typer.Option(4, "--max-workers", help="Número máximo de descargas simultáneas"),
):
    """
    Descarga N series en lote desde el proveedor indicado (en paralelo).
    Ejemplos:
      finlab fetch batch alphavantage --symbols AAPL,MSFT,SPY --format parquet
      finlab fetch batch twelvedata --symbols BTC/USD,ETH/USD --interval 1day --format parquet
      finlab fetch batch marketstack --symbols TSLA,SPY --start 2024-01-01 --end 2024-03-01 --format parquet
      finlab fetch batch yahoo --symbols QQQ,GLD,BTC-USD --interval 1d --format parquet
    """
    import concurrent.futures
    from functools import partial

    syms: List[str] = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise typer.BadParameter("Debes indicar al menos un símbolo en --symbols")

    prov = provider.lower()
    saved: List[Path] = []

    # --- Selecciona la función correspondiente ---
    if prov == "alphavantage":
        fetch_func = partial(
            fetch_prices_alphavantage,
            outdir=outdir,
            start=start,
            end=end,
            adjusted=adjusted,
            outputsize=outputsize,
            fmt=fmt,
        )

    elif prov == "twelvedata":
        fetch_func = partial(
            fetch_prices_twelvedata,
            outdir=outdir / "twelvedata",
            interval=interval,
            start=start,
            end=end,
            fmt=fmt,
        )

    elif prov == "marketstack":
        fetch_func = partial(
            fetch_prices_marketstack,
            outdir=outdir / "marketstack",
            start=start,
            end=end,
            fmt=fmt,
        )

    elif prov == "yahoo":
        fetch_func = partial(
            fetch_prices_yahoo,
            outdir=outdir / "yahoo",
            interval=interval,
            start=start,
            end=end,
            fmt=fmt,
        )

    else:
        raise typer.BadParameter("Proveedor desconocido. Usa: alphavantage | twelvedata | marketstack | yahoo")

    # --- Descarga en paralelo ---
    typer.echo(f"🚀 Iniciando descarga paralela de {len(syms)} símbolos desde {prov} (máx {max_workers} hilos)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(fetch_func, s): s for s in syms}
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                path = future.result()
                saved.append(path)
                typer.echo(f"✅ {sym}: guardado en {path}")
            except Exception as e:
                typer.echo(f"❌ Error en {sym}: {e}")

    typer.echo("\n📦 Descargas completadas:")
    for p in saved:
        typer.echo(f"  - {p}")



# -----------------------------
# SIMULATE: asset / portfolio
# -----------------------------
from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

@simulate_app.command("asset")
def simulate_asset(
    input: Path = typer.Option(..., "--input", help="Ruta a CSV o Parquet con columnas estándar (date, open, high, low, close)"),
    days: int = typer.Option(252, "--days"),
    n_paths: int = typer.Option(1000, "--n-paths"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    initial_value: float = typer.Option(1.0, "--initial-value"),
):
    """Simula un activo a partir de su archivo (CSV o Parquet) usando GBM univariado."""
    c = Candles.from_any(input).clean(fill_method="ffill").to_business_days(fill=True)
    r = c.log_returns()
    mu = float(r.mean())
    sigma = float(r.std())
    dt = 1/252

    import numpy as np
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, days))
    paths[:, 0] = initial_value
    for t in range(1, days):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

    end_vals = paths[:, -1]
    print(f"✅ Simulación activo: mean={end_vals.mean():.4f}, p5={np.percentile(end_vals,5):.4f}, p95={np.percentile(end_vals,95):.4f}")

@simulate_app.command("portfolio")
def simulate_portfolio(
    inputs: str = typer.Option(..., "--inputs", help="Rutas a CSV/Parquet separadas por comas"),
    weights: str = typer.Option(..., "--weights", help="Pesos separados por comas; mismo orden que --inputs"),
    days: int = typer.Option(252, "--days"),
    n_paths: int = typer.Option(1000, "--n-paths"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    initial_value: float = typer.Option(1.0, "--initial-value"),
    by_components: bool = typer.Option(False, "--components", help="Simular también cada activo y combinar"),
    mc_method: str = typer.Option("gbm", "--mc-method", help="gbm | cholesky | copula | bootstrap"),
    # ❌ ELIMINAR estos parámetros que no existen en la versión simplificada
    # halflife_days: Optional[float] = typer.Option(None, "--halflife-days", help="Semivida EWMA para ponderar recencia"),
    # block_len: int = typer.Option(10, "--block-len", help="Tamaño de bloque para bootstrap"),
    mu_scale: float = typer.Option(1.0, "--mu-scale", help="Escala de la media diaria"),
    sigma_scale: float = typer.Option(1.0, "--sigma-scale", help="Escala de la volatilidad diaria"),
    save_dir: Optional[Path] = typer.Option(None, "--save-dir", help="Carpeta para PNGs y report.md"),
):
    """
    Simula una cartera definida por varios archivos (CSV o Parquet) + pesos.
    """
    from finlab.models.candles import Candles
    from finlab.models.portfolio import Portfolio
    import numpy as np
    import os

    paths_list = [p.strip() for p in inputs.split(",") if p.strip()]
    ws_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
    if len(paths_list) != len(ws_list):
        raise typer.BadParameter("El número de --inputs y --weights debe coincidir")

    candles = []
    symbols = []
    for p in paths_list:
        # ✅ CORREGIDO: usar fill_method="none" en lugar de fill=True
        c = Candles.from_any(Path(p)).clean(fill_method="ffill").to_business_days(fill_method="none")
        candles.append(c)
        symbols.append(c.symbol)

    series = {c.symbol: c for c in candles}
    wmap = {sym: w for sym, w in zip(symbols, ws_list)}
    port = Portfolio(series=series, weights=wmap, initial_value=initial_value)

    # Detección de frecuencias mixtas
    freq_info = port.detect_frequency_mismatch()
    for warning in freq_info['warnings']:
        typer.echo(typer.style(warning, fg=typer.colors.YELLOW))

    # Ejecuta simulación
    if by_components:
        comps, port_paths = port.simulate_components(
            days=days, n_paths=n_paths, seed=seed, mu_scale=mu_scale, sigma_scale=sigma_scale
        )
        end_vals = port_paths[:, -1]
        typer.echo(f"✅ Simulación cartera (componentes): mean={end_vals.mean():.4f}, p5={np.percentile(end_vals,5):.4f}, p95={np.percentile(end_vals,95):.4f}")
        typer.echo(f"   Componentes simulados: {list(comps.keys())}")
    else:
        # ✅ CORREGIDO: eliminar parámetros que no existen
        port_paths = port.simulate(
            days=days,
            n_paths=n_paths,
            seed=seed,
            mu_scale=mu_scale,
            sigma_scale=sigma_scale,
            mc_method=mc_method,
            # ❌ ELIMINADOS: halflife_days y block_len
        )
        end_vals = port_paths[:, -1]
        typer.echo(f"✅ Simulación cartera ({mc_method}): mean={end_vals.mean():.4f}, p5={np.percentile(end_vals,5):.4f}, p95={np.percentile(end_vals,95):.4f}")

    # Guardado de reportes/gráficos (opcional)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Reporte Markdown
        md = port.report(
            mc_days=days,
            mc_paths=n_paths,
            mu_scale=mu_scale,
            sigma_scale=sigma_scale,
        )
        (Path(save_dir) / "report.md").write_text(md, encoding="utf-8")
        # Gráficos
        port.plots_report(
            save_dir=str(save_dir),
            mc_days=days,
            mc_paths=n_paths,
            mu_scale=mu_scale,
            sigma_scale=sigma_scale,
        )
        typer.echo(f"📝 report.md y 📈 PNGs guardados en: {save_dir}")

@simulate_app.command("sweep")
def simulate_sweep(
    inputs: str = typer.Option(..., "--inputs", help="Rutas a CSV/Parquet separadas por comas"),
    n: int = typer.Option(1000, "--n"),
    alpha_dirichlet: float = typer.Option(1.0, "--alpha-dirichlet"),
    alpha_var: float = typer.Option(0.95, "--alpha-var"),
    seed: Optional[int] = typer.Option(123, "--seed"),
):
    """
    Genera N carteras aleatorias (pesos ~ Dirichlet) con los activos dados
    y muestra top/bottom por CVaR y por Sharpe.
    """
    from pathlib import Path

    paths_list = [p.strip() for p in inputs.split(",") if p.strip()]
    candles = []
    for p in paths_list:
        c = Candles.from_any(Path(p)).clean(fill_method="ffill").to_business_days(fill=True)
        candles.append(c)

    series = {c.symbol: c for c in candles}
    dummy_weights = {sym: 1.0 / len(series) for sym in series.keys()}
    port = Portfolio(series=series, weights=dummy_weights)

    warn = port.max_correlation_warning(threshold=0.5)
    if warn:
        print(warn)

    summary = port.random_portfolios_summary(
        n=n, alpha_dirichlet=alpha_dirichlet, alpha_var=alpha_var, seed=seed
    )


if __name__ == "__main__":
    app()
