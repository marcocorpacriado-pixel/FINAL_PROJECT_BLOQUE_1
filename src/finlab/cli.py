from pathlib import Path
from typing import List
import typer
from dotenv import load_dotenv
from typing import List, Optional

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
    outdir: Path = typer.Option(Path("data"), "--outdir", help="Carpeta donde guardar el CSV")
):
    """Descarga precios diarios desde AlphaVantage."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_alphavantage(symbol, outdir, start=start, end=end)
    typer.echo(f"✅ Archivo guardado: {path}")

@fetch_app.command("twelvedata")
def fetch_twelvedata_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Símbolo bursátil o cripto, ej: AAPL o BTC/USD"),
    interval: str = typer.Option("1day", "--interval", help="Intervalo: 1min, 5min, 1h, 1day, 1week..."),
    start: str = typer.Option(None, "--start", help="Fecha inicial YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="Fecha final YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data/twelvedata"), "--outdir", help="Carpeta destino CSV")
):
    """Descarga precios desde TwelveData."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_twelvedata(symbol, outdir, interval=interval, start=start, end=end)
    typer.echo(f"✅ Archivo guardado: {path}")

@fetch_app.command("marketstack")
def fetch_marketstack_cmd(
    symbol: str = typer.Option(..., "--symbol", help="Símbolo bursátil, p.ej. TSLA"),
    start: str = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data/marketstack"), "--outdir", help="Carpeta destino CSV")
):
    """Descarga precios EOD desde MarketStack."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = fetch_prices_marketstack(symbol, outdir, start=start, end=end)
    typer.echo(f"✅ Archivo guardado: {path}")

@fetch_app.command("batch")
def fetch_batch(
    provider: str = typer.Argument(..., help="Proveedor: alphavantage | twelvedata | marketstack"),
    symbols: str = typer.Option(..., "--symbols", help="Lista separada por comas, p.ej. AAPL,MSFT,SPY"),
    start: str = typer.Option(None, "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="YYYY-MM-DD"),
    outdir: Path = typer.Option(Path("data"), "--outdir", help="Carpeta base de salida"),
    interval: str = typer.Option("1day", "--interval", help="(solo TwelveData) 1min, 5min, 1h, 1day, ..."),
):
    """
    Descarga N series en lote desde el proveedor indicado.
    Ejemplo:
      finlab fetch batch alphavantage --symbols AAPL,MSFT,SPY
      finlab fetch batch twelvedata --symbols BTC/USD,ETH/USD --interval 1day
      finlab fetch batch marketstack --symbols TSLA,SPY --start 2024-01-01 --end 2024-03-01
    """
    syms: List[str] = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise typer.BadParameter("Debes indicar al menos un símbolo en --symbols")

    saved: List[Path] = []
    if provider.lower() == "alphavantage":
        target = outdir / "alphavantage" / "prices"
        target.mkdir(parents=True, exist_ok=True)
        for s in syms:
            p = fetch_prices_alphavantage(s, target, start=start, end=end)
            saved.append(p)
    elif provider.lower() == "twelvedata":
        target = outdir / "twelvedata"
        for s in syms:
            p = fetch_prices_twelvedata(s, target, interval=interval, start=start, end=end)
            saved.append(p)
    elif provider.lower() == "marketstack":
        target = outdir / "marketstack"
        for s in syms:
            p = fetch_prices_marketstack(s, target, start=start, end=end)
            saved.append(p)
    else:
        raise typer.BadParameter("Proveedor desconocido. Usa: alphavantage | twelvedata | marketstack")

    typer.echo("✅ Descargas completadas:")
    for p in saved:
        typer.echo(f"  - {p}")

# -----------------------------
# SIMULATE: asset / portfolio
# -----------------------------
from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

@simulate_app.command("asset")
def simulate_asset(
    input: Path = typer.Option(..., "--input", help="CSV con columnas estándar (date, open, high, low, close)"),
    days: int = typer.Option(252, "--days"),
    n_paths: int = typer.Option(1000, "--n-paths"),
    seed: Optional[int] = typer.Option(None, "--seed"),   # <-- antes: int
    initial_value: float = typer.Option(1.0, "--initial-value"),
):
    """Simula un activo a partir de su CSV (GBM univariado)."""
    c = Candles.from_csv(input).clean(fill_method="ffill").to_business_days(fill=True)
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
    import numpy as np
    print(f"✅ Simulación activo: mean={end_vals.mean():.4f}, p5={np.percentile(end_vals,5):.4f}, p95={np.percentile(end_vals,95):.4f}")

@simulate_app.command("portfolio")
def simulate_portfolio(
    inputs: str = typer.Option(..., "--inputs", help="Rutas CSV separadas por comas"),
    weights: str = typer.Option(..., "--weights", help="Pesos separados por comas; mismo orden que --inputs"),
    days: int = typer.Option(252, "--days"),
    n_paths: int = typer.Option(1000, "--n-paths"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    initial_value: float = typer.Option(1.0, "--initial-value"),
    by_components: bool = typer.Option(False, "--components", help="Simular también cada activo y combinar"),
):
    """
    Simula una cartera definida por varios CSV + pesos.
    - Por defecto: simula la cartera como un único GBM (rápido).
    - Con --components: simula cada activo y compone la cartera por pesos (rebalanceo diario).
    """
    paths_list = [p.strip() for p in inputs.split(",") if p.strip()]
    ws_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
    if len(paths_list) != len(ws_list):
        raise typer.BadParameter("El número de --inputs y --weights debe coincidir")

    candles = []
    symbols = []
    for p in paths_list:
        c = Candles.from_csv(Path(p)).clean(fill_method="ffill").to_business_days(fill=True)
        candles.append(c)
        symbols.append(c.symbol)

    series = {c.symbol: c for c in candles}
    wmap = {sym: w for sym, w in zip(symbols, ws_list)}
    port = Portfolio(series=series, weights=wmap, initial_value=initial_value)

    if by_components:
        comps, port_paths = port.simulate_components(days=days, n_paths=n_paths, seed=seed)
        import numpy as np
        end_vals = port_paths[:, -1]
        print(f"✅ Simulación cartera (componentes): mean={end_vals.mean():.4f}, p5={np.percentile(end_vals,5):.4f}, p95={np.percentile(end_vals,95):.4f}")
        print(f"   Componentes simulados: {list(comps.keys())}")
    else:
        port_paths = port.simulate(days=days, n_paths=n_paths, seed=seed)
        import numpy as np
        end_vals = port_paths[:, -1]
        print(f"✅ Simulación cartera (GBM global): mean={end_vals.mean():.4f}, p5={np.percentile(end_vals,5):.4f}, p95={np.percentile(end_vals,95):.4f}")

@simulate_app.command("sweep")
def simulate_sweep(
    inputs: str = typer.Option(..., "--inputs", help="Rutas CSV separadas por comas"),
    n: int = typer.Option(1000, "--n"),
    alpha_dirichlet: float = typer.Option(1.0, "--alpha-dirichlet"),
    alpha_var: float = typer.Option(0.95, "--alpha-var"),
    seed: Optional[int] = typer.Option(123, "--seed"),
):
    """
    Genera N carteras aleatorias (pesos ~ Dirichlet) con los activos dados
    y muestra top/bottom por CVaR y por Sharpe.
    """
    from finlab.models.candles import Candles
    from finlab.models.portfolio import Portfolio
    from pathlib import Path

    paths_list = [p.strip() for p in inputs.split(",") if p.strip()]
    candles = []
    for p in paths_list:
        c = Candles.from_csv(Path(p)).clean(fill_method="ffill").to_business_days(fill=True)
        candles.append(c)

    series = {c.symbol: c for c in candles}
    # pesos iniciales no importan: el barrido genera nuevos pesos
    dummy_weights = {sym: 1.0 / len(series) for sym in series.keys()}
    port = Portfolio(series=series, weights=dummy_weights)

    # aviso de correlación alta
    warn = port.max_correlation_warning(threshold=0.5)
    if warn:
        print(warn)

    summary = port.random_portfolios_summary(
        n=n, alpha_dirichlet=alpha_dirichlet, alpha_var=alpha_var, seed=seed
    )

# -----------------------------
# MAIN (¡AL FINAL!)
# -----------------------------
if __name__ == "__main__":
    app()
