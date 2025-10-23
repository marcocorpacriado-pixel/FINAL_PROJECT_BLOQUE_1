📈 FINLAB — Simulación y Análisis de Carteras con Monte Carlo

Proyecto final del Bloque 1 del Máster en Inteligencia Artificial y Computación Cuántica aplicada a los Mercados Financieros (MIAX).

Este proyecto implementa una plataforma modular en Python para:

Descargar y normalizar precios desde múltiples APIs financieras
(TwelveData, Alpha Vantage, MarketStack).

Estandarizar los datos con un modelo Candles (serie OHLC).

Crear carteras (Portfolio) con estadísticas, correlaciones y reportes.

Ejecutar simulaciones de Monte Carlo con distintos modelos:

Movimiento Browniano Geométrico (GBM)

Simulación correlacionada por Cholesky

Dependencia no lineal con Cópulas Gaussianas

Bootstrap por bloques (método no paramétrico)

Generar gráficos y reportes automáticos en formato Markdown y PNG.

🚀 Instalación

git clone https://github.com/marcocorpacriado-pixel/FINAL_PROJECT_BLOQUE_1.git
cd FINAL_PROJECT_BLOQUE_1

python -m venv .venv
. .venv/Scripts/activate     # En PowerShell: .\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
cp .env.example .env         # y añade tus claves API

🔑 Configuración de claves

Copia .env.example a .env y rellena tus claves personales:

ALPHAVANTAGE_API_KEY=tu_clave
MARKETSTACK_API_KEY=tu_clave
TWELVEDATA_API_KEY=tu_clave

💾 Estructura del proyecto

src/finlab/
│
├── cli.py               # Interfaz de línea de comandos (Typer)
├── extractor/           # APIs: TwelveData, AlphaVantage, Marketstack
├── models/
│   ├── candles.py       # Serie OHLC normalizada (DataClass)
│   └── portfolio.py     # Cartera + Monte Carlo + reportes
│
├── data/                # Datos descargados (por proveedor/símbolo)
├── outputs/             # Resultados y gráficos
└── run_plot_mc.py       # Script de demo comparando métodos de Monte Carlo

📡 Descarga de datos (CLI)

Ejemplo con TwelveData:

python -m finlab.cli fetch twelvedata \
  --symbol BTC/USD --interval 1day --format parquet

python -m finlab.cli fetch twelvedata \
  --symbol ETH/USD --interval 1day --format parquet


Los archivos se guardan en data/twelvedata/<SYMBOL>/<SYMBOL>_<interval>.parquet.

🧠 Simulación de cartera (CLI)

python -m finlab.cli simulate portfolio \
  --inputs "data/twelvedata/BTC_USD/BTC_USD_1day.parquet,data/twelvedata/ETH_USD/ETH_USD_1day.parquet" \
  --weights "0.5,0.5" \
  --mc-method cholesky \
  --halflife-days 90 \
  --block-len 10 \
  --save-dir outputs/cli_demo


El resultado incluye métricas, gráficos y reportes Markdown de la simulación.


🧩 Scripts de ejemplo

▶ run_plot_mc.py

Simula una cartera BTC/ETH con los 4 métodos Monte Carlo y guarda los resultados en outputs/mc/.

python run_plot_mc.py


Genera automáticamente:

Archivo	Descripción
mc_gbm_paths.png	Trayectorias simuladas (GBM)
mc_cholesky_paths.png	Trayectorias correlacionadas
mc_copula_paths.png	Dependencia vía cópula
mc_bootstrap_paths.png	Re-muestreo por bloques
mc_compare_bands.png	Bandas 5–95 % y medias comparadas
mc_compare_terminal_all.png	Densidades del valor final

⚙️ Componentes principales

Candles

Limpieza de series (clean, validate).

Normalización de columnas.

Re-muestreo a días laborables (to_business_days).

Retornos logarítmicos (log_returns).

Gráficos básicos.

Portfolio

Combinación de activos y pesos.

Estadísticas (media, volatilidad, Sharpe, VaR, CVaR).

Simulación Monte Carlo:

gbm — proceso univariado.

cholesky — correlaciones lineales.

copula — dependencias no lineales.

bootstrap — bloques históricos.

Ponderación exponencial (más peso a retornos recientes).

Reportes Markdown y gráficos automáticos.

📊 Ejemplo de salida
--- Método Monte Carlo: GBM ---
Final esperado: 1.0010 | p5=0.9465 | p95=1.0580

--- Método Monte Carlo: CHOLESKY ---
Final esperado: 1.6588 | p5=0.6265 | p95=3.2771

--- Método Monte Carlo: COPULA ---
Final esperado: 1.5223 | p5=0.7409 | p95=2.8784

--- Método Monte Carlo: BOOTSTRAP ---
Final esperado: 1.8072 | p5=0.7154 | p95=3.6933

🧩 Rango efectivo de datos

El reporte incluye automáticamente el rango de fechas efectivo de las series alineadas:

## Rango de datos efectivo
- Desde: **2018-05-20**
- Hasta: **2025-10-23**


Esto garantiza que todas las métricas y simulaciones se basen en el periodo común real.

🧮 Métodos Monte Carlo
Método	Descripción	Ventajas
gbm	Movimiento Browniano Geométrico con μ, σ históricos.	Sencillo y base teórica.
cholesky	Multivariante, preserva correlaciones lineales.	Captura co-movimientos.
copula	Dependencias no lineales con cópula gaussiana.	Márgenes empíricas, correlación flexible.
bootstrap	Re-muestreo histórico (bloques).	No paramétrico, mantiene volatilidad agrupada.

Parámetros clave:

halflife_days: pondera retornos recientes (EWMA).

block_len: tamaño de bloque (bootstrap).

n_paths: nº de trayectorias.

days: nº de días simulados.

📚 Dependencias principales

-pandas, numpy, matplotlib

-typer (CLI)

-requests (APIs)

-pyarrow (Parquet)

-tqdm (progreso)

-scipy, statsmodels (estadística avanzada)

-pytest, ruff, black, isort (dev tools)

🧾 Licencia

MIT License — libre uso académico y profesional.
© 2025 Marco Corpa Criado

