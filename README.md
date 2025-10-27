# 📈 FINLAB — Simulación y Análisis de Carteras con Monte Carlo

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-pytest-blue)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Plataforma modular en Python para **descargar, normalizar y analizar series OHLC** desde múltiples APIs financieras, construir **carteras**, y ejecutar **simulaciones de Monte Carlo** (GBM, Cholesky, Cópulas, Bootstrap). Incluye CLI con ejecución **paralela** y reportes automáticos (Markdown y gráficos).

> Proyecto final del Bloque 1 del Máster en Inteligencia Artificial y Computación Cuántica aplicada a los Mercados Financieros (MIAX).

---

## 🗂️ Índice

- [Características](#-características)
- [Arquitectura & Contratos de Datos](#-arquitectura--contratos-de-datos)
- [Instalación](#-instalación)
- [Configuración de claves](#-configuración-de-claves)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Uso rápido (Quickstart)](#-uso-rápido-quickstart)
- [Referencia CLI](#-referencia-cli)
- [Notas sobre Fechas y Proveedores](#-notas-sobre-fechas-y-proveedores)
- [Ejemplos reproducibles](#-ejemplos-reproducibles)
- [Reportes y Visualizaciones](#-reportes-y-visualizaciones)
- [Rendimiento & Paralelismo](#-rendimiento--paralelismo)
- [Calidad, Pruebas y Estilo](#-calidad-pruebas-y-estilo)
- [Roadmap](#-roadmap)
- [Licencia](#-licencia)

---

## ✨ Características

- **Extractores multi-API**: TwelveData, Alpha Vantage, MarketStack.
- **Estandarización OHLC** con `Candles` (dataclass): limpieza, validación, resample a días laborables, retornos log.
- **Carteras (`Portfolio`)**: métricas (μ, σ, Sharpe, VaR, CVaR), correlaciones, avisos, reportes Markdown.
- **Monte Carlo**:
  - `gbm`: proceso univariado.
  - `cholesky`: correlaciones lineales multivariantes.
  - `copula`: dependencias no lineales (Iman–Conover sobre cópula gaussiana).
  - `bootstrap`: block bootstrap con recencia.
  - **Ponderación exponencial** de la historia (EWMA).
- **CLI profesional** con **descarga en paralelo** (`--max-workers`) y batch de símbolos.
- **Reportes**: `.report()` (Markdown) y `.plots_report()` (PNG), listos para documentar.

---

## 🧱 Arquitectura & Contratos de Datos

**Diagrama (alto nivel)**
APIs (TwelveData / AlphaVantage / MarketStack)
│
▼
extractors/* → CSV/Parquet normalizados
│
▼
models.Candles → limpieza/validación/returns
│
▼
models.Portfolio → métricas, MC, reportes
│
▼
CLI Typer → flujos plug-n-play



**Contrato `Candles.frame` (estándar mínimo):**

| Columna | Tipo | Descripción |
|--------|------|-------------|
| `date` | datetime | índice temporal |
| `open` | float | apertura |
| `high` | float | máximo |
| `low`  | float | mínimo |
| `close`| float | cierre |
| `volume` | float | opcional |
| `adj_close` | float | opcional |

Cualquier extractor **mapea** sus columnas al esquema estándar (_data contract_), garantizando que el resto del sistema sea **agnóstico** a la fuente original.

---

## 🚀 Instalación

```bash
git clone https://github.com/marcocorpacri/FINAL_PROJECT_BLOQUE_1
cd FINAL_PROJECT_BLOQUE_1

python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
cp .env.example .env
```

## 🔑 Configuración de claves

Edita `.env` con tus claves:

ALPHAVANTAGE_API_KEY=tu_clave  
MARKETSTACK_API_KEY=tu_clave  
TWELVEDATA_API_KEY=tu_clave

## **📦 Estructura del proyecto**
src/finlab/
│
├── cli.py                       # CLI (Typer)
├── extractor/
│   ├── alphavantage.py
│   ├── twelvedata.py
│   ├── marketstack.py
│   └── io_utils.py
├── models/
│   ├── candles.py               # dataclass Candles
│   └── portfolio.py             # Portfolio + MC + reportes
│
├── data/                        # salida de extractores
├── outputs/                     # gráficos y reportes
└── run_plot_mc.py               # script de demo


## ** ⚡ Uso rápido (Quickstart)**
**DESCARGA PARALELA (ALPHA VANTAGE)**
python -m finlab.cli fetch batch alphavantage \
  --symbols AAPL,NVDA,MSFT \
  --format csv \
  --outputsize compact \
  --max-workers 4

**Descarga con rango de fechas (TwelveData)**
python -m finlab.cli fetch batch twelvedata \
  --symbols AAPL,NVDA \
  --interval 1day \
  --start 2024-01-01 --end 2025-01-01 \
  --format parquet \
  --max-workers 4

**Simulación de Cartera (CLI)**
python -m finlab.cli simulate portfolio \
  --inputs "data/twelvedata/AAPL/AAPL_1day.parquet,data/twelvedata/NVDA/NVDA_1day.parquet" \
  --weights "0.6,0.4" \
  --days 252 --n-paths 2000




##**REFERENCIA CLI**##
python -m finlab.cli --help
python -m finlab.cli fetch --help
python -m finlab.cli simulate --help

**Comandos principales**
--fetch alphavantage|twelvedata|marketstack

--fetch batch <provider> --symbols ... [--max-workers N]

--simulate asset --input ...

--simulate portfolio --inputs ... --weights ... [--components]

**Parámetros clave de simulación**
--days, --n-paths, --seed

--mc-method gbm|cholesky|copula|bootstrap (si usas las API desde Python)

--halflife-days, --block-len

## 🧠 Notas sobre Fechas y Proveedores
| Proveedor         | Control de fechas | Parámetros relevantes                      | Descripción                                                                        |
| ----------------- | ----------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| **Alpha Vantage** | ❌ (API)           | `--outputsize` = `compact` (100d) o `full` | La API no acepta `start/end`. El programa puede filtrar localmente tras descargar. |
| **TwelveData**    | ✅                 | `--interval`, `--start`, `--end`           | Intervalos 1min…1day. Ideal para rangos definidos.                                 |
| **MarketStack**   | ✅                 | `--start`, `--end`                         | Precios diarios en el rango indicado.                                              |

**NOTA**: Para Alpha Vantage, usa --outputsize full si quieres rango completo y, si lo deseas, aplica filtro local con --start/--end (a nivel de programa).

## 📚 Ejemplos reproducibles
**1) Python puro (sin CLI): cargar, limpiar y stats**
from pathlib import Path
from finlab.models.candles import Candles
from finlab.models.portfolio import Portfolio

aapl = Candles.from_any(Path("data/twelvedata/AAPL/AAPL_1day.parquet")).clean(fill_method="ffill").to_business_days()
nvda = Candles.from_any(Path("data/twelvedata/NVDA/NVDA_1day.parquet")).clean(fill_method="ffill").to_business_days()

port = Portfolio(series={"AAPL": aapl, "NVDA": nvda}, weights={"AAPL": 0.6, "NVDA": 0.4}, initial_value=1.0)
print(port.stats())                 # {'mean': ..., 'std': ..., 'sharpe': ...}
print(port.max_correlation_warning())

**2) Monte Carlo (GBM) + bandas**
paths = port.simulate(days=252, n_paths=2000, seed=123, mc_method="gbm")
port.plot_simulation(paths, title="MC — GBM cartera 60/40")

**3) Reporte Markdown**
md = port.report(mc_days=252, mc_paths=1000)
print(md)  # o guarda en outputs/report.md

##**📊 Reportes y Visualizaciones**
**.report()** → Markdown con:

  -Rango efectivo de datos (intersección temporal).
  
  -Pesos, métricas anualizadas (μ, σ, Sharpe), drawdown.
  
  -Advertencias (histórico corto, NaNs, alta correlación).
  
  -Resumen Monte Carlo (esperado, p5–p95).

**.plots_report()** → PNGs:

  -Componentes normalizados (opcional log-y).
  
  -Histograma de retornos.
  
  -Matriz de correlaciones anotada.
  
  -Banda Monte Carlo (media y 5–95%).
  
  -Los archivos se guardan en outputs/ si indicas save_dir

  ##**🚀 Rendimiento & Paralelismo**
  La descarga batch usa **ThreadPoolExecutor**
  python -m finlab.cli fetch batch twelvedata \
  --symbols AAPL,NVDA,MSFT,GOOGL,META \
  --interval 1day --format parquet \
  --max-workers 8

Buenas prácticas:

Ajusta --max-workers según el proveedor y tus límites de API.

Alpha Vantage impone límites estrictos; el extractor incorpora sleep para respetarlos.

TwelveData/MarketStack toleran mejor el paralelismo moderado.

## ✅ Calidad, Pruebas y Estilo##
Estilo: black, isort, ruff.

Tests: pytest (tests unitarios para normalización y simuladores).

Pre-commit (recomendado):
  pip install pre-commit
  pre-commit install

## 🗺️ Roadmap##
 --Filtro local de --start/--end unificado para todos los proveedores (incl. Alpha Vantage tras full).

 --Barra de progreso (tqdm/Rich) en fetch batch.

 --Exportar reporte a HTML y PDF.

 --Perfilado de rendimiento (descarga y MC).

 --Dockerfile + docker-compose para “plug-n-play”.

 ##**🧾Licencia**##
 MIT License — libre uso académico y profesional.
 
