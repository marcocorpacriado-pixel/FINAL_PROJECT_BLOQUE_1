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





