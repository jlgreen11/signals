# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
privately via **GitHub Security Advisories**:

1. Go to https://github.com/jlgreen11/signals/security/advisories/new
2. Fill in the advisory form with:
   - A clear description of the vulnerability
   - Steps to reproduce (if possible)
   - Potential impact
   - Any suggested mitigations
3. Submit the draft advisory

**Please do not open a public issue for security vulnerabilities.**

I will respond to the advisory on a best-effort basis. This is a personal
research project, not a commercial product with guaranteed response times —
please be patient.

## Scope

This project is **experimental research software**. It fetches market data,
runs backtests, and generates trading signals. The main security concerns are:

- **Dependency vulnerabilities**: the Python dependency tree (`pandas`,
  `yfinance`, `hmmlearn`, etc.) is the most likely source of actual
  vulnerabilities. Please report issues affecting how `signals` uses those
  dependencies, not the upstream dependencies themselves (report those to
  their maintainers directly).
- **Model file loading**: the `.save()` / `.load()` methods for HMM models
  use `pickle`, which can execute arbitrary code if loading an untrusted
  file. Do not load model files from untrusted sources.
- **Data fetching**: the project fetches data from yfinance and (optionally)
  CoinGecko. Network issues or upstream API changes may cause unexpected
  behavior but are not security vulnerabilities in the usual sense.

## Out of Scope

- **Financial losses from trading signals.** This project is research
  software with no warranty — see `README.md` for the full disclaimer. A
  bad trading recommendation is not a security vulnerability. If the model
  produces wrong results due to a bug, please open a regular issue rather
  than a security advisory.
- **Vulnerabilities in transitive dependencies** — report those upstream.
- **Suggestions to improve the model or backtest** — open a regular issue
  or look at `IMPROVEMENTS.md`.
