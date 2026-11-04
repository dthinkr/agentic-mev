# AIDA: An AI Outage Registry with Evidence from On-Chain MEV

Replication code and analysis panels for the AI Incident and Disruption Archive (AIDA) and its application to on-chain maximal extractable value (MEV) on Ethereum and Base.

- **Dataset**: [huggingface.co/datasets/venvoo/aida](https://huggingface.co/datasets/venvoo/aida)

## Structure

- `data/incidents_v2/` — AIDA multi-vendor outage registry (`registry.csv`) and the 60m–24h foundation-model developer disruption subset (`registry_dev_60min_24h.csv`)
- `data/ethereum/` — Ethereum 22-month operator, executor, contract-creation, and hourly execution panels
- `data/base22/` — Base 22-month operator, executor, contract-creation, and hourly execution panels
- `data/competition/` — Prospective cohort and matched-calendar event-window panels
- `analysis/` — Registry construction, MEV transaction classification, and panel / event-study estimation scripts
- `results/` — Estimation outputs across the daily, hourly, and episode-window specifications

## Usage

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Rebuild the AIDA registry from provider records
.venv/bin/python analysis/build_registry.py

# Estimate hourly within-day PPML models on Ethereum and Base
.venv/bin/python analysis/chain_outage_hourly.py --registry-v2

# Estimate daily distributed-lag PPML models and episode-window rate ratios
.venv/bin/python analysis/outage_daily_dose_panel.py
.venv/bin/python analysis/outage_deployment_heterogeneity.py
```
