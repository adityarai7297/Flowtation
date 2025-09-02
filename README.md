# SectorFlux

Automated ETF sector price data pipeline that fetches the last 5 years of sector prices and publishes them as GitHub Release assets.

## Overview

- **Scheduler/Runner**: GitHub Actions (cron, free for public repos)
- **Storage/Distribution**: GitHub Releases (public download)
- **Fetcher**: Python + yfinance
- **Formats**: Parquet (analytics-friendly), CSV.gz (browser-friendly)
- **Stability**: Always overwrites a release tagged `data-latest` for stable URLs

## Release URLs (after first run)

```
https://github.com/<owner>/<repo>/releases/download/data-latest/prices_5y.parquet
https://github.com/<owner>/<repo>/releases/download/data-latest/prices_5y.csv.gz
https://github.com/<owner>/<repo>/releases/download/data-latest/metadata.json
```

## Setup

1. Create a public GitHub repository
2. Push this code to your repository
3. Enable GitHub Actions in repository settings
4. Trigger the workflow manually: Actions → "Build & Publish Dataset (free)" → "Run workflow"

## What it does

- Runs nightly (weekdays) for free
- Rebuilds the 5-year snapshot from scratch (few MB)
- Publishes/overwrites a release tagged `data-latest` for stable download URLs
- Fetches data for: XLK, XLY, XLP, XLF, XLV, XLI, XLB, XLE, XLRE, XLU, XLC, SPY

## Local testing

```bash
pip install -r requirements.txt
python scripts/build_dataset.py
```

## Acceptance checklist

- [ ] Workflow runs green on manual trigger (Actions → "Run workflow")
- [ ] Release `data-latest` appears with 3 assets: prices_5y.parquet, prices_5y.csv.gz, metadata.json
- [ ] Download URLs work unauthenticated
- [ ] metadata.json has the current date_max
- [ ] File sizes are small (<10–20 MB total)
