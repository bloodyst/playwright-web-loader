# Playwright Web Page Loader (PoC)

Proof-of-concept for resilient web page loading using Playwright with
proxy support, retries, rate limiting and detailed reporting.

## Features
- Playwright (Chromium)
- HTTP / SOCKS5 proxy support
- Per-request proxy rotation
- Configurable concurrency and delays
- Retry logic with exponential backoff
- Detection of potential anti-bot / challenge pages
- CSV and JSONL reports
- HTML and screenshot artifacts for failed pages

## Use cases
- Pre-validation of web automation pipelines
- Testing proxy quality against real websites
- Batch loading of product pages
- Diagnostics of anti-bot behavior

## Installation
```bash
pip install -r requirements.txt
playwright install chromium
```

## Configuration
Copy example config:
```bash
cp config.yaml.example config.yaml
```

Edit ```config.yaml```:
- concurrency
- delays
- proxy list
- timeouts

## Run
```bash
python run_poc.py
```

## Output
- ```out/report.csv``` - summary report
- ```out/report.jsonl``` - detailed per-request results
- ```out/html/``` - saved HTML for failed pages
- ```out/screens/``` - screenshots for diagnostics

## Notes

This project does not attempt to bypass anti-bot protections.
It focuses on robustness, observability and infrastructure-level tuning.