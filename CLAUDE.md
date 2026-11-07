# RetailPulse

Customer segmentation, return prediction and basket analysis on UCI Online Retail II.
Academic DSF project (NIT Warangal). Design (v2, simplified): @PLAN.md. Progress log: PROGRESS.md.

## Hard rules
- NEVER read or print files in data/** or artifacts/**/*.parquet. Inspect data only through code
  that prints df.shape, df.info(), df.describe(), df.head(5) or small aggregates.
- All logic lives in src/retailpulse/. Notebooks and app/ only call functions from there.
- The app reads artifacts only through app/components/data.py (paths from contracts.py).
- Features use ONLY data before the cutoff in config.yaml. Never leak future data into features.
- Random seed 42 everywhere (config.SEED).
- Zero cost: only free/open-source libraries. No paid APIs or services.
- Plots: Plotly; functions return figures. Colour-blind-safe palette, fixed colour per segment.
- Tests use the synthetic fixture in tests/conftest.py, never the real dataset.
- k and segment names are chosen automatically by the metric rule in segment/cluster.py (user decision,
  P6a); never add TODO(human) for them. config.yaml segment.k / segment.names can still override.
- App v2.1 = 4 pages: Home (overview + pipeline + data), Demo, Segmentation, Prediction.
  Recommendation / Insights pages are hidden in app/hidden_pages/ (not shown). Old app: git tag v1-full.

## Commands (Windows, run from repo root)
- Activate env: .venv\Scripts\activate
- Pipeline: python -m retailpulse.pipeline            (add --force to rebuild all stages)
- Tests: pytest -q
- Lint/format: ruff check --fix . ; ruff format .
- App: streamlit run app/Home.py

## Working style
- Short answers; explain only when asked "explain".
- After finishing a phase: run tests + ruff, update PROGRESS.md (≤25 lines: done, key numbers,
  decisions, next step), then git commit with a clear message.
