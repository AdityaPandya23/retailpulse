# Progress

## P1 Setup + Data — done (2026-11-05)
- Scaffold: git, pyproject (src/retailpulse, -e install), .venv (Py 3.12), config.yaml, .claude deny rules.
- Modules: config, contracts (paths + required cols), ingest, clean (+ missing-ID analysis), warehouse, pipeline (stage skip, `--force`, `--only`).
- Ingest: 1,067,371 rows (sheets 525,461 + 541,910), 2009-12-01 → 2011-12-09.
  Format: xlsx 45.6 MB / ~55–61 s load vs parquet 7.2 MB / 0.14 s (best of 3) → ~6× smaller, ~450× faster.
- Cleaning funnel: duplicates −34,335 → returns split −19,104 → qty/price ≤ 0 −6,019 → missing ID −228,488
  → **779,425 sales lines**, 5,878 customers, 36,969 invoices, 4,631 products, 41 countries.
- Description imputation filled 0 rows: all 4,382 raw missing Descriptions have price ≤ 0 and no ID,
  so step 3 already removes them. The step stays in as a safeguard (it is tested on the fixture).
- Missing Customer ID (22.7% of valid lines): the missingness is per invoice (100% of affected invoices are fully missing).
  Median qty 1 vs 6 (rank-biserial −0.56, large); median price 3.36 vs 1.95 (r 0.32, medium);
  UK 98.8% vs 89.9% (Cramér's V 0.14, small). All p ≈ 0 → not MCAR; consistent with MAR (guest checkouts).
  Dropped, so later results describe registered customers only.
- Warehouse (DuckDB views over parquet): revenue £17.37M (2009 £0.68M, 2010 £8.37M, 2011 £8.32M); returns −£1.46M.
- Tests: 14 pass (synthetic fixture). Ruff clean.

## Decisions
- Returns table keeps all "C" invoices (714 without ID); features must filter on customer_id.
- Description lookup is built from valid sales lines (junk like "?" on zero-price rows is excluded).
- Warehouse is in-memory views (portable, no copy); summary JSON lives in artifacts/quality/.

## Next: P2 Features + Stats
features.py (cutoff 2011-09-09, 90-day horizon, leakage test), stats tests, EDA figures, cube.py, mock artifacts.
