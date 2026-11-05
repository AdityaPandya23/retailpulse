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

## P2 Features + Stats — done (2026-11-06)
- New: features, stats/ (distributions, tests), cube, eda, mock; contracts.py has P2 + P3 contracts and metrics.json helpers; stages features → stats → cube → eda → mock.
- Features at cutoff 2011-09-09: 5,278 customers (full snapshot 5,878; 600 appear only later). Labels [2011-09-09, 2011-12-08): 43.2% bought again (2,281), median spend £533.
  Leakage test: changing, adding or dropping rows from the cutoff on leaves features_cutoff identical.
- Stats (one-line conclusions in metrics.json → "stats"):
  - Order value per customer is right-skewed (median £279, mean £385): lognormal fits far better than normal (AIC 79,139 vs 100,174).
  - Repeat orders are overdispersed (variance/mean 32): negative binomial far beats Poisson (AIC 31,036 vs 79,791), so customers differ a lot in how often they buy.
  - Days between purchase days (median 31, CV 1.33) are only roughly exponential: Weibull shape 0.85 fits better, i.e. more short gaps and a longer tail.
  - Repeat customers have bigger baskets than one-time ones (median £293 vs £229; Mann-Whitney p < 0.001, r 0.19 small); Welch agrees (p 0.033, d 0.05).
  - Y1 vs Y2 spend (2,713 customers in both): paired t not significant (−£186, p 0.071) but Wilcoxon p < 0.001: the typical customer spent less (median −£128, 41% spent more); a few big swings hide it from the mean.
  - Correlation: frequency–monetary strongest (Spearman 0.86 vs Pearson 0.63); biggest gap frequency–return_rate (0.43 vs 0.01): outliers drive Pearson, covariance depends on units → log1p + standardise.
- Cube: 1,038 rows = all 8 grouping sets of CUBE(country, year, month). EDA: 6 tables; Gini 0.74, top 20% of customers = 77% of revenue (top 1% = 32%), UK 83%, peak Nov 2010.
- Mock: artifacts/mock/ (13 parquet + metrics.json, 71 KB, IDs 900001+, "mock": true) for every P3 contract, plus features/labels so joins work.
- Tests: 59 pass (45 new: hand-computed features, leakage, stats on seeded samples, cube, EDA, mock). Ruff clean.

## P2 decisions
- return_rate and n_distinct_products use product codes only (5-digit prefix); "M" manual credits were a third of credit value. monetary stays gross (= P1 revenue); the 2 fully cancelled mega-orders (£168k, £77k) show as return_rate 1.
- mean_interpurchase_days uses distinct purchase days; customers seen on one day get the days since that purchase (a lower bound). Frequency fits use repeat orders (frequency − 1).
- log1p only where it moves skewness toward 0: frequency, monetary, avg_basket_value, mean_interpurchase_days, n_distinct_products (not recency 0.67 → −0.97, not tenure).
- Labels are stored apart from features. metrics.json has one section per stage; the pipeline skips a stage only if its section exists.
- .gitignore: `artifacts/*` instead of `artifacts/`, which silently blocked `!artifacts/mock/`. Y1/Y2 windows and alpha live in config.yaml.

## Next: P3 Models
3a segment/ ∥ 3b predict/ + baskets/ on features_cutoff + labels (log1p_skewed → StandardScaler, is_product for baskets); the human picks k and segment names.
