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

## P3 Models — done (2026-11-06)
- New: segment/ (cluster, baseline, validate), predict/ (dataset, classify, regress, backtest), baskets/rules; stages segment → predict → rules; models in artifacts/models/*.joblib (raw features in); metrics.json writes are lock-protected (parallel stages).
- Segments (7 log1p+scaled features; return_rate, non_uk_share excluded): **k = 3** by rule — all k 3–7 have every segment ≥ 3%, k = 3 has the best silhouette 0.283 and no k within 0.02 (DB 1.32, CH 2,335, subsample ARI 0.98).
  0 **Champions** 2,086 (40% customers, 88% revenue, 65% bought again) · 1 **New Customers** 1,346 (26%, 7%, 44%) · 2 **Lost / Hibernating** 1,846 (35%, 6%, 19%).
- Baseline RFM quartiles: K-Means separates better (silhouette 0.28 vs 0.17, DB 1.32 vs 1.92) but RFM groups predict returning slightly better (CV AUC 0.745 vs 0.718).
  PCA: PC1–3 explain 52/21/14% (87%). Segment × UK: χ² 26, p 2e-6, Cramér's V 0.07 → significant but negligible.
- Prediction (out-of-time): train snapshot 2011-06-11 (4,976 customers, 32.0% returned) → test at cutoff (5,278, 43.2%; Nov peak only in test).
  CV AUC / test AUC: logistic 0.822/0.796, lda 0.818/0.793, qda 0.805/0.780, NB 0.811/0.792, **knn (k 76) 0.823/0.791** = best by train CV.
  Paired t knn vs logistic: +0.001, p 0.46 → no real difference (logistic is slightly better on test). Threshold 0.37 (max-F1 on train OOF).
- Backtest: we predicted 1,876 would buy again; 73% did vs base rate 43% (lift 1.68, recall 60%); top decile 90%, bottom 10%.
- Spend regression (returners, log1p): multiple_lr best (test RMSE 0.83, R² 0.42; mean-only 1.10; £-RMSE 3,765). Expected spend = p_return × expm1(pred).
- Rules: 200 overall + 464 segment rules (support floor 0.01, conf ≥ 0.3, lift > 1). Top: POPPY'S PLAYHOUSE BEDROOM → LIVINGROOM (lift 54.9, conf 71%, χ² 17,467, p < 0.001).
  Recommendations (5 each for all 5,278): hit-rate@5 25.3% vs 16.1% for best sellers (2,194 customers with a new purchase).
- Tests: 112 pass (47 new: segment 16, predict 14, baskets 12, metrics lock 4, wiring 1). Ruff clean.

## P3 decisions (user delegated k + names for this phase)
- k: smallest segment ≥ 3%, max silhouette in 3–7, ties within 0.02 → higher ARI → smaller k. Names: rule on segment z-scores (|z| > 0.5), unique.
  **Override** in config.yaml: `segment.k: 4` and/or `segment.names: {0: "...", 1: "..."}` (ids = revenue rank), then `python -m retailpulse.pipeline` (segment + rules rerun).
- Model selection only on train CV, never test; KNN k chosen on the same folds → its CV AUC is slightly optimistic (noted in metrics).
- Rules scopes: "overall", "segment:<name>"; recommendation = segment rules → overall rules (Σ conf × lift) → segment best sellers; never already-bought items.
- ROC/PR curves keep sklearn's inf end thresholds. Mock rule scopes now use the same names.

## P4 App — done (2026-11-06)
- P3 review first (744fa8b): no leakage / test-set tuning found; fixed 2 edge cases (config names now win over a
  matching automatic name; k_candidates "chosen" = k actually used under a config override). Key numbers unchanged.
- New stage app_data (after rules): customers/timeline.parquet (24,778 customer-months; "after" = label window only)
  + rec_reasons.parquet (rule behind each "rule" rec); 220 + 143 KB. Mock set now covers EVERY app input (174 KB).
- App: Home (KPIs, questions A/B/C) + 4 pages — 1 Data & Cube (Sankey, missing-ID, formats, monthly, countries,
  Lorenz, cohorts, CUBE sunburst + slice/dice), 2 Segments & Stats (3D PCA, biplot/scree, k panels, profile cards,
  box plots, parallel lines, baseline, χ², Q-Q, Poisson vs NB, Pearson vs Spearman, live t vs U test),
  3 Predictions (Time Machine reveal, ROC/PR, model table + tie note, confusion, KNN k slider, regression),
  4 Customer 360 (timeline, segment badge, gauge + actual outcome, expected spend, recs + fired rule, what-if).
- Figures in src/retailpulse/viz/ (palette, common, data_cube, segments, predictions, customer); pages only lay out.
- Real artifacts (AppTest): no exceptions, no mock badge; warm render 0.2–1.1 s per page (cold ≤ 2.3 s).
  `streamlit run` headless: /_stcore/health ok. App needs 3.13 MB of artifacts (33 files, none > 5 MB).
- Tests: 215 pass (+101: app_data, viz helpers, 8 figure/AppTest files incl. widget interaction). Ruff clean.

## P4 decisions
- app/components/data.py is the only reader: mtime-keyed caches; per-file fallback to artifacts/mock + per-section
  metrics fallback, both listed in a sidebar "Mock data" badge; RETAILPULSE_ARTIFACTS overrides the root (tests).
- Okabe-Ito, fixed colour per segment id + per model (models also get a dash: qda/NB pair is 7.6 ΔE deutan);
  yellow (slot 6) too light → names always shown, dark marker ring. Template set on every figure + theme=None
  (Streamlit overrides plotly's default). k / names / best model never hard-coded (metrics.json).
- Cube: grouping sets chosen by grouping_id; distinct customers never summed. Default customer = typical segment-0
  member (15288). What-if baseline = live models on the real features (delta starts at 0).
- Two subagents built pages 1–2 / 3–4 in parallel after the shared foundation; I integrated and reviewed.

## P5 Polish + Ship — done (2026-11-06)
- Review (62b44db): checklist + /code-review high of f26540e..HEAD. Rules held (single reader, no hard-coded k /
  names / model, seed 42); Time Machine = metrics.backtest exactly (1,876 / 73% / lift 1.68), train thresholds,
  cube grouping_id + no customer sums, live models reproduce test predictions. Fixed 5 (each + regression test):
  What-if sliders reset to their minimum after a page switch; sunburst parents now summed from children (penny
  rounding could blank the chart); mock stage reruns on config.yaml; null metrics show "n/a" (was a crash);
  horizon labels from config. Skipped 5 (helper duplication, micro-perf: renders already 0.7–2 s).
- Visual (104aa13): no browser tool → every figure rendered on real data via AppTest capture + kaleido (scratch
  only) at 1280 / 1920 px widths. Fixed label collisions (missing-ID, Pareto, biplot), crowded log ticks, clipped
  3D PCA title, squeezed parallel plot, legends over data (hist, ROC, PR), timeline legend order. Tour 1..n ok.
- Deploy (d5548d3): requirements.txt exact pins + `-e .` (plain `.` would move config.ROOT into site-packages);
  .gitignore block generated from contracts.APP_FILES (`python -m retailpulse.deploy`); 33 app artifacts
  tracked = 3.13 MB (+ mock 0.17 MB); tests/test_deploy.py guards it; CI = GitHub Actions (ruff + pytest, 3.12).
- Clean clone (scratch venv, no data/): mock_files() == [], 3 models load, Home + 4 pages no exception, headless
  /_stcore/health ok, ruff + 238 tests (+23) pass; local pipeline rerun skips every stage.
- Docs: README.md (A/B/C numbers, run, layout, credit, limits); DEMO_SCRIPT.md (~8 min, tour order, live numbers, Q&A).

## P5 decisions
- Pins = the .venv versions (sklearn 1.9.1, numpy 2.5.4 also pinned: pickled models); change only with a rerun.
- Customer timeline keeps its log axis (PLAN rule "log where skewed"); demo avoids extreme single-slider what-ifs.

## Open items (human)
1. Confirm / override `segment.k` and `segment.names` in config.yaml → `python -m retailpulse.pipeline` → commit artifacts.
2. Deploy on share.streamlit.io (repo AdityaPandya23/retailpulse, main, app/Home.py, Python 3.12) → URL into README.
3. Open the live app (no "Mock data" badge) and rehearse DEMO_SCRIPT.md once.
