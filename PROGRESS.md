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

## P6a Simplified backend — done (2026-11-07)
- Backend only for an easy-to-explain demo (old app = tag v1-full; new app = P6b). config.yaml: segment + predict features =
  recency, frequency, monetary, tenure, avg_basket_value, n_distinct_products; predict.models = logistic, knn, naive_bayes;
  rules.per_segment = false. contracts: ALL_CLASSIFIERS (5, keeps palette colours) / CLASSIFIERS = the 3 configured.
- New: models/classifiers.joblib (3 pipelines + train thresholds); PCA(3) in segment_model.joblib (`cluster.pca_coords`
  places a new customer, = pc1–3 exactly); `rules.recommend_from_basket` (overall rules, reason = strongest fired rule,
  best-seller fill, never basket items); eda/sample_rows (5 seeded lines), rules/products (top 500 = every rule item),
  segments/presets (per-segment medians of the 6). All mocked; .gitignore regenerated: 37 app files, ≈3.0 MB.
- Rerun eda, mock, segment, predict, rules, app_data (`--only` already forced reruns: no pipeline change). Ingest untouched.
- Segments: **k = 3** by rule (silhouette 0.287, no k within 0.02; DB 1.29, ARI 0.99): 0 Champions 40% customers / 88%
  revenue · 1 Regulars 31% / 7.5% · 2 Lost / Hibernating 29% / 4.5%. RFM baseline silhouette 0.16, CV AUC 0.745 vs 0.716.
- CV / test AUC: logistic 0.819/0.795 · naive_bayes 0.805/0.790 · **knn (k 75) 0.822/0.794** = best by train CV, threshold
  0.35; knn vs logistic +0.003, p 0.30 → tied. Backtest: 1,942 predicted, 72.7% returned vs base 43.2% (lift 1.68,
  recall 62%; top decile 91%, bottom 11%). Hidden regression: multiple_lr, test RMSE 0.83.
- Rules: 200 overall (top POPPY'S PLAYHOUSE BEDROOM → LIVINGROOM, lift 54.9). Hit-rate@5 22.4% vs 16.1% best sellers
  (2,194 customers; 25.3% with segment rules). Tests: 245 pass (+7), leakage tests pass. Ruff clean.

## P6a decisions
- **k and segment names: chosen automatically by the existing metric rule (user decision 2026-11-07), no TODO(human).**
  config.yaml segment.k / segment.names can still override.
- Presets = per-feature medians (typical, not a real customer). Basket "popular" rows: score NaN, no reason. A basket has
  no segment → only "overall" rules. Rules stage keeps segments.parquet as input so per_segment: true still works.
- Mock segmentation uses segment.features (mock presets fit the mock models); mock rules are overall-only.
- Old app still passes its AppTests (3 models now).

## P6a review + docs for P6b (2026-11-07)
- /code-review high (9702d59, pushed): fixed 6 of 9 — best-vs-other comparison works without logistic, predict.models
  needs ≥ 2 models; app text counts models; summarize() writes the overall-only note; mock thresholds from mock
  metrics; sample_rows samples before copying; recommend_from_basket arg renamed best_sellers. Skipped 3: second
  PCA(3) fit (ms, tested equal), classifier.joblib duplicating classifiers.joblib (v1 Customer 360 still loads it),
  CLAUDE.md k/names rule (fixed below on the user's request).
- Docs moved to v2 (user request): CLAUDE.md (k/names automatic, app v2), PLAN.md rewritten for v2 (6-page app,
  phases, not-shown list), README.md (v2 numbers, page list, 4 limitations without expected spend), DEMO_SCRIPT.md
  marked "outdated (v1)" until P6b rewrites it. predict.knn_k_max stays 100 (KNN slider: 1, 5, 20, best k, 100).
- Checked (Streamlit 1.66.1): st.page_link works in AppTest only with Home.py as entrypoint → smoke tests open Home,
  then at.switch_page("pages/<file>.py"); running a page file directly cannot link back to "Home.py".
- Next: P6b = 6-page hub-and-spokes app (PLAN.md "App v2"), smoke tests, README page list, DEMO_SCRIPT rewrite.

## P6b Simplified 6-page app — done (2026-11-07)
- App v2 = hub and spokes: Home (live demo) + 2_Data, 3_Segmentation, 4_Prediction, 5_Recommendation, 6_Insights;
  the 4 v1 pages and their 2 test files deleted. Pages only lay out; each explainer links back to Home.
- New logic: `demo.py` (KPI facts, pipeline DOT, input specs, presets, score one customer, verdict, offer lines),
  `insights.py` (per-segment table, action rule, README limitations parser), `viz/rules.py`; figures added to
  viz/segments (2D PCA + ★, PC meanings, customer % vs revenue % bars; k panels take a metric subset) and
  viz/predictions (P(return) bars + thresholds, split timeline, simple model table, KNN log-axis chart).
- Real artifacts: k = 3; Typical Champions → knn 63% (threshold 35%, 3/3 agree), Regulars 20%, Lost 3%;
  basket POPPY'S PLAYHOUSE BEDROOM → Kitchen (conf 86%, lift 51.4), Livingroom (71%, 54.9) + 3 best sellers.
  KNN k 1: train 100% / test 67.8%; best k 75: 78.6% / 70.7%. Actions: Champions Reward and keep, Regulars
  Second-order offer, Lost / Hibernating Win-back offer.
- Checks: AppTest on mock + real (Home, every page, presets, reveal, KNN slider, 3D toggle, picker); headless
  `streamlit run`: /_stcore/health ok and all 6 pages rendered over the websocket on the real artifacts with
  0 exceptions and no Mock badge (0.8–2.1 s). Figures rendered with kaleido (scratch venv) and fixed: best-k
  line on the log axis, timeline axis title over the legend, clipped Home PCA title, share-bar legend.
- Tests: 275 pass (245 − 22 v1 app tests + 52 new: app smoke 16, demo 10, insights 9, viz v2 17). Ruff clean.
- Docs: README app section, DEMO_SCRIPT.md rewritten for v2 (~7 min, live numbers), PLAN P6b done.

## P6b decisions
- Default basket = the top rule's antecedents (metrics.json), so the first view shows a "because you bought" line.
- Home inputs live in session state (plain keys) and survive page switches; presets are cast to int / float per input.
- Pipeline diagram: rules start at Clean (mined from invoices, not customer features); DOT string, no graphviz pkg.
- Limitations are parsed from README.md "## Limitations" via data.limitations() (contracts.README_MD), same wording.
- Came back % = profiles.bought_again_share (fallback: test-split outcome); Avg P(return) = chosen model, test split.
- Action rule per segment in order, first match wins; ties go to the lowest segment id (v2 has no "Reward" tie).
- Next: human items below (push tag, deploy, rehearse DEMO_SCRIPT.md v2).

## P6b review fixes (2026-11-07)
- /code-review high of d2fd9b3: fixed all 9. verdict: a missing threshold (NaN in a float column) now falls back
  to 0.5; Time Machine guarded when the chosen model has no test rows (summary computed once); product names
  markdown-escaped in rule texts (md_escape moved to viz/common); PC lines show their variance share; horizon
  fallback = config via data.horizon_days(); overall_rules never falls back to segment rules (= recommender);
  page logic moved to src (rule tables in %, overall_invoices, rule_title, default_product, knn_best_k,
  product_options); insights reuses predictions.test_predictions.
- Tests: 286 pass (+11 regression tests). Ruff clean. Fresh headless server: 6/6 pages, 0 exceptions, no Mock
  badge (an older test server had kept port 8599; it is stopped and the check was rerun on a new one).

## P6c Tests, visual pass, demo script — done (2026-11-07)
- Tests: Home preset test (mock; Typical first vs last segment → badge + chosen model's P(return) both change);
  tests/test_artifact_consistency.py (20 test-split customers, seed 42: all 3 pipelines in classifiers.joblib on
  features_cutoff reproduce the stored p_return within 1e-9). 5 regression tests for the layout fixes below.
- Visual pass at 1280 px (AppTest capture on real artifacts → kaleido PNG, scratch venv): Streamlit 1.66 sidebar
  300 px + 5 rem padding → content 820 px, half 402, [3, 2] 482 / 322, Home cards ≈ 228. Fixed in viz/: Home PCA
  title clipped; P(return) bars (rotated ticks, legend over axis title → 3 ticks, no legend); Pareto legend over
  the "top 1%" label; scree legend beside the title; share-bar legend swatches looked the same (rgba, not
  opacity); box plots (legend dropped, wrapped names, decade ticks); 3D PCA cropped (legend on top, camera
  further); KNN x-range jumped with the slider (fixed range, round ticks, k = 1 dot no longer clipped); bubble title.
- Headless `streamlit run` on port 8611: health ok, 6/6 pages over the websocket, 0 exceptions, no Mock badge
  (Home cold 13.9 s, others 0.6–1.5 s); stopped only that server (3 older streamlit processes left running).
- DEMO_SCRIPT.md rewritten for the hub flow (~8 min, checklist, Q&A); numbers read from the app + metrics.json.
  Examiner examples: (10, 12, £3000, 600, £250, 120) → Champions, KNN 75%; (150, 2, £300, 300, £150, 20) →
  Regulars, KNN 11%; both 3/3 agree. Typical Lost + recency 30 → Regulars, 11%.
- Tests: 293 pass (286 + 7). Ruff clean. Headline numbers unchanged (k = 3, KNN k 75, AUC 0.790–0.795, lift 1.68).

## P6c decisions
- Consistency test uses the TEST split only: train p_return are out-of-fold CV scores, not the refit pipelines.
- It reads committed artifacts via contracts (never data/), skips if a file is missing; README notes the exception.
- Demo checklist: collapse the sidebar after switching the tour on; at 110–125% zoom with it open, Home cards are
  ~190 px and chart titles clip. Examples kept away from the 35% threshold: (30, 1, £200, 30, £200, 15) gives
  exactly 35% with 1 of 3 models agreeing, so the script warns about borderline customers instead.
- README changed only in the test lines (no number or page description changed). Next: human items below.

## Open items (human)
1. Push the tag so README's v1 link works: `git push origin v1-full`.
2. Deploy on share.streamlit.io (repo AdityaPandya23/retailpulse, main, app/Home.py, Python 3.12) → URL into README.
3. Open the live app (no "Mock data" badge) and rehearse DEMO_SCRIPT.md (v2) once.
4. Optional after P6b: drop classifier.joblib (classifiers.joblib has the best model too) — contracts, stage, tests.
