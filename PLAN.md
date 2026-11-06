# RetailPulse — Project Plan (source of truth)

Customer behaviour analytics on UCI Online Retail II: **segment → predict → recommend**.
Extends the original proposal (docs/proposal.pdf: cleaning → RFM → K-Means → profiles → app).
One-day build. Zero cost: only free/open-source tools; no paid APIs.

## Questions the project answers
- **A. Who are our customers?** Segmentation with extended RFM features, K-Means vs a rule-based
  RFM-quartile baseline, validated statistically.
- **B. Who will come back?** Predict "buys again in the next 90 days" and "how much", using a
  time-based split (no leakage), and backtest against what actually happened.
- **C. What do they buy together?** Association rules per segment → product recommendations.

## Data
- UCI Online Retail II (id 502), CC BY 4.0, ~1.07M rows, 2009-12-01 → 2011-12-09, two xlsx sheets.
- Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.
- Download once → data/raw/ → convert to Parquet (log file sizes + load times: xlsx vs parquet).

## Cleaning (record rows removed per step → Sankey funnel in app)
1. Remove exact duplicate rows.
2. Split invoices starting with "C" into a separate returns table (used for return_rate feature).
3. Drop Quantity <= 0 or Price <= 0 from sales.
4. Missing Customer ID: compare these rows with the rest (Mann-Whitney U on Quantity/Price,
   chi-square on UK vs non-UK) and discuss MCAR/MAR, then drop from customer-level analysis.
   (Done in P1: 22.7% of lines, missing per whole invoice, not MCAR → results cover registered customers.)
5. Impute missing Description with the most common Description for that StockCode.
- Line value = Quantity × Price.

## Warehouse / OLAP (DuckDB over Parquet)
- Views: fact_sales, fact_returns, dim_customer, dim_product, dim_date, dim_country.
- Cube: GROUP BY CUBE(country, year, month) and segment roll-ups; drill-down / slice / dice in app.

## Features (snapshot at a cutoff; use ONLY data before cutoff)
- cutoff = 2011-09-09, horizon = 90 days (config.yaml).
- Per customer: recency, frequency (distinct invoices), monetary, tenure, avg_basket_value,
  mean_interpurchase_days, n_distinct_products, return_rate, non_uk_share.
- Labels (from the horizon window after cutoff): returned_flag (0/1), next_spend.
- log1p on skewed features, StandardScaler before distance-based models. Leakage test required.

## Part A — Segmentation + statistics
- K-Means, k = 2..10: silhouette, WCSS (elbow), Davies-Bouldin, Calinski-Harabasz.
- Baseline: RFM quartile scores (1-4 each) → compare on same metrics and on predicting returned_flag.
- Stability: Adjusted Rand Index over 10 random 80% subsamples.
- PCA: scree plot, 2D/3D coordinates, loadings (biplot).
- Profiles: size, mean features, revenue share, business name per segment (chosen by the human).
- Stats: distribution fits + Q-Q (order value ~ lognormal, purchases ~ Poisson vs negative binomial,
  inter-purchase gap ~ exponential); Welch t-test vs Mann-Whitney U (repeat vs one-time basket value);
  paired t-test (same customers' spend year 1 vs year 2); Pearson vs Spearman correlation +
  covariance; chi-square segment × UK/non-UK.

## Part B — Prediction
- Classifiers (scaled sklearn pipelines): Logistic Regression, LDA, QDA, Gaussian Naive Bayes, KNN.
  5-fold CV on train; evaluate on future window: ROC-AUC, PR, confusion matrix.
- Bias-variance: KNN k = 1..100 train vs test error curve.
- Paired t-test on per-fold AUCs (best model vs logistic).
- Regression on log1p(next_spend): simple LR (frequency), multiple LR, KNN regression → RMSE, R², residuals.
- Backtest: predictions at cutoff vs actual outcome ("we predicted X would return; Y% did").

## Part C — Baskets
- Invoice × product boolean matrix (top 500 products), mlxtend apriori + association_rules
  (support, confidence, lift; lift = P(A∩B)/(P(A)P(B)) → test of independence), overall + per segment.
- recommend(customer_id) → top 5 products not yet bought.

## Architecture
Offline pipeline writes artifacts → Streamlit app only reads artifacts.
```
src/retailpulse/
  config.py contracts.py pipeline.py        # contracts = artifact paths + required columns
  ingest.py clean.py warehouse.py features.py cube.py
  stats/ (distributions.py, tests.py)
  segment/ (cluster.py, baseline.py, validate.py)
  predict/ (dataset.py, classify.py, regress.py, backtest.py)
  baskets/rules.py
app/Home.py  app/pages/1_Data_and_Cube.py 2_Segments_and_Stats.py 3_Predictions.py 4_Customer_360.py
app/components/data.py   # single loader, st.cache_data, falls back to artifacts/mock/
artifacts/{features,segments,predictions,rules,cube,figures,mock}/  metrics.json
tests/   # synthetic fixture only, never real data
```
- pipeline.py runs stages in order and skips a stage whose outputs already exist (--force to redo).
- segment/ and predict/+baskets/ are independent after features exist → can be built in parallel.

## App (demo) — 4 pages + Guided Tour sidebar toggle (numbered callouts)
1. **Data & Cube**: cleaning Sankey funnel, missing-data findings, format comparison, OLAP sunburst
   with drill-down / slice.
2. **Segments & Stats**: 3D PCA scatter, profile cards, box plots, parallel coordinates, k-selection
   charts, live test picker (choose 2 segments + feature → t-test vs U-test, plain-English result).
3. **Predictions**: Time Machine (predicted vs actual reveal), ROC overlay of 5 models, confusion
   matrix, KNN-k bias-variance slider, regression comparison.
4. **Customer 360**: customer ID lookup → timeline, segment badge, return-probability gauge,
   recommendations; What-if sliders → live segment + probability.
- Colour-blind-safe palette, fixed colour per segment, log axes where skewed.
- Free deploy: Streamlit Community Cloud from GitHub (deploy small artifacts only, not raw data).

## Syllabus coverage (DSF)
Random variables & distributions · descriptive stats & quartiles · dependent/independent events (lift)
· hypothesis tests (t, paired t, U, chi-square) · correlation & covariance · association rules · PCA ·
linear regression vs KNN · logistic, LDA, QDA, Naive Bayes, KNN comparison · model accuracy &
bias-variance · histograms, frequency polygons, box plots, scatter, heat maps, higher-dim displays,
visual data mining · data acquisition, formats, imputation, split-apply-combine, de-duplication ·
warehousing, OLAP, cubes.

## Build phases (one day)
| Phase | Scope | Depends on |
|---|---|---|
| P1 Setup + Data | scaffold, venv, git, ingest, clean, missing-ID analysis, warehouse | — |
| P2 Features + Stats | features, labels, contracts, stats tests, EDA figures, cube, mock artifacts | P1 |
| P3 Models | 3a segment/ · 3b predict/ + baskets/ (parallel OK) | P2 |
| P4 App | 4 pages on real artifacts | P3 |
| P5 Polish + Ship | review, demo fixes, deploy, DEMO_SCRIPT.md | P4 |

## P2 detail (updated after P1 — see PROGRESS.md)
Already done in P1, do NOT redo: contracts.py (extend it), pipeline Stage pattern (register new stages),
missing-ID analysis, cleaning funnel, warehouse views, format comparison.

Inputs: data/processed/sales.parquet (779,425 lines, registered customers only) and returns.parquet
(contains 714 "C" lines with no customer_id → filter on customer_id before per-customer features).

1. **features.py**
   - `build_features(sales, returns, cutoff)` → one row per customer with ≥1 purchase before cutoff;
     uses ONLY rows with invoice_date < cutoff. Recency measured from cutoff.
   - `build_labels(sales, cutoff, horizon_days)` → returned_flag, next_spend from [cutoff, cutoff+90d).
     cutoff 2011-09-09 + 90d ends 2011-12-08, inside the data (ends 2011-12-09).
   - Two snapshots: **features_cutoff** (modelling: P3a segmentation AND P3b prediction both use this,
     so segment labels can be used as a feature without leakage) and **features_full**
     (cutoff = max date + 1 day; descriptive EDA only).
   - Leakage test: changing any row after cutoff must not change features_cutoff.
2. **stats/**
   - distributions.py: per-customer order value ~ lognormal, frequency ~ Poisson vs negative binomial
     (dispersion ratio), inter-purchase gap ~ exponential; save Q-Q points + fit params + AIC.
   - tests.py: repeat vs one-time avg basket value → Welch t vs Mann-Whitney U (+ effect sizes);
     paired t-test on customers active in both Y1 (2009-12-01→2010-11-30) and Y2 (2010-12-01→2011-11-30);
     Pearson vs Spearman correlation + covariance matrices of features_full.
   - Every result → metrics entry with a one-line plain-English conclusion.
3. **cube.py**: DuckDB `GROUP BY CUBE(country, year, month)` over the warehouse views → small parquet
   (revenue, orders, customers) for the OLAP page.
4. **EDA data**: small aggregate tables (not images) for histograms/frequency polygons, box plots,
   monthly sales/orders, country revenue, top-customer revenue share (Lorenz/Pareto), cohort retention
   matrix → artifacts/eda/.
5. **Mock artifacts**: artifacts/mock/ (tracked in git) with tiny fake files matching every P3 contract
   (segments, predictions, rules, model metrics) so P4 can start before P3 finishes.
6. Register stages `features`, `stats`, `cube`, `eda`, `mock` in pipeline.py; tests on the fixture.

## P4 detail (updated after P3 — see PROGRESS.md)
Real artifacts exist for every P3 contract; the app still reads only through app/components/data.py
(paths from contracts.py, fallback artifacts/mock/). Segment ids 0..k-1 = revenue rank → fixed colours;
k = 3 now but read k / names from metrics.json `segment` (overridable in config.yaml), never hard-code.

| Page | Artifacts | metrics.json |
|---|---|---|
| 1 Data & Cube | quality/*.json, cube/sales_cube, eda/monthly, countries | eda, features |
| 2 Segments & Stats | segments (pc1-3, r/f/m_score, rfm_group), profiles, k_selection (+ari_mean), pca_loadings, eda/boxplots, histograms, stats/* | segment (k_candidates, k_choice_reason, baseline, chi2_segment_uk), stats |
| 3 Predictions | classification (split train/test), roc_curves, pr_curves, knn_bias_variance, regression | classify, regress, backtest (deciles) |
| 4 Customer 360 | features_cutoff, labels, segments, classification (test, best model), recommendations (source rule/popular), rules | segment.segment_names |

- What-if: joblib dicts in artifacts/models/ take a RAW (un-logged) DataFrame of their `features`:
  segment_model {pipeline, features, label_to_segment, segment_names} → use segment.cluster.predict_segment;
  classifier {pipeline, features, model, threshold} → predict_proba[:, 1]; regressor {pipeline, features, model,
  target, note} → log1p(next_spend | returns).
- Expected 90-day spend = p_return × expm1(regressor prediction) (spend if they return × chance they return).
- Rules: scope "overall" or "segment:<name>"; antecedents/consequents are " + "-joined descriptions (codes in
  *_codes). ROC first / PR last threshold is inf → drop before plotting thresholds.
- Live test picker: reuse stats/tests.py functions on features_cutoff joined to segments.

## P5 detail (updated after P4 — see PROGRESS.md)
1. **Review**: `/code-review` of the P4 diff (app/, src/retailpulse/viz/, app_data, mock); click through every page
   in a browser (light theme), Guided Tour on, at 1280 px and 1920 px; fix label collisions / empty states.
2. **Deploy (Streamlit Community Cloud, free)**: entry point app/Home.py; requirements from pyproject (pin versions
   used locally: streamlit 1.66, plotly 7.1, scikit-learn 1.9, pandas 3.1 — joblib models are version-bound);
   `pip install -e .` equivalent (add `-e .` or `.` to requirements.txt so `retailpulse` imports).
   Track ONLY the app inputs = contracts.APP_FILES (3.13 MB, 33 files: features_cutoff, labels, segments/*,
   predictions/*, rules/*, customers/*, cube, eda/*, stats/*, quality/*.json, metrics.json, models/*.joblib) via
   explicit `!artifacts/<path>` lines in .gitignore; never data/ or features_full. Budget: < 10 MB tracked.
   Smoke: app shows no "Mock data" badge on the cloud.
3. **DEMO_SCRIPT.md** (~8 min, follows the Guided Tour numbers): Home tiles → A: Page 1 (Sankey, missing-ID MAR,
   cube drill-down + why customers aren't summed) → Page 2 (k evidence + override note, PCA, profiles, live test)
   → B: Page 3 (Time Machine: predict, then reveal 73% vs 43%; models tied; KNN slider) → C: Page 4 (customer
   15288: timeline, gauge vs actual, recs + fired rule, what-if) → close on limitations (registered customers
   only, one cutoff, Nov peak only in test, expected spend is a typical-customer estimate).
4. Human TODO: confirm or override k and segment names (config.yaml segment.k / segment.names) before the demo.
