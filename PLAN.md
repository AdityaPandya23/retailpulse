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
