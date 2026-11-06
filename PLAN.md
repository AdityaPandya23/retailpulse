# RetailPulse — Project Plan (source of truth, v2 simplified)

Customer behaviour analytics on UCI Online Retail II: **segment → predict → recommend**.
Extends the original proposal (docs/proposal.pdf: cleaning → RFM → K-Means → profiles → app).
Zero cost: only free/open-source tools; no paid APIs.

**Versions.** `v1-full` (git tag) = the full analysis app (4 pages: OLAP cube, stats tests, 5 classifiers incl.
LDA/QDA, spend regression, per-segment rules, Customer 360, What-if). **v2 (current, P6)** = an easy-to-explain demo:
6 features, 3 classifiers, overall rules, a 6-page "hub and spokes" app. The v1 backend code (stats, cube,
regression, LDA/QDA, per-segment rules) stays in src/ and keeps running in the pipeline; v2 just does not show it.

## Questions the project answers
- **A. Who are our customers?** K-Means on 6 features → segments with automatic k and names.
- **B. Will they come back?** "Buys again in the next 90 days" with logistic, KNN and Naive Bayes on a time-based
  split (no leakage), backtested against what actually happened.
- **C. What should we offer them?** Overall association rules → top-5 products for a basket / a customer.

## Data + cleaning (done in P1, unchanged)
- UCI Online Retail II (id 502), CC BY 4.0, 1,067,371 rows, 2009-12-01 → 2011-12-09, two xlsx sheets → Parquet.
- Steps: drop exact duplicates → split "C" invoices into returns → drop qty/price ≤ 0 → drop missing Customer ID
  (22.7%, not MCAR; guest checkouts) → impute Description. Result: 779,425 lines, 5,878 customers. Line value = qty × price.
- DuckDB warehouse views; CUBE table (kept for v1, not shown in v2).

## Features (snapshot at cutoff 2011-09-09, horizon 90 days; ONLY data before the cutoff)
- features_cutoff has 9 columns; **v2 models use 6** (config.yaml segment.features = predict.features):
  recency, frequency, monetary, tenure, avg_basket_value, n_distinct_products.
  Kept but unused by models: mean_interpurchase_days, return_rate, non_uk_share.
- log1p on skewed features (frequency, monetary, avg_basket_value, n_distinct_products) + StandardScaler inside each
  pipeline (models take RAW values). Labels: returned_flag, next_spend. Leakage tests required.

## A — Segmentation
- K-Means, k scanned 2–10 (WCSS, silhouette, DB, CH); **k and names chosen automatically** (user decision P6a):
  k = smallest segment ≥ 3%, best silhouette in 3–7, ties within 0.02 → higher subsample ARI → smaller k;
  names from segment z-scores (cluster.name_one). config.yaml segment.k / names can override; no TODO(human).
- Segment ids 0..k-1 = revenue rank → fixed colour per id. PCA(3) stored in segment_model.joblib
  (`cluster.pca_coords` places a new customer on the scatter). presets.parquet = per-segment medians of the 6.

## B — Prediction
- Train snapshot = cutoff − 90 days (labels end at the cutoff), test = the real cutoff window (out-of-time).
- Classifiers (config.yaml predict.models; contracts.CLASSIFIERS ⊂ ALL_CLASSIFIERS): logistic, knn, naive_bayes.
  5-fold CV on train; best = max mean CV ROC-AUC (test never used for choices); threshold = max F1 on train OOF.
- KNN k scanned 1..100 (predict.knn_k_max = 100) → bias-variance curve (train vs test error).
- Backtest: "we predicted X would return; Y% did vs the base rate" + deciles. Regression stays hidden.
- classifier.joblib = best model; classifiers.joblib = all 3 pipelines + thresholds.

## C — Baskets
- Invoice × product matrix (top 500 products), apriori + association_rules, **overall scope only**
  (rules.per_segment = false). products.parquet = the 500 products (every rule item) with invoice counts.
- `rules.recommend_from_basket(codes, rules, best_sellers)`: rule score Σ confidence × lift, reason = strongest fired
  rule, best-seller fill, never an item in the basket. Per-customer recs + hit-rate@5 vs best sellers.

## Architecture
Offline pipeline writes artifacts → Streamlit app only reads artifacts (app/components/data.py: table / json_file /
metrics / model, paths from contracts.py, fallback to artifacts/mock/ with a "Mock data" badge).
```
src/retailpulse/  config contracts pipeline ingest clean warehouse features cube eda mock app_data deploy
                  stats/ segment/ predict/ baskets/ viz/ (Plotly figure functions)
app/Home.py  app/pages/2_Data.py 3_Segmentation.py 4_Prediction.py 5_Recommendation.py 6_Insights.py
app/components/{data,layout,tour}.py
artifacts/  (app inputs = contracts.APP_FILES, tracked via the generated .gitignore block; mock/ tracked)
```
Stages: ingest → clean → warehouse → features → stats → cube → eda → mock → segment → predict → rules → app_data
(`--only <stage>` forces a rerun; plain runs skip up-to-date stages).

## App v2 — 6 pages, "hub and spokes"
1. **Home = live demo (hub):** problem line, 4 KPI tiles, pipeline diagram (graphviz). Inputs: "Typical <segment>"
   preset buttons, 6 number inputs, product multiselect. Cards: (a) segment badge + 2D PCA scatter with the new
   customer as ★, (b) P(return) of the 3 models with thresholds + verdict, (c) top-5 basket recommendations with
   "because you bought X". Each card links to its explainer page.
2. **Data:** dataset card + column meanings, sample rows, cleaning Sankey, monthly trend, countries, Pareto.
3. **Segmentation:** the 6 features, elbow + silhouette (chosen k marked), PCA 2D/3D + scree + PC meanings,
   profile cards, customer % vs revenue % bar, box plots by segment.
4. **Prediction:** out-of-time split, model table + ROC (3 models), confusion matrix of the chosen model,
   Time Machine reveal, simple KNN bias-variance chart (k 1..100, log axis, select_slider).
5. **Recommendation:** support / confidence / lift on the top rule, top-rules table, support-vs-confidence bubble,
   product picker → its rules, hit-rate bar (rules vs best sellers).
6. **Insights:** one row per segment (customers %, revenue %, came back %, mean P(return), top-3 recommended
   products, suggested action) + the 4 limitations from README.md.
- Not shown in v2: LDA/QDA, regression / expected spend, PR curves, RFM baseline, ARI, biplot, parallel coordinates,
  OLAP cube, missing-ID / format / cohort charts, stats tests, Customer 360 timeline, What-if.
- Okabe-Ito palette, fixed colour per segment / model, log axes where skewed; never hard-code k, names, best model.
- AppTest: run Home, then `at.switch_page("pages/<file>.py")` — st.page_link only resolves with Home as entrypoint.

## Build phases
| Phase | Scope | Status |
|---|---|---|
| P1–P5 | data, features + stats, models, 4-page app, polish + deploy prep (= tag v1-full) | done |
| P6a | simplified backend: 6 features, 3 models, overall rules, demo artifacts (+ review fixes) | done |
| P6b | 6-page hub-and-spokes app, smoke tests, README app list, DEMO_SCRIPT rewrite | next |

## Remaining human TODOs
1. Deploy on share.streamlit.io: repo AdityaPandya23/retailpulse, branch main, main file app/Home.py,
   Python 3.12; paste the URL into README.md ("Live demo").
2. Open the live app, confirm there is no "Mock data" badge, rehearse DEMO_SCRIPT.md once (after P6b rewrites it).
