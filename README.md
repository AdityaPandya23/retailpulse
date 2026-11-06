# RetailPulse

Customer analytics on two years of a UK online gift retailer's sales (UCI Online Retail II,
Dec 2009 – Dec 2011): **segment → predict → recommend**. Academic Data Science Foundations
project (NIT Warangal), built only with free, open-source tools.

**Live demo:** _deploy link goes here_ (Streamlit Community Cloud)

## What it answers

The cleaned data has 779,425 sales lines from 5,878 registered customers, £17.4M revenue.
Customers are modelled at a cutoff of 2011-09-09, and the outcome is whether they buy again
in the next 90 days.

| Question | Method | Headline result |
|---|---|---|
| **A. Who are our customers?** | Extended RFM features, K-Means vs an RFM-quartile baseline, PCA, stability, hypothesis tests | 3 segments (silhouette 0.28 vs 0.17 for RFM quartiles, stability ARI 0.98). The most valuable segment is 40% of customers and 88% of revenue. Revenue is concentrated: Gini 0.74, top 20% of customers = 77% of revenue. |
| **B. Who will come back?** | Logistic, LDA, QDA, Naive Bayes, KNN on an out-of-time split; spend regression; backtest | Test ROC-AUC 0.78–0.80 for all five models, which are practically tied (paired t p = 0.46). Backtest: of the 1,876 customers predicted to return, 73% did, vs a 43% base rate (lift 1.68). |
| **C. What do they buy together?** | Association rules (support, confidence, lift) overall and per segment → top-5 recommendations | 200 overall + 464 segment rules. Recommendations caught a newly bought product for 25.3% of customers, vs 16.1% for plain best sellers. |

All numbers come from `artifacts/metrics.json`. The app reads them live, and each page
explains its own numbers in plain English.

## Run it locally (Windows; use `source .venv/bin/activate` on macOS / Linux)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt pytest ruff   # exact versions; installs retailpulse (-e .)

streamlit run app/Home.py                     # the app, on the committed artifacts
python -m retailpulse.pipeline                # rebuild: downloads the data once, then runs every stage
python -m retailpulse.pipeline --force        # rebuild every stage from scratch
pytest -q                                     # tests (synthetic fixture + artifacts/mock only)
ruff check . ; ruff format .
```

The app works straight after cloning because the small app inputs (about 3 MB, listed in
`contracts.APP_FILES`) are committed. If an input is missing, the app falls back to tiny
synthetic files in `artifacts/mock/` and shows a "Mock data" badge.

## Project layout

```
config.yaml            cutoff, horizon, seed, model settings, optional segment k / names override
src/retailpulse/       all logic: ingest, clean, warehouse (DuckDB), features, cube, stats/,
                       segment/, predict/, baskets/, app_data, viz/ (Plotly figures), pipeline
app/Home.py, pages/    Streamlit app: 1 Data & Cube, 2 Segments & Stats, 3 Predictions,
                       4 Customer 360 (pages only lay out; app/components/data.py reads artifacts)
artifacts/             pipeline outputs; only the app inputs + artifacts/mock/ are in git
data/                  raw + cleaned data (downloaded by the pipeline, never committed)
tests/                 pytest on a synthetic fixture, never the real dataset
PLAN.md, PROGRESS.md   design and build log;  DEMO_SCRIPT.md: 8-minute walkthrough
```

## Data and credit

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) by Daqing Chen
(2012), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The cleaned
data and derived artifacts in this repository are adapted from it.

## Limitations

- **Registered customers only:** 22.7% of valid lines have no Customer ID (guest checkouts).
  They are not missing at random, so the results do not describe guest buyers.
- **One cutoff:** the models are trained and tested on one time split (train snapshot
  2011-06-11, test snapshot 2011-09-09). Other periods may behave differently.
- **Seasonality:** the November peak falls only in the test window, so the training data never
  saw it.
- **Expected spend is a typical-customer estimate:** P(return) × the back-transformed log-scale
  prediction. That is closer to the median than the mean, so totals are underestimated.
- Segment count and names come from an automatic rule and can be overridden in `config.yaml`.
