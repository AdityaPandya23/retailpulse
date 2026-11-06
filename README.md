# RetailPulse

Customer analytics on two years of a UK online gift retailer's sales (UCI Online Retail II,
Dec 2009 – Dec 2011): **segment → predict → recommend**. Academic Data Science Foundations
project (NIT Warangal), built only with free, open-source tools.

**Live demo:** _deploy link goes here_ (Streamlit Community Cloud)

This is **v2**, a simplified, easy-to-explain demo. The full analysis app (OLAP cube, hypothesis
tests, five classifiers, spend regression, per-segment rules, Customer 360) is kept at the git tag
[`v1-full`](../../tree/v1-full).

## What it answers

The cleaned data has 779,425 sales lines from 5,878 registered customers, £17.4M revenue.
Customers are described at a cutoff of 2011-09-09, and the outcome is whether they buy again
in the next 90 days. Every model uses the same 6 features: recency, frequency, monetary,
tenure, average basket value and number of distinct products.

| Question | Method | Headline result |
|---|---|---|
| **A. Who are our customers?** | K-Means on the 6 features; k and segment names chosen by an automatic rule (silhouette, segment size, stability) | 3 segments: Champions (40% of customers, 88% of revenue), Regulars (31%, 7.5%), Lost / Hibernating (29%, 4.5%). Revenue is concentrated: Gini 0.74, top 20% of customers = 77% of revenue. |
| **B. Will they come back?** | Logistic regression, KNN and Naive Bayes on an out-of-time split, backtested | Test ROC-AUC 0.79–0.80; the models are practically tied (KNN vs logistic, paired t p = 0.30). Backtest: of the 1,942 customers predicted to return, 73% did, vs a 43% base rate (lift 1.68). |
| **C. What should we offer them?** | Association rules (support, confidence, lift) → top-5 products not yet bought | 200 rules. Recommendations caught a newly bought product for 22.4% of customers, vs 16.1% for plain best sellers. |

All numbers come from `artifacts/metrics.json`. The app reads them live, and each page
explains its own numbers in plain English.

## The app: 6 pages, "hub and spokes"

The Home page is a live demo. Each of its three answer cards links to a page that explains how
that answer is made. A **Guided Tour** toggle in the sidebar adds numbered notes in demo order.

1. **Home (live demo):** the problem in one line, 4 headline tiles (clean sales lines, customers,
   revenue, date range) and the pipeline diagram. Click a "Typical …" button (one per segment)
   or type the 6 numbers, and add products to a basket. Three cards answer:
   **who is this customer?** (segment badge and a ★ on the PCA map of real customers),
   **will they come back?** (P(return) from all 3 models with their thresholds, and a verdict from
   the chosen model), and **what to offer** (top 5 products, each "because you bought X
   (confidence, lift)" or "best seller").
2. **Data:** dataset card (source, licence, raw rows, dates), the 8 raw columns, 5 sample rows,
   the cleaning funnel, monthly sales, revenue concentration (Pareto) and countries.
3. **Segmentation:** the 6 features, choosing k (elbow + silhouette, chosen k marked), the PCA map
   (2D or 3D) with a scree chart and what each axis means, segment profile cards, customers % vs
   revenue % per segment, box plots by segment.
4. **Prediction:** the out-of-time split, model table and ROC curves for the 3 models, confusion
   matrix of the chosen model, the Time Machine reveal (what really happened), and KNN train vs
   test error for k = 1 … 100.
5. **Recommendation:** support, confidence and lift explained on the strongest rule, the top
   rules, every rule as a support-vs-confidence bubble, a product picker showing its rules, and
   the hit-rate of rules vs best sellers.
6. **Insights:** one row per segment (customers %, revenue %, came back %, mean predicted
   P(return), top 3 recommended products, suggested action) and the limitations below.

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
config.yaml            cutoff, horizon, seed, the 6 features, the 3 models, rule settings,
                       optional segment k / names override
src/retailpulse/       all logic: ingest, clean, warehouse (DuckDB), features, eda, segment/,
                       predict/, baskets/, viz/ (Plotly figures), pipeline (v1 extras: stats/, cube)
app/Home.py, pages/    Streamlit app (pages only lay out; app/components/data.py reads artifacts)
artifacts/             pipeline outputs; only the app inputs + artifacts/mock/ are in git
data/                  raw + cleaned data (downloaded by the pipeline, never committed)
tests/                 pytest on a synthetic fixture, never the real dataset
PLAN.md, PROGRESS.md   design and build log;  DEMO_SCRIPT.md: demo walkthrough
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
  saw it (32% of customers came back in the train window vs 43% in the test window).
- **Recommendations show co-purchase, not cause:** rules say which products are bought together,
  and they were checked only against what customers bought next (hit-rate), never in a live test.
