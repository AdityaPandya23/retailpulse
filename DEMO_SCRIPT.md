# RetailPulse: demo script (~8 minutes)

Follows the Guided Tour numbers page by page. Turn on **Guided Tour** in the sidebar first; it
stays on across pages. All numbers below are the live values from `artifacts/metrics.json`
(cutoff 2011-09-09, 90-day horizon). If you rename the segments or change k in `config.yaml`, the
app shows your choice, so update the segment names and k in this script to match.

**Before you start:** open the live app about 5 minutes early, because a sleeping Streamlit Cloud
app needs a minute to wake. Click through all 4 pages once so the caches are warm. Keep
`streamlit run app/Home.py` ready locally as a backup. Check that the sidebar shows no
"Mock data" badge.

| Time | Where |
|---|---|
| 0:00 | Opening (30 s) |
| 0:30 | Home (30 s) |
| 1:00 | Page 1 Data & Cube (1.5 min) |
| 2:30 | Page 2 Segments & Stats (2 min) |
| 4:30 | Page 3 Predictions (1.5 min) |
| 6:00 | Page 4 Customer 360 (1.5 min) |
| 7:30 | Limitations close (30 s) |

---

## Opening (30 s)

> "RetailPulse looks at two years of sales from a UK online gift shop, about a million invoice
> lines. It answers three questions: **who are our customers, who will come back, and what do
> they buy together?** Everything runs on free, open-source tools. An offline pipeline builds
> the results, and this app only reads them."

## Home (30 s): Tour 1 → 2

- **Tour 1 · Start here.** Point at the first row of tiles: **£17.37M** revenue, **5,878**
  customers, **k = 3** segments, best model test ROC-AUC **0.791**.
  > "Each tile is one headline result. The pages behind questions A, B and C show how we got it."
- Don't dwell on the backtest tiles (1,876 / 73%). Say: *"I'll come back to these two on
  page 3."* That keeps the reveal for later.
- **Tour 2 · Demo route.** Click **Data & Cube** under question A.

*If slow:* the first page load on the cloud can take ~10 s. Keep talking through the opening.

## Page 1 · Data & Cube (1.5 min): A, can we trust the data?

**Syllabus:** data acquisition and formats, de-duplication, imputation, missing data
(MCAR/MAR), hypothesis tests (U, chi-square), histograms / frequency polygons, warehousing,
OLAP, cubes.

- **Tour 1 · Cleaning funnel** (Data quality tab). Follow the Sankey left to right.
  > "1,067,371 raw rows. We drop 34,335 duplicates, move 19,104 return lines to their own
  > table, drop 6,019 invalid quantities or prices, and the biggest loss is **228,488 lines
  > with no Customer ID**. That leaves **779,425 clean sales lines**."
- Point at the **missing-ID** chart:
  > "Lines without an ID are not random. Quantities are much smaller (r = −0.56, large), prices
  > higher (+0.32), and they ship to the UK more often (V = 0.14). So the IDs are **not missing
  > completely at random**. It looks like guest checkouts (MAR), and our results describe
  > registered customers only."
- Optional, one sentence: Parquet is **6.3× smaller** and loads **455× faster** than the Excel file.
- **Tour 2 · Seasonality** (Sales over time tab):
  > "Revenue peaks every November, at £1.2M in Nov 2010. Remember this for page 3."
- **Tour 3 · Pareto** (Customers & countries tab):
  > "Gini **0.74**: the top 20% of customers bring **77%** of revenue, and the top 1% bring 32%.
  > That's why we segment instead of using averages."
- **Tour 4 · OLAP cube** (OLAP cube tab). The default dice is the top 5 countries.
  > "One `GROUP BY CUBE` query gives all 8 grouping sets. Revenue and orders add up, but
  > **customers don't**: a customer who buys in two countries would be counted twice. So
  > distinct customers are always read from the rolled-up row, never summed."
  Point at the tile: **≤ 5,579** customers for 5 countries (an upper bound). Then pick only
  **United Kingdom** to get the exact **5,350**. Click *United Kingdom* in the sunburst to drill
  into years and months, then click the centre to go back.

*If slow:* skip the drill-down and open the "How CUBE works" expander. The "Σ customers ÷ true
total" column (4.36× at the finest level) makes the same point.

## Page 2 · Segments & Stats (2 min): A, who are they?

**Syllabus:** descriptive stats and quartiles, box plots, scatter, higher-dimensional displays,
PCA, distributions (lognormal, Poisson vs negative binomial), t / U / chi-square tests,
correlation and covariance, visual data mining.

- **Tour 1 · Segment profiles** (Segments tab):
  > "Three segments, ranked by revenue. **Champions**: 2,086 customers (40%) bring **88% of
  > revenue**, and 65% of them bought again. **New Customers**: 1,346 (26%), 7% of revenue.
  > **Lost / Hibernating**: 1,846 (35%), 6% of revenue, last purchase about a year before the
  > cutoff, and only 19% came back."
- **Tour 2 · PCA view:** drag the 3D plot once.
  > "Seven features squeezed into three principal components (52 + 21 + 14% = 87% of the
  > variance). The segments occupy different regions."
- **Tour 3 · Choosing k** (Choosing k & PCA tab):
  > "K-Means needs k up front. Our rule: every segment holds at least 3% of customers, pick the
  > best silhouette from k = 3–7, and break ties within 0.02. **k = 3** wins with silhouette
  > **0.283**, and it is very stable (ARI **0.98** over 80% subsamples). A human can override
  > k and the names in `config.yaml`."
- Optional (Validation tab), one sentence: K-Means separates better than the RFM-quartile
  baseline (silhouette 0.28 vs 0.17), but the simple RFM groups predict returning slightly
  better (AUC 0.745 vs 0.718).
- **Tour 4 · Live hypothesis test** (Live test tab): the defaults are Champions vs New
  Customers, on monetary.
  > "The app runs Welch's t-test and Mann-Whitney U live. Champions spend more: median £2,592
  > vs £512, p < 0.001, r = +0.90 (large), and both tests agree."
  Then switch to **New Customers vs Lost / Hibernating** on **recency**: median **94 vs 354
  days**, again a large effect.

*If slow:* the PCA scatter is the heaviest chart. Skip the rotation and go to the k panels.

## Page 3 · Predictions (1.5 min): B, who will come back?

**Syllabus:** logistic regression, LDA, QDA, Naive Bayes, KNN; ROC/PR, confusion matrix; model
accuracy and bias-variance; paired t-test; linear regression vs KNN regression.

- **Tour 1 · The prediction** (Time Machine tab):
  > "We stand at the cutoff, 9 Sep 2011. The model was trained on an earlier snapshot
  > (11 Jun 2011) and has never seen the future. It scores all **5,278** customers. Using a
  > threshold of **0.37** chosen on training data, we write down **1,876** customers we
  > expect to buy again in the next 90 days."
- Click **Reveal what actually happened**.
- **Tour 2 · The reveal:**
  > "**73%** of our list really came back, against a base rate of **43%**. That's a lift of
  > **1.68**, and the list holds 60% of everyone who returned. The top tenth came back **90%**
  > of the time, the bottom tenth only **10%**."
- **Tour 3 · Five models, one story** (Model comparison tab):
  > "All five models score 0.78–0.80 test ROC-AUC. KNN was chosen on training cross-validation,
  > and a paired t-test against logistic gives p = 0.46, so there is no real difference. The
  > signal is in the features, not the algorithm."
- **Tour 4 · Bias vs variance** (KNN bias-variance tab): drag the slider to **k = 1**, then
  back to **76**.
  > "With k = 1 the model memorises: train error 0%, test error high. Larger k smooths. At the
  > chosen k = 76, train error is 21.6% and test error 29.2%."
- Optional (Spend regression tab): multiple linear regression wins (test R² 0.42 vs 0 for the
  mean-only baseline).

*If slow:* the Reveal button reruns the page. If it hangs, open the Model comparison tab: its
confusion matrix shows the same 1,366 of 1,876 (73%).

## Page 4 · Customer 360 (1.5 min): C, one customer and what to offer

**Syllabus:** association rules (support, confidence, lift), independence (lift = P(A∩B) /
P(A)P(B)), KNN / logistic in use, regression.

- **Tour 1 · Pick a customer:** the default is **15288**, a typical Champion.
  > "Nine orders, £3,415 spent, last purchase 64 days before the cutoff."
- **Tour 2 · One-customer backtest:**
  > "At the cutoff the model gave **67%** (threshold 37%), so predicted to come back. They did:
  > one order of **£300** in the next 90 days. The prediction was right. Expected spend:
  > 67% × £490 typical spend if they return ≈ **£329**."
- **Tour 3 · Why this product?** Point at row 1:
  > "**Poppy's Playhouse Kitchen**, because they bought the Living Room. That Champions-segment
  > rule has confidence 90% and lift 54: buying one makes the other 54 times more likely than
  > chance. We never recommend something they already bought. Across all customers, the top 5
  > caught a new purchase for 25.3% of customers, vs 16.1% for best sellers."
- **Tour 4 · What-if:** drag **Recency** to **300**.
  > "Same customer, but silent for 300 days: P(buys again) drops from 67% to about **28%**."
  Then also set **Frequency 1**, **Monetary 400**, **Distinct products 20** and **Average
  basket 400**. The live K-Means now places them in **Lost / Hibernating**, at about 16%.
  Click **Reset**.
  Don't move Monetary alone to its maximum: the sliders change one feature at a time, and
  that combination is unrealistic.

*If slow:* skip the What-if and go to the close. The backtest card already tells the story.

## Close: limitations (30 s)

> "Four honest limits. **One:** registered customers only; guest checkouts were 23% of lines.
> **Two:** one cutoff, a single time split. **Three:** the November peak falls only in the test
> window. **Four:** expected spend is a typical-customer estimate, so it understates totals.
> Everything is reproducible: one pipeline command, 238 tests, and the code is on GitHub."

---

## Likely examiner questions

**Why k = 3?** A fixed rule decides it before looking at names: every segment must hold at
least 3% of customers, then the best silhouette from k = 3–7 wins. k = 3 has 0.283 and no
other k is within 0.02. It is also stable (ARI 0.98 over subsamples). The elbow is soft, so
we did not rely on it. A human can override k in `config.yaml`.

**Why an out-of-time split instead of a random split?** The real task is to predict the future
from the past. A random split would mix future behaviour into training. We train on a
June 2011 snapshot and test on the September cutoff, with features built only from data
before each cutoff. A leakage test checks that rows after the cutoff never change the features.

**Why do the models tie?** All five land at 0.78–0.80 test AUC, and the difference between
the best and logistic is not significant (paired t p = 0.46; corrected p = 0.61). The features
(recency, frequency, monetary) carry the signal, and the decision boundary is fairly simple. We
chose the model on training CV only, never on test, which is why the chosen KNN isn't the top
test model (logistic is, by 0.005).

**Why aren't customers summed in the cube?** Revenue and orders are additive: each invoice
belongs to one country and one month. Distinct customers are not: one customer active in 5
months appears in 5 month rows. Summing the finest level over-counts by 4.36×, so customer
totals are read from the rolled-up grouping set (grouping_id), and a multi-country slice shows
only an upper bound.

**MCAR vs MAR for the missing Customer IDs?** MCAR means missingness unrelated to anything.
We tested it: missing-ID lines have much smaller quantities (Mann-Whitney, r = −0.56), higher
prices (r = 0.32) and more UK shipping (chi-square, V = 0.14), so it is **not MCAR**. It is
consistent with **MAR** (guest checkouts, explained by observed variables). MNAR cannot be
ruled out from the data alone. Because those lines cannot be linked to a customer, we drop
them, and the results describe registered customers only.

**Why log1p and scaling?** Spend and frequency are heavily right-skewed (lognormal and
negative binomial fit far better than normal and Poisson). K-Means and KNN use distances, so
skewed, unscaled features would let one variable dominate.

**What does lift mean statistically?** Lift = P(A∩B) / (P(A)·P(B)). Lift 1 means A and B are
independent. Our top rules have lift > 50 and a chi-square test rejects independence
(p < 0.001).
