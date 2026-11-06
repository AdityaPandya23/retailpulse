# RetailPulse: demo script (~7 minutes, app v2)

Live demo first, then the explainer pages, then the limitations. Turn on **Guided Tour** in the
sidebar: every page shows numbered notes in this order, and it stays on across pages. All
numbers below are the live values from `artifacts/metrics.json` (cutoff 2011-09-09, 90-day
horizon) and what the app showed on the committed artifacts. If you change k or the segment
names in `config.yaml`, the app follows; update the names and numbers here to match.

**Before you start:** open the live app about 5 minutes early (a sleeping Streamlit Cloud app
needs a minute to wake) and click through all 6 pages once so the caches are warm. Check that
the sidebar shows no "Mock data" badge. Keep `streamlit run app/Home.py` ready locally as a backup.

| Time | Where |
|---|---|
| 0:00 | Opening (20 s) |
| 0:20 | Home: live demo (2.5 min) |
| 2:50 | Data (30 s) |
| 3:20 | Segmentation (1 min) |
| 4:20 | Prediction (1.5 min) |
| 5:50 | Recommendation (45 s) |
| 6:35 | Insights + limitations (45 s) |

---

## Opening (20 s)

> "RetailPulse uses two years of invoices from a UK online gift shop to answer three questions:
> **who are our customers, will they come back, and what should we offer them?** It uses only
> free, open-source tools. An offline pipeline builds the models, and this app only reads them."

## Home: live demo (2.5 min)

- **Tour 1 · The data.** Point at the tiles: **779,425** clean sales lines, **5,878** customers,
  **£17.37M** revenue, **2009-12-01 → 2011-12-09**. Open **How it works: the pipeline** for
  5 seconds: Raw Excel → Clean → Features → K-Means / 3 classifiers / association rules → Dashboard.
- **Tour 2 · Try it.** The page opens on the **Typical Champions** customer (the median of each
  number: last order 66 days ago, 7 orders, £2,553 spent, customer for 576 days, £344 per
  basket, 105 different products). The basket already holds **POPPY'S PLAYHOUSE BEDROOM**.
- **Tour 3 · Three answers.** Walk the cards left to right:
  - **Who is this customer?** "K-Means puts them in **Champions**; the star is where they sit
    among real customers on the PCA map."
  - **Will they come back?** "The chosen model, KNN, gives **63%**, above its **35%**
    threshold, so it says **likely to come back** within 90 days. All 3 models agree; each
    black tick is that model's own threshold."
  - **What to offer** "**Poppy's Playhouse Kitchen**, because you bought the Bedroom
    (confidence 86%, lift 51.4), then the **Living Room** (71%, lift 54.9). The other 3 slots
    are best sellers, and we never offer what is already in the basket."
- Click **Typical Lost / Hibernating**: last order 396 days ago, 1 order, £306. The star jumps
  to the Lost region and P(return) falls to **3%**, so the verdict is **unlikely to come back**.
  (Typical Regulars gives 20%.)
- Optional: type **Recency 30** for this customer: P(return) only rises from 3% to **11%**,
  because one order and £306 still look like a one-off buyer.
- "Each card has a **How does this work?** link. Let's follow them."

*If slow:* the first load on the cloud can take ~10 s (the models load once). Keep talking
through the opening.

## Data (30 s): can we trust it?

- Dataset card: UCI Online Retail II, CC BY 4.0, **1,067,371** raw rows in two Excel sheets.
- **Tour 1 · Cleaning funnel:** "We drop 34,335 duplicates, move 19,104 returns to their own
  table, drop 6,019 invalid rows, and the biggest loss is **228,488 lines with no Customer ID**
  (guest checkouts). That leaves **779,425** clean lines."
- **Tour 3 · Concentration:** "The top 20% of customers bring **77%** of revenue (Gini 0.74).
  That's why we segment instead of using averages."

## Segmentation (1 min): A, who are they?

- The 6 features, one line each: recency, frequency, monetary, tenure, average basket,
  distinct products. Skewed ones get log(1 + x), then all are standardised.
- **Tour 1 · Choosing k:** "WCSS always falls, so the elbow is soft. Our fixed rule: every
  segment holds at least 3% of customers, then the best silhouette for k = 3–7 wins. **k = 3**
  (silhouette **0.287**, no other k within 0.02), and it is stable (ARI 0.99). A human can
  override k and the names in `config.yaml`."
- **Tour 2 · PCA map:** "PC1 (55% of the variance) is 'buys more': higher spend, more orders,
  more products. PC2 (21%) is mostly time: tenure and recency." Flip the 3D toggle once if
  there is time.
- **Tour 3 · Profiles:** "**Champions**: 40% of customers, **88% of revenue**. **Regulars**:
  31%, 7.5%. **Lost / Hibernating**: 29%, 4.5%." Point at the customers % vs revenue % bars.

## Prediction (1.5 min): B, will they come back?

- **Tour 1 · Out-of-time split:** "The models learn from customers as they were on
  **11 Jun 2011** and whether they bought in the next 90 days. They are judged on the real
  cutoff, **9 Sep 2011**: a future they never saw. A random split would leak the future."
- **Tour 2 · Model comparison:** "Test ROC-AUC: logistic **0.795**, KNN **0.794**, Naive Bayes
  **0.790**. KNN was chosen on training cross-validation only (0.822); against logistic the
  difference is not significant (p = 0.30). The models are practically tied."
  Confusion matrix: "KNN flags **1,942** of 5,278 customers; **73%** of them came back, and they
  include 62% of everyone who came back."
- Click **Reveal what actually happened**. **Tour 3 · The reveal:** "73% of our list came back,
  against a base rate of **43%**: a lift of **1.68**. The top tenth came back 91% of the time,
  the bottom tenth 11%."
- **Tour 4 · KNN:** move the slider to **1**: "Train accuracy 100%, test 67.8%: a 32-point gap.
  It memorises the training data (overfitting)." Then **75 (best)**: "78.6% vs 70.7%, gap 7.9
  points: the best balance." Then **100**: "too simple; from here on the curve is flat, more
  neighbours stop helping."

## Recommendation (45 s): C, what should we offer?

- **Tour 1 · Three numbers** on the strongest rule, POPPY'S PLAYHOUSE BEDROOM → LIVINGROOM:
  "**Support 1.1%**: the pair is in about 321 of 29,962 invoices. **Confidence 71%**: of the
  baskets with the Bedroom, 71% also have the Living Room. **Lift 54.9**: the Living Room is in
  only 1.3% of baskets, so having the Bedroom makes it 55 times more likely."
- **Tour 2 · Every rule:** 200 rules; big dark bubbles have the highest lift.
- Optional: pick another product in the picker to show its rules.
- **Tour 3 · Hit-rate:** "We gave every customer 5 suggestions from data before the cutoff and
  checked what they bought next: rules caught a new purchase for **22.4%** of 2,194 customers,
  best sellers for **16.1%**."

## Insights + limitations (45 s)

- **Tour 1 · Per segment:** "**Champions** (40% of customers, 88% of revenue, 64% came back):
  **Reward and keep**. **Regulars**, the newest customers: **Second-order offer**.
  **Lost / Hibernating**, last order about 410 days ago on average and only 17% came back:
  **Win-back offer**." The action follows a fixed rule (expander on the page).
- Close on the 4 limitations at the bottom of the page:
  > "**One:** registered customers only; 22.7% of lines had no Customer ID. **Two:** one
  > cutoff, a single time split. **Three:** the November peak falls only in the test window.
  > **Four:** rules show co-purchase, not cause, and were never tested live. Everything is
  > reproducible: one pipeline command, 275 tests, and the code is on GitHub."

---

## Likely examiner questions

**Why k = 3?** A fixed rule decides it before looking at names: every segment must hold at
least 3% of customers, then the best silhouette for k = 3–7 wins. k = 3 has 0.287 and no other
k is within 0.02; it is also stable (ARI 0.99 over subsamples). The elbow is soft, so we did not
rely on it alone. A human can override k in `config.yaml`.

**Why an out-of-time split instead of a random split?** The real task is to predict the future
from the past. A random split would mix future behaviour into training. We train on the
11 Jun 2011 snapshot and test at the 9 Sep 2011 cutoff, with features built only from data
before each snapshot. A leakage test checks that rows after the cutoff never change the features.

**Why do the models tie?** All three land at 0.79–0.80 test ROC-AUC, and the best (KNN) vs
logistic difference is not significant (paired t, p = 0.30). Recency, frequency and spend carry
the signal, and the boundary is fairly simple. We choose on training CV only, never on test.

**Why is k = 100 "too simple" when its accuracy equals k = 75?** Between 75 and 100 the curve
is flat; the best k is chosen by cross-validated ROC-AUC on the training data, not by test
accuracy. With ever more neighbours KNN drifts towards predicting the same answer for everyone.

**Why do the models disagree on the Home page?** Each model has its own threshold (best F1 on
training), so compare each bar with its own tick. Naive Bayes is often over-confident
(probabilities near 0 or 1) because it assumes the features are independent, which they are not
(frequency and spend move together).

**Why are most suggestions best sellers?** A rule only fires when the basket holds its first
product, and the 200 rules cover 105 starting products. Without a fired rule, the honest
fallback is a best seller.

**Why log1p and scaling?** Spend and frequency are heavily right-skewed. K-Means and KNN use
distances, so skewed, unscaled features would let one variable dominate.

**What does lift mean statistically?** Lift = P(A and B) / (P(A) · P(B)). Lift 1 means A and B
are independent. The top rule has lift 54.9, and a chi-square test rejects independence
(p < 0.001).

**MCAR vs MAR for the missing Customer IDs?** Lines without an ID have much smaller quantities,
higher prices and ship to the UK more often, so the IDs are not missing completely at random.
It fits guest checkouts (MAR). Those lines cannot be linked to a customer, so the results
describe registered customers only.
