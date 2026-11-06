# RetailPulse: demo script (~8 minutes, app v2 "hub and spokes")

The Home page is the hub. You start there, visit the Data page, come back, and then follow each
answer card to its "How does this work?" page and back. All numbers below were read from the app
on the committed artifacts and from `artifacts/metrics.json` (cutoff 2011-09-09, 90-day horizon).
If you change k or the segment names in `config.yaml`, the app follows; then update the names and
numbers here to match what the app shows.

## Before you start

- [ ] Run `streamlit run app/Home.py` locally as a backup, in a second browser tab.
- [ ] Open the live app about 5 minutes early (a sleeping Streamlit Cloud app needs a minute
      to wake; the first Home load can take ~15 s while the models load).
- [ ] Click every page once (Home, Data, Segmentation, Prediction, Recommendation, Insights) so
      the caches are warm.
- [ ] Check that there is **no "Mock data" badge** under any page title or in the sidebar.
- [ ] Set the browser zoom to **110–125%** so the room can read it.
- [ ] Switch **Guided Tour** on in the sidebar (numbered notes on every page), then collapse the
      sidebar (the « arrow). With the sidebar open at 110–125% zoom, the three Home cards get
      too narrow for their chart titles. Move between pages with the links on the cards and the
      "← Back to the live demo" link at the bottom of each page.

| Time | Where |
|---|---|
| 0:00 | Home: top strip (45 s) |
| 0:45 | Data (45 s) |
| 1:30 | Home: Typical Champions, card 1 → Segmentation (1 min 15 s) |
| 2:45 | Card 2 → Prediction (1 min 45 s) |
| 4:30 | Card 3 → Recommendation (45 s) |
| 5:15 | Home: Typical Lost / Hibernating (30 s) |
| 5:45 | The examiner types their own numbers (45 s) |
| 6:30 | Insights + the 4 limitations (1 min 15 s) |

---

## 0:00 Home: top strip (45 s)

> "RetailPulse uses two years of invoices from a UK online gift shop to answer three questions:
> **who are our customers, will they come back, and what should we offer them?** We use only
> free, open-source tools."

- **Tour 1 · The data.** Point at the 4 tiles: **779,425** clean sales lines, **5,878**
  customers, **£17.37M** revenue, **2009-12-01 → 2011-12-09**.
- Open **How it works: the pipeline** for a few seconds: "Raw Excel with 2 sheets, then Clean,
  then 6 features per customer built only from data before the cutoff. The features feed
  K-Means and the 3 classifiers; the association rules come straight from the cleaned baskets.
  An offline pipeline builds all of this once; the app only reads the results."
- "First, can we trust the data?" Click **The data and how it was cleaned** at the bottom.

## 0:45 Data (45 s)

- Dataset card: UCI Online Retail II, licence CC BY 4.0, **1,067,371** raw rows in **2** Excel
  sheets, 8 columns, and 5 sample rows after cleaning.
- **Tour 1 · Cleaning funnel:** "We drop **34,335** duplicates, move **19,104** returns to their
  own table and drop **6,019** lines with an invalid quantity or price. The biggest loss is
  **228,488** lines with no Customer ID: guest checkouts. That leaves **779,425** clean lines,
  so our results describe registered customers only."
- **Tour 2 · Seasonality:** "Revenue peaks every November: about **£1.2M** in November 2010.
  The last month is partial, so it is drawn hollow."
- **Tour 3 · Concentration:** "The top **20%** of customers bring **77%** of revenue, the top
  1% bring 32% (Gini **0.74**), and the UK brings 83%. That is why we look at segments
  instead of one average customer."
- Click **← Back to the live demo**.

## 1:30 Home: a typical Champions customer, card 1 → Segmentation (1 min 15 s)

- **Tour 2 · Try it.** Click **Typical Champions**: "These are the medians of the most valuable
  segment: last order **66** days ago, **7** orders, **£2,553.20** spent, a customer for
  **576** days, **£343.73** per basket, **105** different products." The basket already holds
  **POPPY'S PLAYHOUSE BEDROOM**.
- **Tour 3 · Three answers.** Card 1, **Who is this customer?**: "K-Means puts them in
  **Champions (segment 0)**. The star is where they sit among about 2,000 real customers on
  the PCA map." Click **How does this work? →**.

**Segmentation page**

- **The 6 numbers:** recency, frequency, monetary, tenure, average basket value, distinct
  products. "Skewed ones get log(1 + x), then every feature is standardised so no single one
  dominates the distances."
- **Tour 1 · Choosing k:** "On the left, the error (WCSS) always falls as k grows, and the
  elbow is soft. On the right, the silhouette. A fixed rule picks k automatically: among k = 3
  to 7, every segment must hold at least 3% of customers, and the best silhouette wins. That is
  **k = 3** (silhouette **0.287**); no other k is within 0.02 of it."
- **Tour 2 · PCA map:** "PC1 (**55%** of the variance) means higher total spend, more orders
  and more distinct products. PC2 (**21%**) is mostly time: tenure and recency." Optional:
  flip **3D view** once.
- **Tour 3 · Profiles:** "**Champions**: 2,107 customers (40%) bring **88%** of revenue, and
  64% bought again. **Regulars**: 31% of customers, 8% of revenue. **Lost / Hibernating**:
  29% of customers, 5% of revenue, and only 17% came back." Point at the customers % vs
  revenue % bars.
- Click **← Back to the live demo**. The inputs are kept.

## 2:45 Card 2 → Prediction (1 min 45 s)

- Card 2, **Will they come back?**: "The chosen model, KNN, gives **63%**, above its **35%**
  threshold, so: **likely to come back** within 90 days. **3 of 3** models agree. Each black
  tick is that model's own threshold: logistic gives 49%, Naive Bayes 95%." Click
  **How does this work? →**.

**Prediction page**

- **Tour 1 · Out-of-time split:** "The models learn from customers as they looked on
  **2011-06-11** and whether they bought in the next 90 days. They are judged on the real
  cutoff, **2011-09-09**: a future they never saw. A random split would leak the future into
  training."
- **Tour 2 · Model comparison:** "Test ROC-AUC: logistic **0.795**, KNN **0.794**, Naive Bayes
  **0.790**. We chose KNN (k = 75) on training cross-validation only (0.822). The spread on
  test is 0.006, about the size of the fold-to-fold noise, so the models are practically
  tied." Confusion matrix: "KNN flags **1,942** of 5,278 customers; 73% of them came back, and
  they include 62% of everyone who came back."
- **Tour 3 · Time Machine:** "On 9 Sep 2011 the model wrote down 1,942 names." Click
  **Reveal what actually happened**: "**73%** of them bought again, against a base rate of
  **43%**: a lift of **1.68×**. The top tenth came back 91% of the time, the bottom tenth only
  11%."
- **Tour 4 · KNN:** move the slider:
  - **1**: "Train accuracy **100.0%**, test **67.8%**, a **32.2-point** gap. With one
    neighbour it memorises the training data: overfitting."
  - **75 (best)**: "**78.6%** vs **70.7%**, a **7.9-point** gap: the best balance, chosen by
    cross-validation on the training data."
  - **100**: "Also **78.6%** vs **70.7%**. From about 50 neighbours on the curve is flat, so
    100 is not worse here; it is just on the simpler side. With far more neighbours every
    customer would get the same average answer: underfitting."
- Click **← Back to the live demo**.

## 4:30 Card 3 → Recommendation (45 s)

- Card 3, **What to offer**: "**1. POPPY'S PLAYHOUSE KITCHEN**, because you bought the Bedroom
  (confidence **86%**, lift **51.4**). **2. POPPY'S PLAYHOUSE LIVINGROOM** (71%, lift 54.9).
  The other three are best sellers, and we never offer what is already in the basket."
  Click **How does this work? →**.

**Recommendation page**

- **Tour 1 · Three numbers** on the strongest rule, BEDROOM → LIVINGROOM: "**Support 1.1%**:
  the pair is in about **321 of 29,962** invoices. **Confidence 71%**: of the baskets with the
  Bedroom, 71% also have the Living Room. **Lift 54.9**: the Living Room is in only 1.3% of
  baskets, so the Bedroom makes it about 55 times more likely."
- **Tour 2 · Every rule:** "200 rules; big, dark bubbles have the highest lift." Optional: pick
  another product in the picker.
- **Tour 3 · Hit-rate:** "We gave every customer 5 suggestions using only data before the
  cutoff and checked what they bought next. Rules caught a new purchase for **22.4%** of
  2,194 customers, best sellers for **16.1%**."
- Click **← Back to the live demo**.

## 5:15 Home: the least active segment (30 s)

- Click **Typical Lost / Hibernating**: last order **396** days ago, **1** order, **£305.68**,
  a customer for 506 days, 18 products.
- What changes: "The badge turns **Lost / Hibernating (segment 2)**, and the star jumps to the
  green group. P(return) drops to **3%**, below the 35% threshold: **unlikely to come back**,
  and all 3 models agree." (Typical Regulars gives 20%.)
- What does not change: "The offers stay the same, because they come from the basket, not from
  the customer's numbers."
- Optional: set **Recency** to **30**: the segment becomes **Regulars**, but P(return) only
  rises to **11%**, because one order of £306 still looks like a one-off buyer.

## 5:45 The examiner types their own numbers (45 s)

> "Please give me a customer of your own."

Keep **tenure ≥ recency** (otherwise a warning appears: the first purchase cannot be after the
last one) and keep the average basket close to monetary ÷ frequency. Two safe examples:

| Example | Recency | Frequency | Monetary (£) | Tenure | Avg basket (£) | Products | App shows |
|---|---|---|---|---|---|---|---|
| Loyal buyer | 10 | 12 | 3000 | 600 | 250 | 120 | **Champions**, KNN **75%**, likely, 3 of 3 agree |
| Fading buyer | 150 | 2 | 300 | 300 | 150 | 20 | **Regulars**, KNN **11%**, unlikely, 3 of 3 agree |

If the examiner's customer lands near the 35% threshold, the models can split (for example a
brand-new customer with one order 30 days ago gets exactly 35%, and only 1 of 3 models agrees).
Say so openly: "This is a borderline customer; that is exactly what a threshold means."

## 6:30 Insights + limitations (1 min 15 s)

- Click **What should the business do? (Insights)**.
- **Tour 1 · Per segment:** "**Champions**: 40% of customers, 88% of revenue, 64% came back:
  **Reward and keep** (highest spend, mean £5,802). **Regulars**, the newest customers (mean
  tenure 223 days): **Second-order offer**. **Lost / Hibernating**, last order **410 days**
  ago on average, 17% came back: **Win-back offer**." The action comes from a fixed rule
  (open **How the action is chosen**).
- Close on the 4 limitations at the bottom of the page:
  > "**One:** registered customers only; 22.7% of lines had no Customer ID. **Two:** one
  > cutoff, a single time split. **Three:** the November peak falls only in the test window
  > (32% came back in the train window vs 43% in the test window). **Four:** rules show
  > co-purchase, not cause, and were never tested live. Everything is reproducible: one
  > pipeline command, 293 tests, and the code is on GitHub."

---

## Likely examiner questions

**Why these 6 features?** Recency, frequency and monetary are the classic RFM trio of customer
value. Tenure separates new customers from old ones, average basket separates big wholesale
orders from small ones, and distinct products shows how broad a customer's buying is. Each is
one easy number per customer, built only from data before the cutoff. Three more (mean days
between purchases, return rate, non-UK share) are still computed but not used: the first
mostly repeats recency and frequency, and non-UK share is 0 for most customers (about 9 in 10
are in the UK).

**Why K-Means, and how was k chosen?** K-Means is simple, fast and gives every customer exactly
one segment, and each segment's centre reads as a "typical customer". The elbow (WCSS) is
soft, so a fixed rule decides automatically, before anyone looks at names: among k = 3 to 7,
every segment must hold at least 3% of customers; the best silhouette wins; k values within
0.02 of it compete on stability (ARI over subsamples), and a tie goes to the smaller k. Result:
k = 3, silhouette 0.287, no k within 0.02, stability ARI 0.99. Names come from a rule on each
segment's z-scores. A human can override both with `segment.k` and `segment.names` in
`config.yaml`.

**What does PCA do here?** It rotates the 6 standardised features into new axes ordered by how
much of the variation they carry. PC1 (55%) and PC2 (21%) together keep 76% of it, enough for
an honest 2D map. PCA is only for drawing and checking; K-Means uses all 6 features. A new
customer is placed with the same saved PCA, which is the ★ on Home.

**Why an out-of-time split instead of a random split?** The real task is to predict the future
from the past. We train on the 2011-06-11 snapshot and test at the 2011-09-09 cutoff, with
features built only from data before each snapshot. A random split would mix future behaviour
into training and look better than reality. A leakage test checks that rows after the cutoff
never change the features.

**Why do the models tie?** Test ROC-AUC is 0.790 to 0.795, a spread of 0.006, about the size of
the fold-to-fold noise (±0.013). On the training folds KNN beats logistic by only +0.003 (paired
t, p = 0.299, not significant). Most of the signal is in recency and frequency, which every
model picks up. We choose on training CV only, so we keep KNN even though logistic is slightly
better on test, and we say so.

**Overfitting vs underfitting on the KNN chart?** With k = 1 every training customer is its own
nearest neighbour, so train accuracy is 100% but test is 67.8%: a 32-point gap, overfitting
(high variance). More neighbours average more votes, so the model gets smoother: train error
rises and the gap shrinks. k = 75 is the best by cross-validation (78.6% vs 70.7%). **Why does
k = 100 look like k = 75?** Once there are many neighbours, 25 more barely change any vote, so
the curve is flat and the rounded accuracies are the same. The "too simple" label only says that
100 is on the simpler side of the best k; with far more neighbours every customer would get the
same base-rate answer and both errors would rise. The best k comes from cross-validated ROC-AUC
on the training data, never from test accuracy.

**Support vs confidence vs lift?** On BEDROOM → LIVINGROOM: support = how common the pair is
(1.1% of invoices, about 321 of 29,962); confidence = how often the rule is right
(71% of Bedroom baskets also have the Living Room); lift = confidence ÷ how common the Living
Room is anyway (71% ÷ 1.3% ≈ 54.9). Lift 1 means no link. We need all three: high confidence
alone can just be a best seller (lift near 1), and high lift on a tiny support can be luck. So
the rules need support ≥ 1%, confidence ≥ 30% and lift > 1.

**Why do the models disagree on Home?** Each model has its own threshold (best F1 on training),
so compare each bar with its own tick. Naive Bayes is often over-confident (95% for the typical
Champion, 0% for the typical Lost customer) because it assumes the features are independent,
and they are not (frequency and spend move together).

**Why are most suggestions best sellers?** A rule only fires when the basket holds its first
product. With no fired rule, the honest fallback is a best seller.

**Missing Customer IDs: MCAR or MAR?** Lines without an ID have much smaller quantities, higher
prices and are more often from the UK, so they are not missing completely at random; it fits
guest checkouts (MAR). They cannot be linked to a customer, so the results describe registered
customers only.
