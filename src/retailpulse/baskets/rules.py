"""Association rules on invoices before the cutoff, and rule-based recommendations.

Baskets: every invoice (strictly before the cutoff, registered customer) with >= 1 product line
(features.is_product) is one row; columns = the scope's top_n_products by invoice count, so
support = share of ALL the scope's product invoices. Per scope the support threshold is
max(min_support, min_abs_invoices / n_invoices), so a rule always rests on >= min_abs_invoices
baskets. Rules keep confidence >= min_confidence and lift > 1, top max_rules_per_scope by lift.

Scopes: "overall" and "segment:<segment_name>" (invoices of that segment's customers).

Recommendations (per customer, n items, never an already-bought product):
  1. products suggested by rules whose antecedents the customer already bought; score =
     sum of confidence x lift over those rules (segment + overall). Products backed by a
     segment rule rank before products backed only by overall rules.
  2. filled with the segment's best sellers (else overall) not yet bought; score = the
     product's support in that scope (share of invoices containing it), always ranked after
     rule-based items. `source` says which: "rule" | "popular".

Basket recommendations (recommend_from_basket, for products picked in the app): the same rule
score over the "overall" rules, plus the reason = the strongest fired rule (antecedents,
confidence, lift); filled with best sellers; never an item already in the basket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import sparse, stats

from retailpulse.config import ALPHA, RULES
from retailpulse.features import is_product

OVERALL = "overall"
RULE_COLUMNS = [
    "scope",
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift",
    "antecedent_codes",
    "consequent_codes",
    "antecedent_support",
    "consequent_support",
    "n_invoices",
    "min_support",
]
REC_COLUMNS = ["customer_id", "rank", "stock_code", "description", "score", "source"]
BASKET_REC_COLUMNS = [
    "rank",
    "stock_code",
    "description",
    "score",  # sum of confidence x lift of the fired rules; NaN for "popular"
    "source",  # "rule" | "popular"
    "antecedents",  # reason = strongest fired rule (None for "popular")
    "confidence",
    "lift",
]


def segment_scope(name) -> str:
    return f"segment:{name}"


# ----------------------------------------------------------------- basket data
def product_lines(sales: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Product lines of registered customers with start <= invoice_date < end."""
    s = sales[sales["customer_id"].notna()]
    if start is not None:
        s = s[s["invoice_date"] >= pd.Timestamp(start)]
    if end is not None:
        s = s[s["invoice_date"] < pd.Timestamp(end)]
    s = s[is_product(s["stock_code"])]
    return pd.DataFrame(
        {
            "invoice": s["invoice"].astype(str).to_numpy(),
            "customer_id": s["customer_id"].astype("int64").to_numpy(),
            "stock_code": s["stock_code"].astype(str).to_numpy(),
            "description": s["description"].astype(str).to_numpy(),
        }
    )


def descriptions(lines: pd.DataFrame) -> pd.Series:
    """stock_code -> most common description (ties: alphabetically first)."""
    counts = lines.groupby(["stock_code", "description"]).size().rename("n").reset_index()
    counts = counts.sort_values(["stock_code", "n", "description"], ascending=[True, False, True])
    return counts.drop_duplicates("stock_code").set_index("stock_code")["description"]


def popularity(lines: pd.DataFrame) -> pd.Series:
    """stock_code -> number of invoices containing it, best sellers first (ties: code)."""
    n = lines.groupby("stock_code")["invoice"].nunique().rename("n_invoices")
    df = n.reset_index().sort_values(["n_invoices", "stock_code"], ascending=[False, True])
    return df.set_index("stock_code")["n_invoices"]


def product_table(
    lines: pd.DataFrame, top_n: int = RULES.top_n_products, desc: pd.Series | None = None
) -> pd.DataFrame:
    """Top-n products by invoice count (the overall rule universe), best sellers first."""
    desc = descriptions(lines) if desc is None else desc
    pop = popularity(lines).head(top_n)
    codes = [str(c) for c in pop.index]
    return pd.DataFrame(
        {
            "stock_code": codes,
            "description": [str(desc.get(c, c)) for c in codes],
            "n_invoices": pop.to_numpy(dtype="int64"),
        }
    )


def basket_matrix(lines: pd.DataFrame, products) -> pd.DataFrame:
    """Sparse boolean invoice x product frame (every invoice in `lines` is a row)."""
    products = list(products)
    inv_codes, invoices = pd.factorize(lines["invoice"], sort=True)
    col = pd.Index(products).get_indexer(lines["stock_code"])
    keep = col >= 0
    m = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype="int32"), (inv_codes[keep], col[keep])),
        shape=(len(invoices), len(products)),
    )
    m = (m > 0).astype(bool)
    return pd.DataFrame.sparse.from_spmatrix(m, index=invoices, columns=products)


def support_floor(n_invoices: int, min_support: float, min_abs_invoices: int) -> float:
    return float(max(min_support, min_abs_invoices / max(n_invoices, 1)))


def _join(items, lookup=None) -> str:
    items = sorted(str(i) for i in items)
    if lookup is None:
        return ",".join(items)
    return " + ".join(sorted(str(lookup.get(i, i)) for i in items))


# ----------------------------------------------------------------- rules
def mine_rules(
    lines: pd.DataFrame,
    scope: str = OVERALL,
    desc: pd.Series | None = None,
    top_n_products: int = RULES.top_n_products,
    min_support: float = RULES.min_support,
    min_abs_invoices: int = RULES.min_abs_invoices,
    min_confidence: float = RULES.min_confidence,
    max_len: int = RULES.max_len,
    max_rules: int = RULES.max_rules_per_scope,
) -> tuple[pd.DataFrame, dict]:
    """Rules for one scope -> (RULE_COLUMNS frame, {n_invoices, min_support, n_rules, ...})."""
    desc = descriptions(lines) if desc is None else desc
    n_inv = int(lines["invoice"].nunique())
    sup = support_floor(n_inv, min_support, min_abs_invoices)
    info = {"n_invoices": n_inv, "min_support": sup, "n_frequent_itemsets": 0, "n_rules": 0}
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in RULE_COLUMNS})
    empty = empty.astype({c: "object" for c in RULE_COLUMNS[:3] + RULE_COLUMNS[6:8]})
    if n_inv == 0:
        return empty, info

    products = popularity(lines).index[:top_n_products]
    basket = basket_matrix(lines, products)
    freq = apriori(basket, min_support=sup, use_colnames=True, max_len=max_len, low_memory=True)
    info["n_frequent_itemsets"] = int(len(freq))
    if freq.empty or freq["itemsets"].map(len).max() < 2:
        return empty, info
    rules = association_rules(
        freq, num_itemsets=n_inv, metric="confidence", min_threshold=min_confidence
    )
    rules = rules[(rules["lift"] > 1) & (rules["confidence"] >= min_confidence)]
    info["n_rules_before_cap"] = int(len(rules))
    if rules.empty:
        return empty, info
    lookup = desc.to_dict()
    out = pd.DataFrame(
        {
            "scope": scope,
            "antecedents": rules["antecedents"].map(lambda s: _join(s, lookup)),
            "consequents": rules["consequents"].map(lambda s: _join(s, lookup)),
            "support": rules["support"].astype("float64"),
            "confidence": rules["confidence"].astype("float64"),
            "lift": rules["lift"].astype("float64"),
            "antecedent_codes": rules["antecedents"].map(_join),
            "consequent_codes": rules["consequents"].map(_join),
            "antecedent_support": rules["antecedent support"].astype("float64"),
            "consequent_support": rules["consequent support"].astype("float64"),
            "n_invoices": n_inv,
            "min_support": sup,
        }
    )
    out = out.sort_values(
        ["lift", "confidence", "support", "antecedent_codes", "consequent_codes"],
        ascending=[False, False, False, True, True],
    ).head(max_rules)
    info["n_rules"] = int(len(out))
    return out.reset_index(drop=True), info


def scope_lines(
    lines: pd.DataFrame, segments: pd.DataFrame | None
) -> list[tuple[str, pd.DataFrame]]:
    """[("overall", all lines), ("segment:<name>", lines of that segment's customers), ...]."""
    scopes = [(OVERALL, lines)]
    if segments is None or segments.empty:
        return scopes
    seg = segments[["customer_id", "segment", "segment_name"]].drop_duplicates("customer_id")
    for (_, name), grp in seg.groupby(["segment", "segment_name"], sort=True):
        ids = grp["customer_id"].astype("int64")
        scopes.append((segment_scope(name), lines[lines["customer_id"].isin(ids)]))
    return scopes


def mine_all(
    lines: pd.DataFrame, segments: pd.DataFrame | None = None, **params
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Rules for every scope (overall + one per segment when `segments` is given)."""
    desc = descriptions(lines)
    frames, infos = [], {}
    for scope, sl in scope_lines(lines, segments):
        r, info = mine_rules(sl, scope=scope, desc=desc, **params)
        frames.append(r)
        infos[scope] = info
    non_empty = [f for f in frames if not f.empty]
    rules = pd.concat(non_empty, ignore_index=True) if non_empty else frames[0]
    return rules, infos


def rule_contingency(lines: pd.DataFrame, antecedent_codes: str, consequent_codes: str) -> dict:
    """2x2 invoice counts for A (all antecedent items) vs B (all consequent items) + chi-square."""
    a = set(antecedent_codes.split(","))
    b = set(consequent_codes.split(","))
    invoices = pd.Index(lines["invoice"].unique())

    def has_all(items: set) -> pd.Series:
        hit = lines[lines["stock_code"].isin(items)].groupby("invoice")["stock_code"].nunique()
        return pd.Series(invoices.isin(hit[hit == len(items)].index), index=invoices)

    has_a, has_b = has_all(a), has_all(b)
    n_ab = int((has_a & has_b).sum())
    n_a_notb = int((has_a & ~has_b).sum())
    n_nota_b = int((~has_a & has_b).sum())
    n_none = int((~has_a & ~has_b).sum())
    table = np.array([[n_ab, n_a_notb], [n_nota_b, n_none]])
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    phi = float(np.sqrt(chi2 / n)) if n else float("nan")
    p_b_given_a = n_ab / max(n_ab + n_a_notb, 1)
    p_b_given_not_a = n_nota_b / max(n_nota_b + n_none, 1)
    return {
        "counts": {"a_and_b": n_ab, "a_not_b": n_a_notb, "not_a_b": n_nota_b, "neither": n_none},
        "expected_a_and_b": float(expected[0, 0]),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "phi": phi,
        "p_b_given_a": float(p_b_given_a),
        "p_b_given_not_a": float(p_b_given_not_a),
    }


# ----------------------------------------------------------------- recommendations
def _codes(s: str) -> list[str]:
    return [c for c in str(s).split(",") if c]


def _rule_scores(
    rules: pd.DataFrame, customer_ids: np.ndarray, bought: dict[int, set]
) -> dict[int, dict[str, float]]:
    """customer -> {product: sum of confidence x lift over fired rules}, bought items excluded."""
    if rules.empty or len(customer_ids) == 0:
        return {}
    ante = [_codes(s) for s in rules["antecedent_codes"]]
    cons = [_codes(s) for s in rules["consequent_codes"]]
    items = sorted({c for lst in ante + cons for c in lst})
    col = {c: j for j, c in enumerate(items)}

    def indicator(lists: list[list[str]], n_rows: int) -> sparse.csr_matrix:
        r = [i for i, lst in enumerate(lists) for c in lst if c in col]
        c = [col[c] for lst in lists for c in lst if c in col]
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n_rows, len(items)))

    A = indicator(ante, len(rules))
    K = indicator(cons, len(rules))
    B = indicator([list(bought.get(int(cid), ())) for cid in customer_ids], len(customer_ids))
    B.data[:] = 1.0
    fired = (B @ A.T).toarray() >= np.asarray(A.sum(axis=1)).ravel()[None, :]
    w = (rules["confidence"] * rules["lift"]).to_numpy()
    S = (fired * w) @ K.toarray()
    S[B.toarray() > 0] = 0.0
    out = {}
    for i, j in zip(*np.nonzero(S), strict=True):
        out.setdefault(int(customer_ids[i]), {})[items[j]] = float(S[i, j])
    return out


def recommend_all(
    lines: pd.DataFrame,
    rules: pd.DataFrame,
    customer_ids,
    segments: pd.DataFrame | None = None,
    n: int = RULES.n_recommend,
    desc: pd.Series | None = None,
) -> pd.DataFrame:
    """n recommendations for every customer in `customer_ids` (REC_COLUMNS)."""
    desc = descriptions(lines) if desc is None else desc
    customer_ids = np.asarray(pd.Series(customer_ids).astype("int64").unique())
    bought = lines.groupby("customer_id")["stock_code"].agg(set).to_dict()
    seg_of: dict[int, str] = {}
    if segments is not None and not segments.empty:
        seg_of = dict(
            zip(
                segments["customer_id"].astype("int64"),
                segments["segment_name"].astype(str),
                strict=True,
            )
        )

    # popularity lists (support = share of scope invoices) for the fill step
    pop: dict[str, list[tuple[str, float]]] = {}
    for scope, sl in scope_lines(lines, segments):
        n_inv = max(int(sl["invoice"].nunique()), 1)
        p = popularity(sl)
        pop[scope] = list(zip(p.index, (p / n_inv).to_numpy(dtype="float64"), strict=True))
    pop_rank = {code: i for i, (code, _) in enumerate(pop[OVERALL])}

    rules = rules if rules is not None else pd.DataFrame(columns=RULE_COLUMNS)
    overall = _rule_scores(
        rules[rules["scope"] == OVERALL] if not rules.empty else rules, customer_ids, bought
    )
    seg_scores: dict[int, dict[str, float]] = {}
    for name in sorted(set(seg_of.values())):
        scope = segment_scope(name)
        ids = np.array([c for c in customer_ids if seg_of.get(int(c)) == name], dtype="int64")
        sr = rules[rules["scope"] == scope] if not rules.empty else rules
        seg_scores.update(_rule_scores(sr, ids, bought))

    lookup = desc.to_dict()
    rows = []
    for cid in customer_ids:
        cid = int(cid)
        have = bought.get(cid, set())
        s_seg, s_all = seg_scores.get(cid, {}), overall.get(cid, {})
        cand = {c: s_seg.get(c, 0.0) + s_all.get(c, 0.0) for c in set(s_seg) | set(s_all)}
        ranked = sorted(
            cand,
            key=lambda c: (c not in s_seg, -cand[c], pop_rank.get(c, len(pop_rank)), c),
        )
        picks = [(c, cand[c], "rule") for c in ranked[:n]]
        chosen = {c for c, _, _ in picks}
        seg_scope_name = segment_scope(seg_of[cid]) if cid in seg_of else None
        for scope in (seg_scope_name, OVERALL):
            if len(picks) >= n or scope not in pop:
                continue
            for code, support in pop[scope]:
                if len(picks) >= n:
                    break
                if code not in have and code not in chosen:
                    picks.append((code, support, "popular"))
                    chosen.add(code)
        for rank, (code, score, source) in enumerate(picks, start=1):
            rows.append((cid, rank, code, lookup.get(code, code), score, source))
    out = pd.DataFrame(rows, columns=REC_COLUMNS)
    return out.astype({"customer_id": "int64", "rank": "int64", "score": "float64"})


def recommend(
    customer_id: int,
    lines: pd.DataFrame,
    rules: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    n: int = RULES.n_recommend,
) -> pd.DataFrame:
    """Top-n products the customer has not bought yet (see module docstring for the scoring)."""
    return recommend_all(lines, rules, [customer_id], segments, n)


def recommend_from_basket(
    codes, rules: pd.DataFrame | None, popularity, n: int = RULES.n_recommend
) -> pd.DataFrame:
    """Top-n products for a basket of stock codes, never one already in the basket.

    rules: rules.parquet-like frame; only the "overall" scope is used (a basket has no segment).
    popularity: products.parquet-like frame (stock_code, n_invoices[, description]) or a
    Series stock_code -> n_invoices (popularity()). Rule items rank by score = sum of
    confidence x lift (_rule_scores), their reason is the strongest fired rule; the rest are
    best sellers ("popular", score NaN, no reason). Returns BASKET_REC_COLUMNS.
    """
    basket = {str(c) for c in codes}
    lookup: dict[str, str] = {}
    if isinstance(popularity, pd.DataFrame):
        products = popularity.assign(stock_code=popularity["stock_code"].astype(str))
        if "description" in products.columns:
            lookup = dict(
                zip(products["stock_code"], products["description"].astype(str), strict=True)
            )
        pop = products.set_index("stock_code")["n_invoices"]
    else:
        pop = pd.Series(popularity, dtype="float64")
        pop.index = pop.index.astype(str)
    order = sorted(pop.index, key=lambda c: (-float(pop[c]), c))  # best sellers first
    pop_rank = {c: i for i, c in enumerate(order)}

    r = pd.DataFrame(columns=RULE_COLUMNS) if rules is None else rules
    if not r.empty and "scope" in r.columns:
        r = r[r["scope"] == OVERALL]
    r = r.reset_index(drop=True)
    scores = _rule_scores(r, np.array([0]), {0: basket}).get(0, {}) if not r.empty else {}

    # reason: the strongest (confidence x lift) fired rule whose consequents hold the item
    reason: dict[str, int] = {}
    if scores:
        weight = (r["confidence"] * r["lift"]).to_numpy(dtype="float64")
        for i in np.argsort(-weight, kind="stable"):
            if not set(_codes(r.at[i, "antecedent_codes"])) <= basket:
                continue
            for item in _codes(r.at[i, "consequent_codes"]):
                if item in scores:
                    reason.setdefault(item, int(i))

    ranked = sorted(scores, key=lambda c: (-scores[c], pop_rank.get(c, len(pop_rank)), c))
    picks = [(c, scores[c], "rule") for c in ranked[:n]]
    chosen = {c for c, _, _ in picks}
    for c in order:
        if len(picks) >= n:
            break
        if c not in basket and c not in chosen:
            picks.append((c, float("nan"), "popular"))
            chosen.add(c)

    ante_col = "antecedents" if "antecedents" in r.columns else "antecedent_codes"
    rows = []
    for rank, (code, score, source) in enumerate(picks, start=1):
        i = reason.get(code)
        rows.append(
            (
                rank,
                code,
                lookup.get(code, code),
                score,
                source,
                None if i is None else str(r.at[i, ante_col]),
                float("nan") if i is None else float(r.at[i, "confidence"]),
                float("nan") if i is None else float(r.at[i, "lift"]),
            )
        )
    out = pd.DataFrame(rows, columns=BASKET_REC_COLUMNS)
    return out.astype(
        {"rank": "int64", "score": "float64", "confidence": "float64", "lift": "float64"}
    )


def hit_rate(recs: pd.DataFrame, before: pd.DataFrame, window: pd.DataFrame, k: int = 5) -> dict:
    """Share of customers who bought a NEW product in the window and got >= 1 of it in top-k."""
    old = before.groupby("customer_id")["stock_code"].agg(set)
    w = window[window["customer_id"].isin(recs["customer_id"])]
    new_items = {}
    for cid, codes in w.groupby("customer_id")["stock_code"].agg(set).items():
        new = codes - old.get(cid, set())
        if new:
            new_items[int(cid)] = new
    top = recs[recs["rank"] <= k].groupby("customer_id")["stock_code"].agg(set)
    hits = sum(bool(top.get(cid, set()) & new) for cid, new in new_items.items())
    n_eval = len(new_items)
    return {
        "n_eval_customers": n_eval,
        "hits": int(hits),
        "hit_rate": hits / n_eval if n_eval else 0.0,
    }


# ----------------------------------------------------------------- summary
def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def summarize(
    rules: pd.DataFrame,
    infos: dict[str, dict],
    chi: dict | None,
    hit: dict,
    pop_hit: dict,
    min_confidence: float = RULES.min_confidence,
    k: int = 5,
) -> dict:
    """metrics["rules"] with every P3_METRICS_KEYS["rules"] key."""
    n_rules = {scope: int(info["n_rules"]) for scope, info in infos.items()}
    top_rule = None
    ov = rules[rules["scope"] == OVERALL] if not rules.empty else rules
    if not ov.empty:
        t = ov.iloc[0]
        top_rule = {
            "scope": OVERALL,
            "antecedents": str(t["antecedents"]),
            "consequents": str(t["consequents"]),
            "antecedent_codes": str(t["antecedent_codes"]),
            "consequent_codes": str(t["consequent_codes"]),
            "support": float(t["support"]),
            "confidence": float(t["confidence"]),
            "lift": float(t["lift"]),
        }
    if chi is not None and top_rule is not None:
        dep = "not independent" if chi["p_value"] < ALPHA else "consistent with independence"
        chi = {
            **chi,
            "conclusion": (
                f"Invoices with {top_rule['antecedents']} contain {top_rule['consequents']} "
                f"{chi['p_b_given_a']:.0%} of the time vs {chi['p_b_given_not_a']:.1%} without it "
                f"(chi-square = {chi['chi2']:,.0f}, p {fmt_p(chi['p_value'])}): the two are {dep}, "
                f"matching lift {top_rule['lift']:.1f} > 1."
            ),
        }
    hr, phr = hit["hit_rate"], pop_hit["hit_rate"]
    seg_scopes = [s for s in n_rules if s != OVERALL]
    top_txt = (
        f"strongest overall rule: {top_rule['antecedents']} -> {top_rule['consequents']} "
        f"(lift {top_rule['lift']:.1f}, confidence {top_rule['confidence']:.0%})"
        if top_rule
        else "no overall rule passed the thresholds"
    )
    return {
        "min_support": infos[OVERALL]["min_support"],
        "min_support_by_scope": {s: i["min_support"] for s, i in infos.items()},
        "min_confidence": float(min_confidence),
        "n_invoices": {s: i["n_invoices"] for s, i in infos.items()},
        "n_rules": n_rules,
        "n_rules_before_cap": {s: int(i.get("n_rules_before_cap", 0)) for s, i in infos.items()},
        "segments_used": bool(seg_scopes),
        "top_rule": top_rule,
        "chi2_top_rule": chi,
        f"hit_rate_at_{k}": hr,
        f"popularity_hit_rate_at_{k}": phr,
        "n_eval_customers": hit["n_eval_customers"],
        "recommendation_note": (
            "score = sum of confidence x lift of fired rules (segment rules rank first); "
            "'popular' fill items score their invoice support in the segment (else overall)"
        ),
        "conclusion": (
            f"{n_rules[OVERALL]} overall rules"
            + (f" and {sum(n_rules[s] for s in seg_scopes)} segment rules" if seg_scopes else "")
            + f"; {top_txt}. Top-{k} recommendations caught a newly bought product for "
            f"{hr:.1%} of the {hit['n_eval_customers']:,} customers who bought something new in "
            f"the next window, vs {phr:.1%} for plain best sellers."
        ),
    }
