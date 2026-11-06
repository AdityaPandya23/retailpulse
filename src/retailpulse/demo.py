"""Live demo logic for the Home page (the hub): headline facts, the pipeline diagram, the
6 customer inputs, scoring one typed-in customer with the saved models, and the
"what to offer" lines. Pure functions (frames / dicts in, values out); the page only lays
them out. Nothing here is hard-coded: k, names, the best model and thresholds come from the
saved models / metrics.json, the features from config.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from retailpulse import contracts as C
from retailpulse.config import HORIZON_DAYS, PREDICT, SEGMENT
from retailpulse.segment import cluster

# the 6 model inputs in config order (segment.features, then any extra predict.features)
DEMO_FEATURES: list[str] = list(dict.fromkeys([*SEGMENT.features, *PREDICT.features]))

# feature -> number-input spec (label, help, min, step, integer?)
INPUT_SPECS: dict[str, dict] = {
    "recency": {
        "label": "Recency (days)",
        "help": "Days since the customer's last purchase.",
        "min": 0,
        "step": 1,
        "integer": True,
    },
    "frequency": {
        "label": "Frequency (orders)",
        "help": "Number of separate orders (invoices).",
        "min": 1,
        "step": 1,
        "integer": True,
    },
    "monetary": {
        "label": "Monetary (£)",
        "help": "Total amount spent so far.",
        "min": 0.0,
        "step": 50.0,
        "integer": False,
    },
    "tenure": {
        "label": "Tenure (days)",
        "help": "Days since the customer's first purchase.",
        "min": 0,
        "step": 1,
        "integer": True,
    },
    "avg_basket_value": {
        "label": "Average basket (£)",
        "help": "Average amount per order (monetary / frequency).",
        "min": 0.0,
        "step": 10.0,
        "integer": False,
    },
    "n_distinct_products": {
        "label": "Distinct products",
        "help": "Number of different products bought.",
        "min": 1,
        "step": 1,
        "integer": True,
    },
}


def input_spec(feature: str) -> dict:
    """Number-input spec of a feature; an unknown (config-added) feature gets a float spec."""
    default = {
        "label": feature.replace("_", " ").capitalize(),
        "help": None,
        "min": 0.0,
        "step": 1.0,
        "integer": False,
    }
    return {**default, **INPUT_SPECS.get(feature, {})}


def input_value(feature: str, value) -> int | float:
    """Cast a raw value to the input's type (int / float), never below the input's minimum.

    st.number_input needs value, min and step of ONE numeric type.
    """
    spec = input_spec(feature)
    v = float(value) if value is not None and np.isfinite(float(value)) else float(spec["min"])
    v = max(v, float(spec["min"]))
    return int(round(v)) if spec["integer"] else round(v, 2)


def preset_values(
    presets: pd.DataFrame | None, segment_id: int, features: Sequence[str] = DEMO_FEATURES
) -> dict:
    """{feature: value} of the typical (median) customer of one segment, cast for the inputs."""
    if presets is None or presets.empty or segment_id not in set(presets["segment"].astype(int)):
        return {}
    row = presets.loc[presets["segment"].astype(int) == int(segment_id)].iloc[0]
    return {f: input_value(f, row[f]) for f in features if f in row}


def input_warnings(values: Mapping) -> list[str]:
    """Plain-English notes on impossible input combinations (the models still score them)."""
    notes = []
    rec, ten = values.get("recency"), values.get("tenure")
    if rec is not None and ten is not None and float(ten) < float(rec):
        notes.append(
            "Tenure is shorter than recency: the first purchase cannot be after the last one."
        )
    return notes


# ------------------------------------------------------------------ headline facts
def kpi_facts(warehouse: Mapping | None, cleaning: Mapping | None, fmt: Mapping | None) -> dict:
    """Clean rows, customers, revenue and the raw date range from artifacts/quality/*.json."""
    warehouse, cleaning, fmt = warehouse or {}, cleaning or {}, fmt or {}
    counts = warehouse.get("row_counts") or {}
    steps = cleaning.get("steps") or []
    clean_rows = counts.get("fact_sales", steps[-1].get("rows_out") if steps else None)

    def _day(x):
        return str(x)[:10] if x else None

    return {
        "clean_rows": clean_rows,
        "raw_rows": fmt.get("rows", cleaning.get("raw_rows")),
        "customers": counts.get("dim_customer"),
        "revenue": warehouse.get("total_revenue"),
        "date_min": _day(fmt.get("date_min")),
        "date_max": _day(fmt.get("date_max")),
    }


def pipeline_dot(
    models: Sequence[str] = tuple(C.CLASSIFIERS), n_features: int = len(DEMO_FEATURES)
) -> str:
    """Graphviz DOT of the pipeline: Raw Excel -> Clean -> Features -> models -> Dashboard.

    Association rules are mined from the cleaned invoices (baskets), not from the customer
    features, so that arrow starts at Clean.
    """
    model_list = " · ".join(models)
    return "\n".join(
        [
            "digraph pipeline {",
            '  rankdir=LR; bgcolor="transparent";',
            '  node [shape=box, style="rounded,filled", fillcolor="#F2F7FB", color="#0072B2",'
            ' fontname="Helvetica", fontsize=11];',
            '  edge [color="#595959"];',
            '  raw [label="Raw Excel\\n(2 sheets)"];',
            '  clean [label="Clean\\n(duplicates, returns,\\nmissing IDs)"];',
            f'  feats [label="Features\\n({n_features} per customer,\\nbefore the cutoff)"];',
            '  kmeans [label="K-Means\\nsegments"];',
            f'  cls [label="{len(models)} classifiers\\n{model_list}"];',
            '  rules [label="Association rules\\n(baskets)"];',
            '  app [label="Dashboard\\n(this app)", fillcolor="#FFFFFF"];',
            "  raw -> clean -> feats;",
            "  feats -> kmeans; feats -> cls; clean -> rules;",
            "  kmeans -> app; cls -> app; rules -> app;",
            "}",
        ]
    )


# ------------------------------------------------------------------ scoring one customer
def raw_frame(values: Mapping, features: Sequence[str] = DEMO_FEATURES) -> pd.DataFrame:
    """One-row frame of raw feature values (the saved pipelines take raw values)."""
    return pd.DataFrame([{f: float(values[f]) for f in features}])


def place_segment(segment_model: Mapping | None, raw: pd.DataFrame) -> dict:
    """Segment id + its PCA position (pc1-pc3) for one raw customer row."""
    if not segment_model:
        return {"segment": None, "pc": None}
    seg = int(cluster.predict_segment(segment_model, raw)[0])
    pc = cluster.pca_coords(segment_model, raw)[0] if "pca" in segment_model else None
    return {"segment": seg, "pc": None if pc is None else [float(v) for v in pc]}


def return_probabilities(classifiers: Mapping | None, raw: pd.DataFrame) -> pd.DataFrame:
    """P(return) of every saved classifier (C.CLASSIFIERS order) + its train threshold."""
    cols = ["model", "p_return", "threshold", "chosen"]
    if not classifiers or not classifiers.get("pipelines"):
        return pd.DataFrame(columns=cols)
    pipes, thresholds = classifiers["pipelines"], classifiers.get("thresholds") or {}
    best = classifiers.get("best_model")
    feats = list(classifiers.get("features") or DEMO_FEATURES)
    order = [m for m in C.ALL_CLASSIFIERS if m in pipes] + [
        m for m in pipes if m not in C.ALL_CLASSIFIERS
    ]
    rows = []
    for m in order:
        p = float(pipes[m].predict_proba(raw[feats])[0, 1])
        t = thresholds.get(m)
        rows.append((m, p, None if t is None else float(t), m == best))
    return pd.DataFrame(rows, columns=cols)


def verdict(probs: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> dict:
    """Plain verdict from the chosen model: p vs its own threshold, + how many models agree."""
    if probs is None or probs.empty:
        return {"model": None, "will_return": None, "text": "No classifier available."}
    chosen = probs[probs["chosen"]]
    row = (chosen if len(chosen) else probs).iloc[0]
    thr = row["threshold"] if row["threshold"] is not None else 0.5
    yes = bool(row["p_return"] >= thr)
    votes = probs.assign(t=probs["threshold"].fillna(0.5))
    agree = int(((votes["p_return"] >= votes["t"]) == yes).sum())
    head = "Likely to come back" if yes else "Unlikely to come back"
    side = "above" if yes else "below"
    text = (
        f"**{head}** within {horizon_days} days: the chosen model ({row['model']}) gives "
        f"{row['p_return']:.0%}, {side} its {thr:.0%} threshold. "
        f"{agree} of {len(probs)} models agree."
    )
    return {
        "model": str(row["model"]),
        "p_return": float(row["p_return"]),
        "threshold": float(thr),
        "will_return": yes,
        "n_agree": agree,
        "text": text,
    }


# ------------------------------------------------------------------ what to offer
def default_basket(rules_metrics: Mapping | None, product_codes: Sequence[str]) -> list[str]:
    """Antecedents of the top overall rule (metrics.json), so the demo starts with a rule hit."""
    top = (rules_metrics or {}).get("top_rule") or {}
    codes = [c for c in str(top.get("antecedent_codes") or "").split(",") if c]
    available = {str(c) for c in product_codes}
    return [c for c in codes if c in available]


def recommendation_lines(recs: pd.DataFrame | None) -> list[dict]:
    """One dict per recommended product: rank, product, stock code and the plain reason."""
    if recs is None or recs.empty:
        return []
    out = []
    for r in recs.sort_values("rank").itertuples(index=False):
        if r.source == "rule" and r.antecedents:
            reason = (
                f"because you bought {r.antecedents} "
                f"(confidence {r.confidence:.0%}, lift {r.lift:.1f})"
            )
        else:
            reason = "best seller"
        out.append(
            {
                "rank": int(r.rank),
                "product": str(r.description),
                "code": str(r.stock_code),
                "source": str(r.source),
                "reason": reason,
            }
        )
    return out


def md_escape(text: str) -> str:
    """Escape markdown / LaTeX characters in product names before st.markdown."""
    out = str(text)
    for ch in ("\\", "*", "_", "$", "`", "[", "]", "#", "~", "|", "<", ">"):
        out = out.replace(ch, "\\" + ch)
    return out


def matching_preset(
    presets: pd.DataFrame | None, values: Mapping, features: Sequence[str] = DEMO_FEATURES
) -> int | None:
    """Segment id whose typical (preset) values equal the inputs, else None (custom values)."""
    if presets is None or presets.empty:
        return None
    for sid in sorted(presets["segment"].astype(int)):
        p = preset_values(presets, sid, features)
        if p and all(p.get(f) == values.get(f) for f in features):
            return int(sid)
    return None
