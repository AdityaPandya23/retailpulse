"""Insights page logic: one row per segment (size, revenue, came back, mean predicted P(return),
top recommended products, suggested action) and the limitations listed in README.md.

Action rule (exactly one action per segment, checked in this order, first match wins):
  1. the segment with the highest mean recency      -> "Win-back offer"
  2. the segment with the highest mean monetary     -> "Reward and keep"
  3. the segment with the lowest mean tenure        -> "Second-order offer"
  4. otherwise                                      -> "Keep engaged"
Ties go to the lowest segment id. Segment ids and names come from the artifacts, never
hard-coded.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from retailpulse.viz import common as VC
from retailpulse.viz.predictions import test_predictions

WIN_BACK = "Win-back offer"
REWARD = "Reward and keep"
SECOND_ORDER = "Second-order offer"
KEEP_ENGAGED = "Keep engaged"
ACTIONS = (WIN_BACK, REWARD, SECOND_ORDER, KEEP_ENGAGED)

INSIGHT_COLUMNS = [
    "segment",
    "Segment",
    "Customers %",
    "Revenue %",
    "Came back %",
    "Avg P(return)",
    "Top 3 recommended products",
    "Suggested action",
    "Why",
]


def suggest_actions(profiles: pd.DataFrame | None) -> dict[int, tuple[str, str]]:
    """{segment id: (action, reason)} from segment mean recency / monetary / tenure."""
    if profiles is None or profiles.empty:
        return {}
    p = profiles.assign(segment=profiles["segment"].astype(int)).set_index("segment").sort_index()
    top_recency = int(p["recency"].idxmax())
    top_monetary = int(p["monetary"].idxmax())
    low_tenure = int(p["tenure"].idxmin())
    out = {}
    for sid, r in p.iterrows():
        sid = int(sid)
        if sid == top_recency:
            out[sid] = (
                WIN_BACK,
                f"longest since last purchase (mean recency {r['recency']:,.0f} days)",
            )
        elif sid == top_monetary:
            out[sid] = (REWARD, f"highest spend (mean {VC.money(float(r['monetary']))})")
        elif sid == low_tenure:
            out[sid] = (
                SECOND_ORDER,
                f"newest customers (mean tenure {r['tenure']:,.0f} days)",
            )
        else:
            out[sid] = (KEEP_ENGAGED, "no extreme on recency, spend or tenure")
    return out


def _with_segment(frame: pd.DataFrame | None, segments: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or segments is None or frame.empty or segments.empty:
        return pd.DataFrame()
    return frame.merge(segments[["customer_id", "segment"]], on="customer_id", how="inner")


def mean_p_return(
    classification: pd.DataFrame | None, segments: pd.DataFrame | None, model: str | None
) -> dict[int, float]:
    """{segment: mean predicted P(return)} of one model on the test split."""
    j = _with_segment(test_predictions(classification, model), segments)
    if j.empty:
        return {}
    return {int(s): float(v) for s, v in j.groupby("segment")["p_return"].mean().items()}


def came_back_share(
    profiles: pd.DataFrame | None,
    classification: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
    model: str | None = None,
) -> dict[int, float]:
    """{segment: share who bought again in the label window}: profiles.bought_again_share,
    else the actual outcome of the test split."""
    if profiles is not None and "bought_again_share" in profiles.columns:
        return {
            int(s): float(v)
            for s, v in zip(profiles["segment"], profiles["bought_again_share"], strict=True)
        }
    j = _with_segment(test_predictions(classification, model), segments)
    if j.empty:
        return {}
    return {int(s): float(v) for s, v in j.groupby("segment")["returned_flag"].mean().items()}


def top_recommended(
    recommendations: pd.DataFrame | None, segments: pd.DataFrame | None, n: int = 3
) -> dict[int, list[str]]:
    """{segment: the n products most often recommended to its customers} (ties: by name)."""
    j = _with_segment(recommendations, segments)
    if j.empty:
        return {}
    counts = j.groupby(["segment", "description"]).size().rename("n").reset_index()
    counts = counts.sort_values(["segment", "n", "description"], ascending=[True, False, True])
    return {
        int(s): g["description"].astype(str).head(n).tolist() for s, g in counts.groupby("segment")
    }


def segment_insights(
    profiles: pd.DataFrame | None,
    names: Mapping[int, str] | None,
    segments: pd.DataFrame | None,
    classification: pd.DataFrame | None,
    model: str | None,
    recommendations: pd.DataFrame | None,
    n_products: int = 3,
) -> pd.DataFrame:
    """One row per segment (id order) with INSIGHT_COLUMNS; shares are 0-1 floats."""
    if profiles is None or profiles.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    names = dict(names or VC.segment_names({}, profiles))
    actions = suggest_actions(profiles)
    p_ret = mean_p_return(classification, segments, model)
    back = came_back_share(profiles, classification, segments, model)
    products = top_recommended(recommendations, segments, n_products)
    rows = []
    for r in profiles.sort_values("segment").itertuples(index=False):
        sid = int(r.segment)
        action, why = actions.get(sid, (KEEP_ENGAGED, ""))
        rows.append(
            {
                "segment": sid,
                "Segment": f"{sid} · {names.get(sid, r.segment_name)}",
                "Customers %": float(r.customer_share),
                "Revenue %": float(r.revenue_share),
                "Came back %": back.get(sid, np.nan),
                "Avg P(return)": p_ret.get(sid, np.nan),
                "Top 3 recommended products": ", ".join(products.get(sid, [])) or "n/a",
                "Suggested action": action,
                "Why": why,
            }
        )
    return pd.DataFrame(rows, columns=INSIGHT_COLUMNS)


def limitations(readme_text: str, heading: str = "## Limitations") -> list[str]:
    """The bullet points under `heading` in README.md (continuation lines joined), as markdown."""
    items: list[str] = []
    inside = False
    for line in readme_text.splitlines():
        if line.strip() == heading:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("#"):
            break
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith("  ") and line.strip() and items:
            items[-1] += " " + line.strip()
    return items
