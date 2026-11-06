"""Figures + table helpers for the v2 Recommendation page (association rules).

Inputs: rules.parquet (one row per rule: antecedents -> consequents with support, confidence,
lift; *_codes columns hold the stock codes), products.parquet (code -> description) and
metrics.json "rules". Only the "overall" scope is shown (rules.per_segment = false in v2).
Every function returns a DataFrame, a dict or a go.Figure and never raises on missing data.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from retailpulse.viz import common as VC
from retailpulse.viz import palette as P

OVERALL = "overall"
# display table: support and confidence already in percent (0-100) so the page only formats
RULE_TABLE_COLUMNS = [
    "If a basket has",
    "it often also has",
    "Support (%)",
    "Confidence (%)",
    "Lift",
]
RULES_COLOUR = P.BLUE
BEST_SELLER_COLOUR = P.GREY


def overall_rules(rules: pd.DataFrame | None) -> pd.DataFrame:
    """The overall-scope rules, strongest lift first; all rules only when there is no scope
    column. Same rule set as rules.recommend_from_basket (overall only), so the Home offers and
    this page never disagree."""
    empty = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    if rules is None or rules.empty:
        return empty
    r = rules
    if "scope" in r.columns:
        r = r[r["scope"] == OVERALL]
        if r.empty:
            return empty
    return r.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def _table(r: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "If a basket has": r["antecedents"].astype(str).to_numpy(),
            "it often also has": r["consequents"].astype(str).to_numpy(),
            "Support (%)": 100 * r["support"].astype(float).to_numpy(),
            "Confidence (%)": 100 * r["confidence"].astype(float).to_numpy(),
            "Lift": r["lift"].astype(float).to_numpy(),
        },
        columns=RULE_TABLE_COLUMNS,
    )


def top_rules_table(rules: pd.DataFrame | None, n: int = 10) -> pd.DataFrame:
    """The n strongest overall rules (by lift) as a display table (support, confidence in %)."""
    return _table(overall_rules(rules).head(n))


def overall_invoices(rules_metrics: Mapping | None) -> int | None:
    """Number of invoices behind the overall rules (metrics.json rules.n_invoices)."""
    n = (rules_metrics or {}).get("n_invoices")
    n = n.get(OVERALL) if isinstance(n, Mapping) else n
    return int(n) if n else None


def rule_title(top_rule: Mapping | None) -> str | None:
    """'**A → B**' (markdown-escaped) for one rule, or None."""
    t = top_rule or {}
    if not t.get("antecedents") or not t.get("consequents"):
        return None
    return f"**{VC.md_escape(t['antecedents'])} → {VC.md_escape(t['consequents'])}**"


def default_product(top_rule: Mapping | None, codes) -> str | None:
    """First antecedent code of the top rule if it is a picker option, else the first option."""
    codes = list(codes)
    first = str((top_rule or {}).get("antecedent_codes") or "").split(",")[0]
    return first if first in codes else (codes[0] if codes else None)


def rule_explainer(top_rule: Mapping | None, n_invoices: int | None = None) -> dict[str, str]:
    """Support / confidence / lift of one rule in plain words (markdown), keyed by measure.

    P(A) = support / confidence and P(B) = confidence / lift follow from the definitions, so
    only the three stored measures are needed.
    """
    t = top_rule or {}
    try:
        s, c, lift = float(t["support"]), float(t["confidence"]), float(t["lift"])
    except (KeyError, TypeError, ValueError):
        return {}
    a = VC.md_escape(t.get("antecedents", "A"))  # product names can hold *, _ or $
    b = VC.md_escape(t.get("consequents", "B"))
    p_b = c / lift if lift else float("nan")
    count = (
        f" (about {round(s * n_invoices):,} of {int(n_invoices):,} invoices)" if n_invoices else ""
    )
    return {
        "support": (
            f"**Support {s:.1%}**: *{a}* and *{b}* are in the same basket in {s:.1%} of all "
            f"invoices{count}. Support says how common the pair is."
        ),
        "confidence": (
            f"**Confidence {c:.0%}**: of the baskets that contain *{a}*, {c:.0%} also contain "
            f"*{b}*. Confidence says how often the rule is right."
        ),
        "lift": (
            f"**Lift {lift:.1f}**: *{b}* is in only {p_b:.1%} of all baskets, so a basket with "
            f"*{a}* is {lift:.1f}× more likely to have it. Lift 1 = no link; above 1 = bought "
            "together more than by chance."
        ),
    }


def rules_bubble(rules: pd.DataFrame | None) -> go.Figure:
    """Support (x) vs confidence (y) per overall rule; bubble size and colour = lift."""
    title = "Every rule: support vs confidence (size and colour = lift)"
    r = overall_rules(rules)
    if r.empty:
        return VC.empty_figure("No association rules.", title)
    lift = r["lift"].astype(float)
    size = 8 + 30 * (lift - lift.min()) / max(float(lift.max() - lift.min()), 1e-9)
    fig = go.Figure(
        go.Scatter(
            x=r["support"],
            y=r["confidence"],
            mode="markers",
            name="rule",
            marker={
                "size": size,
                "color": lift,
                "colorscale": P.SEQUENTIAL,
                "showscale": True,
                "colorbar": {"title": {"text": "Lift"}},
                "line": {"color": P.MUTED_INK, "width": 0.8},
                "opacity": 0.85,
            },
            customdata=np.column_stack(
                [r["antecedents"].astype(str), r["consequents"].astype(str), lift]
            ),
            hovertemplate=(
                "%{customdata[0]} → %{customdata[1]}<br>support %{x:.2%} · confidence "
                "%{y:.0%} · lift %{customdata[2]:.1f}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(tickformat=".0%")
    VC.style(fig, title, "Support (share of all invoices)", "Confidence", height=460)
    if VC.is_skewed(r["support"]):
        VC.log_axis(fig, "x", "Support (share of all invoices)", tickformat=".1%")
    else:
        fig.update_xaxes(tickformat=".1%")
    fig.update_layout(showlegend=False)
    return fig


def _code_list(s) -> list[str]:
    return [c for c in str(s).split(",") if c]


def antecedent_products(
    rules: pd.DataFrame | None, products: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Products that start at least one overall rule: stock_code, description, n_rules
    (most rules first). Descriptions from products.parquet, else the rule text."""
    cols = ["stock_code", "description", "n_rules"]
    r = overall_rules(rules)
    if r.empty:
        return pd.DataFrame(columns=cols)
    if "antecedent_codes" in r.columns:
        codes = r["antecedent_codes"].map(_code_list)
        single = r["antecedents"].where(codes.map(len) == 1)
    else:  # no codes: the antecedent text is the key
        codes = r["antecedents"].map(lambda s: [str(s)])
        single = r["antecedents"]
    exploded = pd.DataFrame({"stock_code": codes, "text": single}).explode("stock_code")
    lookup = {}
    if products is not None and not products.empty:
        lookup = dict(
            zip(
                products["stock_code"].astype(str), products["description"].astype(str), strict=True
            )
        )
    out = (
        exploded.groupby("stock_code")
        .agg(n_rules=("stock_code", "size"), text=("text", "first"))
        .reset_index()
    )
    out["description"] = [
        lookup.get(c) or (t if isinstance(t, str) else c)
        for c, t in zip(out["stock_code"], out["text"], strict=True)
    ]
    out = out.sort_values(["n_rules", "description"], ascending=[False, True])
    return out[cols].reset_index(drop=True)


def rules_for_product(rules: pd.DataFrame | None, code: str | None) -> pd.DataFrame:
    """Overall rules whose antecedents contain the product, strongest lift first."""
    r = overall_rules(rules)
    if r.empty or code is None:
        return _table(r.head(0))
    if "antecedent_codes" in r.columns:
        mask = r["antecedent_codes"].map(lambda s: str(code) in _code_list(s))
    else:
        mask = r["antecedents"].astype(str) == str(code)
    return _table(r[mask])


def hit_rate_bar(rules_metrics: Mapping | None, k: int = 5) -> go.Figure:
    """Hit-rate@k: share of customers who bought one of their k suggestions, rules vs best
    sellers (metrics.json rules)."""
    title = f"Hit-rate@{k}: did customers buy one of their {k} suggestions?"
    m = rules_metrics or {}
    hit, pop = m.get(f"hit_rate_at_{k}"), m.get(f"popularity_hit_rate_at_{k}")
    if hit is None or pop is None:
        return VC.empty_figure("Hit-rate not recorded.", title)
    labels = [f"Association rules (top {k})", f"Best sellers (top {k})"]
    values = [float(hit), float(pop)]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": [RULES_COLOUR, BEST_SELLER_COLOUR]},
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont={"color": P.INK, "size": 14},
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            width=0.5,
        )
    )
    fig.update_yaxes(tickformat=".0%", range=[0, max(values) * 1.25 + 0.01])
    n = m.get("n_eval_customers")
    y_title = "Customers with a hit" + (f" (of {int(n):,})" if n else "")
    VC.style(fig, title, None, y_title, height=380)
    fig.update_layout(showlegend=False)
    return fig
