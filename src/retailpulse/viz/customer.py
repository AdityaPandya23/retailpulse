"""Page 4 (Customer 360): one customer's timeline, segment badge, return gauge, expected spend,
recommendations with the rule that fired, and the What-if model application.

Models are the joblib payloads from artifacts/models/ and take RAW (un-logged) features:
  segment_model {pipeline, features, label_to_segment, segment_names} -> predict_segment
  classifier    {pipeline, features, model, threshold}                 -> P(return)
  regressor     {pipeline, features, model, target, note}              -> log1p(spend | return)
Expected 90-day spend = P(return) x expm1(regressor prediction).
"""

from __future__ import annotations

import html
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from retailpulse import contracts as C
from retailpulse.segment.cluster import predict_segment
from retailpulse.viz import common as VC
from retailpulse.viz import palette as P

FEATURE_LABELS: dict[str, str] = {
    "recency": "Recency (days since last purchase)",
    "frequency": "Frequency (orders)",
    "monetary": "Monetary (total spend, £)",
    "tenure": "Tenure (days since first purchase)",
    "avg_basket_value": "Average basket (£)",
    "mean_interpurchase_days": "Days between purchases (mean)",
    "n_distinct_products": "Distinct products bought",
    "return_rate": "Return rate (share of value returned)",
    "non_uk_share": "Non-UK share of spend",
}
SHARE_FEATURES = ("return_rate", "non_uk_share")
MONEY_FEATURES = ("monetary", "avg_basket_value")
SLIDER_QUANTILE = 0.99  # slider max = 99th percentile (or the customer's own value if larger)


# ------------------------------------------------------------------ lookup
def customer_ids(features: pd.DataFrame | None, segments: pd.DataFrame | None = None) -> list[int]:
    """Every modelled customer (features_cutoff, else segments), ascending."""
    for df in (features, segments):
        if df is not None and "customer_id" in df and len(df):
            return sorted(int(c) for c in df["customer_id"].unique())
    return []


def customer_labels(ids, segments: pd.DataFrame | None, names: Mapping[int, str]) -> dict:
    """{id: "id · segment name"} for the searchable selectbox."""
    seg = {}
    if segments is not None and {"customer_id", "segment"} <= set(segments):
        seg = dict(
            zip(segments["customer_id"].astype(int), segments["segment"].astype(int), strict=True)
        )
    out = {}
    for cid in ids:
        s = seg.get(int(cid))
        out[int(cid)] = f"{cid} · {names.get(s, f'segment {s}')}" if s is not None else f"{cid}"
    return out


def default_customer(
    features: pd.DataFrame | None,
    segments: pd.DataFrame | None,
    segment_id: int = 0,
    prefer_ids: list | None = None,
) -> int | None:
    """A TYPICAL customer of a segment (default 0 = most valuable): the medoid-like member whose
    log1p features are closest (sum of |robust z|) to the segment median. Customers in every
    `prefer_ids` set are preferred (e.g. has a test prediction and a rule recommendation).
    Ties -> smallest id. Deterministic; falls back to the first customer."""
    ids = customer_ids(features, segments)
    if not ids:
        return None
    if segments is None or features is None or "segment" not in segments:
        return ids[0]
    members = segments.loc[segments["segment"] == segment_id, "customer_id"].astype(int)
    pool = features[features["customer_id"].astype(int).isin(members)]
    if pool.empty:
        return ids[0]
    for pref in prefer_ids or []:
        sub = pool[pool["customer_id"].astype(int).isin(set(int(i) for i in pref))]
        if len(sub):
            pool = sub
    cols = [c for c in C.FEATURE_COLUMNS if c in pool and c not in SHARE_FEATURES]
    x = np.log1p(pool[cols].clip(lower=0).astype("float64"))
    med = x.median()
    iqr = (x.quantile(0.75) - x.quantile(0.25)).replace(0, 1.0)
    dist = ((x - med) / iqr).abs().sum(axis=1)
    pick = pd.DataFrame({"cid": pool["customer_id"].astype(int), "d": dist}).sort_values(
        ["d", "cid"]
    )
    return int(pick["cid"].iloc[0])


def row_of(df: pd.DataFrame | None, cid: int) -> pd.Series | None:
    """The single row of a per-customer table, or None (missing table / customer)."""
    if df is None or "customer_id" not in df:
        return None
    hit = df[df["customer_id"] == int(cid)]
    return None if hit.empty else hit.iloc[0]


def rows_of(df: pd.DataFrame | None, cid: int) -> pd.DataFrame:
    if df is None or "customer_id" not in df:
        return pd.DataFrame()
    return df[df["customer_id"] == int(cid)]


def test_prediction(
    classification: pd.DataFrame | None, cid: int, model: str | None
) -> dict | None:
    """The backtest prediction for one customer: p_return, pred_return, returned_flag."""
    if classification is None or model is None:
        return None
    hit = classification[
        (classification["customer_id"] == int(cid))
        & (classification["split"] == "test")
        & (classification["model"] == model)
    ]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "p_return": float(r["p_return"]),
        "pred_return": int(r["pred_return"]),
        "returned_flag": int(r["returned_flag"]),
    }


# ------------------------------------------------------------------ models (What-if)
def model_features(*payloads) -> list[str]:
    """Union of the payloads' raw features, in the contract order (sliders)."""
    used = {f for p in payloads if p for f in p.get("features", [])}
    extra = sorted(used - set(C.FEATURE_COLUMNS))
    return [f for f in C.FEATURE_COLUMNS if f in used] + extra


def predict_segments(segment_model: Mapping | None, raw: pd.DataFrame) -> np.ndarray:
    if not segment_model:
        return np.full(len(raw), -1, dtype="int64")
    return predict_segment(dict(segment_model), raw)


def predict_return(classifier: Mapping | None, raw: pd.DataFrame) -> np.ndarray:
    """P(buys again within the horizon) from the classifier payload."""
    if not classifier:
        return np.full(len(raw), np.nan)
    p = classifier["pipeline"].predict_proba(raw[classifier["features"]])[:, 1]
    return np.clip(p.astype("float64"), 0.0, 1.0)


def predict_spend_if_return(regressor: Mapping | None, raw: pd.DataFrame) -> np.ndarray:
    """Typical 90-day spend (£) IF the customer comes back: expm1 of the log1p prediction."""
    if not regressor:
        return np.full(len(raw), np.nan)
    pred = regressor["pipeline"].predict(raw[regressor["features"]])
    return np.clip(np.expm1(np.asarray(pred, dtype="float64")), 0.0, None)


def expected_spend(p_return, spend_if_return):
    """Expected 90-day spend = P(return) x spend if they return."""
    return np.asarray(p_return, dtype="float64") * np.asarray(spend_if_return, dtype="float64")


def apply_models(
    raw: pd.DataFrame,
    segment_model: Mapping | None,
    classifier: Mapping | None,
    regressor: Mapping | None,
) -> pd.DataFrame:
    """Raw feature rows -> segment, p_return, spend_if_return, expected_spend (one row each)."""
    raw = raw.reset_index(drop=True)
    p = predict_return(classifier, raw)
    spend = predict_spend_if_return(regressor, raw)
    return pd.DataFrame(
        {
            "segment": predict_segments(segment_model, raw),
            "p_return": p,
            "spend_if_return": spend,
            "expected_spend": expected_spend(p, spend),
        }
    )


def _nice_step(span: float) -> float:
    if not np.isfinite(span) or span <= 0:
        return 1.0
    return float(10 ** np.floor(np.log10(span / 200)))


def slider_specs(features: pd.DataFrame, row: pd.Series, columns: list[str]) -> list[dict]:
    """One slider per raw feature: range from features_cutoff (min .. 99th percentile, widened
    to hold the customer's value), integers where the column is integer, prefilled value."""
    specs = []
    for col in columns:
        if col not in features:
            continue
        s = features[col].astype("float64")
        val = (
            float(row[col])
            if row is not None and col in row and pd.notna(row[col])
            else float(s.median())
        )
        is_int = pd.api.types.is_integer_dtype(features[col])
        if col in SHARE_FEATURES:
            lo, hi, step = 0.0, 1.0, 0.01
        else:
            lo = min(float(s.min()), val, 0.0) if col != "frequency" else min(float(s.min()), val)
            hi = max(float(s.quantile(SLIDER_QUANTILE)), val)
            step = 1.0 if is_int else _nice_step(hi - lo)
        if is_int:
            lo, hi, val, step = int(np.floor(lo)), int(np.ceil(hi)), int(round(val)), 1
            hi = max(hi, lo + 1)
        else:
            val = float(np.clip(val, lo, hi))
            hi = hi if hi > lo else lo + step
        specs.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "min": lo,
                "max": hi,
                "step": step,
                "value": val,
                "is_int": bool(is_int),
            }
        )
    return specs


def raw_frame(
    values: Mapping[str, float], base: pd.Series | None, columns: list[str]
) -> pd.DataFrame:
    """One raw feature row: slider values over the customer's actual row (other columns)."""
    out = {}
    for col in columns:
        if col in values:
            out[col] = values[col]
        elif base is not None and col in base:
            out[col] = base[col]
        else:
            out[col] = 0.0
    return pd.DataFrame([out], columns=columns)


# ------------------------------------------------------------------ figures / display
def timeline_figure(timeline: pd.DataFrame, cutoff, horizon_days: int | None = None) -> go.Figure:
    """Monthly revenue bars coloured before / after the cutoff, cutoff line + outcome window."""
    title = "Monthly spend before and after the cutoff"
    if timeline is None or timeline.empty:
        return VC.empty_figure("No purchases recorded for this customer.", title)
    t = timeline.sort_values("month")
    fig = go.Figure()
    labels = {"before": "before the cutoff (model inputs)", "after": "after the cutoff (outcome)"}
    for period in ("before", "after"):
        d = t[t["period"] == period]
        if d.empty:
            continue
        colour = P.PERIOD_COLOURS[period]
        fig.add_trace(
            go.Bar(
                x=d["month"],
                y=d["revenue"],
                name=labels[period],
                marker={"color": colour, "line": {"color": P.SURFACE, "width": 1}},
                customdata=d[["orders"]],
                hovertemplate=(
                    "%{x|%b %Y}<br>£%{y:,.2f}<br>%{customdata[0]} order(s)"
                    f"<br>{period} the cutoff<extra></extra>"
                ),
                xperiod="M1",
                xperiodalignment="middle",
            )
        )
    if cutoff is not None:
        cut = pd.Timestamp(cutoff)
        if horizon_days:
            end = cut + pd.Timedelta(days=int(horizon_days))
            fig.add_vrect(
                x0=cut, x1=end, fillcolor=P.GRID, opacity=0.6, line_width=0, layer="below"
            )
        fig.add_shape(
            type="line",
            x0=cut,
            x1=cut,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": P.INK, "width": 2, "dash": "dash"},
        )
        fig.add_annotation(
            x=cut,
            y=1,
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            text=f"cutoff {cut.date()} ",
            font={"color": P.INK, "size": 12},
        )
    fig.update_layout(barmode="stack", legend={"orientation": "h", "y": -0.25})
    y_title = "Revenue per month (£)"
    if VC.is_skewed(t["revenue"]):
        VC.log_axis(fig, "y", y_title)
        y_title = None
    return VC.style(fig, title, "Month", y_title, height=360)


def gauge_figure(p: float | None, threshold: float | None, model: str | None) -> go.Figure:
    """Return-probability gauge (0-100%) in the model's colour, threshold as an ink tick."""
    title = f"Chance of buying again within 90 days · {model}"
    if p is None or not np.isfinite(p):
        return VC.empty_figure("No prediction for this customer.", title)
    colour = P.model_colour(model or "")
    gauge = {
        "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": P.LIGHT_GREY},
        "bar": {"color": colour, "thickness": 0.6},
        "bgcolor": P.GRID,
        "borderwidth": 0,
    }
    if threshold is not None:
        gauge["threshold"] = {
            "line": {"color": P.INK, "width": 3},
            "thickness": 0.9,
            "value": 100 * threshold,
        }
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * p,
            number={"suffix": "%", "valueformat": ".0f", "font": {"color": P.INK}},
            gauge=gauge,
        )
    )
    fig.update_layout(margin={"l": 30, "r": 30, "t": 60, "b": 10})
    return VC.style(fig, title, height=260)


def segment_badge_html(segment_id: int | None, name: str | None, note: str = "") -> str:
    """Coloured square (segment colour) + the name in ink text: identity never colour-alone."""
    if segment_id is None or segment_id < 0:
        return f'<span style="color:{P.MUTED_INK}">segment unknown</span>'
    colour = P.segment_colour(int(segment_id))
    ring = P.INK if colour.upper() == P.YELLOW else colour
    label = html.escape(str(name if name is not None else f"segment {segment_id}"))
    extra = f' <span style="color:{P.MUTED_INK}">{html.escape(note)}</span>' if note else ""
    return (
        f'<span style="display:inline-block;width:0.9em;height:0.9em;border-radius:3px;'
        f"background:{colour};border:1px solid {ring};vertical-align:-0.1em;"
        f'margin-right:0.4em"></span>'
        f'<span style="color:{P.INK};font-weight:600">{label}</span>'
        f' <span style="color:{P.MUTED_INK}">(segment {int(segment_id)})</span>{extra}'
    )


def format_feature(col: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if col in SHARE_FEATURES:
        return VC.pct(float(value), 1)
    if col in MONEY_FEATURES:
        return VC.money(float(value))
    if col in ("recency", "tenure", "mean_interpurchase_days"):
        return f"{float(value):,.0f} days"
    return f"{float(value):,.0f}"


def key_features(row: pd.Series | None, columns: list[str] | None = None) -> list[dict]:
    """kpi_row items (label, value) for the customer's raw features."""
    if row is None:
        return []
    columns = columns or [c for c in C.FEATURE_COLUMNS if c in row]
    return [
        {
            "label": FEATURE_LABELS.get(c, c).split(" (")[0],
            "value": format_feature(c, row[c]),
            "help": FEATURE_LABELS.get(c),
        }
        for c in columns
        if c in row
    ]


def recommendation_table(
    recs: pd.DataFrame | None, reasons: pd.DataFrame | None, cid: int
) -> pd.DataFrame:
    """Top recommendations of one customer + the rule that fired (rule recs) or the fallback."""
    cols = [
        "Rank",
        "Product",
        "Stock code",
        "Score",
        "Source",
        "Why",
        "Confidence",
        "Lift",
        "Rule scope",
    ]
    r = rows_of(recs, cid)
    if r.empty:
        return pd.DataFrame(columns=cols)
    r = r.sort_values("rank").head(5)
    why = rows_of(reasons, cid)
    if not why.empty:
        why = why.drop_duplicates("stock_code").set_index(
            why.drop_duplicates("stock_code")["stock_code"].astype(str)
        )
    rows = []
    for rec in r.itertuples(index=False):
        source = str(getattr(rec, "source", "rule") or "rule")
        code = str(rec.stock_code)
        reason = why.loc[code] if not why.empty and code in why.index else None
        if reason is not None:
            text = f"because you bought {reason['antecedents']} → {reason['consequents']}"
            conf, lift, scope = (
                float(reason["confidence"]),
                float(reason["lift"]),
                str(reason["scope"]),
            )
        else:
            text = (
                "best seller in your segment (no rule fired)"
                if source == "popular"
                else "association rule (details not stored)"
            )
            conf = lift = None
            scope = ""
        rows.append(
            {
                "Rank": int(rec.rank),
                "Product": str(rec.description),
                "Stock code": code,
                "Score": float(rec.score),
                "Source": source,
                "Why": text,
                "Confidence": conf,
                "Lift": lift,
                "Rule scope": scope,
            }
        )
    return pd.DataFrame(rows, columns=cols)
