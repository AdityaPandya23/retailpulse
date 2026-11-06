"""Figures + table helpers for page 2 (Segments & Stats).

k and segment names are never hard-coded: every function takes `names` ({id: name}, ids
sorted, 0 = most valuable) from data.segment_names(); colours follow the id
(palette.segment_colour). Optional columns of the real artifacts (rfm_group, ari_mean,
median_monetary, bought_again_share ...) are used when present and skipped otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from retailpulse import contracts as C
from retailpulse.config import HORIZON_DAYS, SEED
from retailpulse.stats.tests import compare_groups, fmt_p
from retailpulse.viz import common as VC
from retailpulse.viz import palette as P

MAX_POINTS = 3000

# feature -> (label, unit for axis titles, unit passed to compare_groups)
FEATURE_INFO: dict[str, tuple[str, str, str]] = {
    "recency": ("Recency", "days since last purchase", "days"),
    "frequency": ("Frequency", "orders", "orders"),
    "monetary": ("Monetary", "£", "£"),
    "tenure": ("Tenure", "days since first purchase", "days"),
    "avg_basket_value": ("Average basket value", "£", "£"),
    "mean_interpurchase_days": ("Mean days between purchases", "days", "days"),
    "n_distinct_products": ("Distinct products", "products", "products"),
    "return_rate": ("Return rate", "share of value returned", ""),
    "non_uk_share": ("Non-UK share", "share of spend", ""),
}


# feature -> phrase used inside the live test's plain-English conclusion
FEATURE_WHAT = {
    "recency": "recency (days since last purchase)",
    "frequency": "order count",
    "monetary": "total spend",
    "tenure": "tenure (days since first purchase)",
    "avg_basket_value": "average basket value",
    "mean_interpurchase_days": "days between purchases",
    "n_distinct_products": "number of distinct products",
    "return_rate": "return rate",
    "non_uk_share": "non-UK share of spend",
}

# feature -> one plain line for the v2 Segmentation page (what the number means)
FEATURE_MEANING = {
    "recency": "days since the customer's last purchase (small = bought recently)",
    "frequency": "number of separate orders (invoices)",
    "monetary": "total amount spent (£)",
    "tenure": "days since the first purchase (how long they have been a customer)",
    "avg_basket_value": "average amount per order (£) = monetary / frequency",
    "n_distinct_products": "number of different products bought",
    "mean_interpurchase_days": "average days between two purchase days",
    "return_rate": "share of the bought value that was returned",
    "non_uk_share": "share of spend shipped outside the UK",
}
STAR_SIZE = 26


def feature_label(feature: str, with_unit: bool = True) -> str:
    label, unit, _ = FEATURE_INFO.get(feature, (feature.replace("_", " ").capitalize(), "", ""))
    return f"{label} ({unit})" if with_unit and unit else label


def _empty(df) -> bool:
    return df is None or len(df) == 0


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _names(names: Mapping[int, str] | None, df: pd.DataFrame | None = None) -> dict[int, str]:
    """{id: name}; falls back to the frame's segment / segment_name columns."""
    if names:
        return {int(k): str(v) for k, v in sorted(names.items())}
    if df is not None and {"segment", "segment_name"} <= set(df.columns):
        return VC.segment_names({}, df)
    if df is not None and "segment" in df.columns:
        return {int(s): f"Segment {int(s)}" for s in sorted(df["segment"].unique())}
    return {}


def _pct_axis(explained: Sequence[float] | None, i: int) -> str:
    if explained is not None and i < len(explained) and np.isfinite(_num(explained[i])):
        return f"PC{i + 1} ({_num(explained[i]):.0%} of variance)"
    return f"PC{i + 1}"


# ================================================================== PCA
def sample_points(df: pd.DataFrame, n: int = MAX_POINTS, seed: int = SEED) -> pd.DataFrame:
    """Stratified (per segment, proportional) sample of at most n rows; seed 42."""
    if _empty(df) or len(df) <= n:
        return df
    if "segment" not in df.columns:
        return df.sample(n=n, random_state=seed)
    frac = n / len(df)
    out = df.groupby("segment", group_keys=False).sample(frac=frac, random_state=seed)
    return out.head(n)


def pca_scatter_3d(
    segments: pd.DataFrame | None,
    names: Mapping[int, str] | None = None,
    explained: Sequence[float] | None = None,
    n: int = MAX_POINTS,
) -> go.Figure:
    """3D PC1-3 scatter, one trace per segment (id order, fixed colours), <= n points."""
    if _empty(segments) or not {"pc1", "pc2", "pc3", "segment"} <= set(segments.columns):
        return VC.empty_figure("No PCA coordinates.", "Customers in PCA space")
    names = _names(names, segments)
    pts = sample_points(segments, n)
    fig = go.Figure()
    for sid, name in names.items():
        d = pts[pts["segment"] == sid]
        if d.empty:
            continue
        colour = P.segment_colour(sid)
        fig.add_trace(
            go.Scatter3d(
                x=d["pc1"],
                y=d["pc2"],
                z=d["pc3"],
                mode="markers",
                name=f"{sid} · {name}",
                marker={
                    "size": 3,
                    "color": colour,
                    "opacity": 0.8,
                    "line": {**P.marker_line(colour), "width": 0},
                },
                customdata=d[["customer_id"]].to_numpy(),
                hovertemplate=(
                    f"Customer %{{customdata[0]}}<br>{name}<br>"
                    "PC1 %{x:.2f} · PC2 %{y:.2f} · PC3 %{z:.2f}<extra></extra>"
                ),
            )
        )
    shown = len(pts)
    title = "Customers in PCA space" + (
        f" ({shown:,} of {len(segments):,} shown, seed {SEED})" if shown < len(segments) else ""
    )
    VC.style(fig, title, height=600, legend_title="Segment")
    fig.update_layout(
        scene={
            "xaxis": {"title": {"text": _pct_axis(explained, 0)}},
            "yaxis": {"title": {"text": _pct_axis(explained, 1)}},
            "zaxis": {"title": {"text": _pct_axis(explained, 2)}},
        },
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        # horizontal legend under the title: on the right it squeezed the scene in a
        # ~480 px column and the axes were cropped
        legend={
            "itemsizing": "constant",
            "orientation": "h",
            "x": 0,
            "y": 1.0,
            "yanchor": "bottom",
            "title": {"text": ""},
        },
        # camera a bit further away and lower: the PC2 title is not clipped at the bottom
        scene_camera={
            "eye": {"x": 2.0, "y": 2.0, "z": 1.2},
            "center": {"x": 0, "y": 0, "z": -0.15},
        },
    )
    return fig


def pca_scatter_2d(
    segments: pd.DataFrame | None,
    names: Mapping[int, str] | None = None,
    explained: Sequence[float] | None = None,
    n: int = MAX_POINTS,
    star: Sequence[float] | None = None,
    star_segment: int | None = None,
    star_label: str = "New customer",
    height: int = 520,
) -> go.Figure:
    """PC1 vs PC2 scatter, one trace per segment (id order, fixed colours), <= n points.

    With `star` (pc1, pc2[, pc3]) the real customers are faded and the new customer is a big
    ★ in its segment's colour (dark outline), so it stands out on any background.
    """
    # short star title: the Home card is ~230 px wide at a 1280 px window
    title = "PCA map: ★ = this customer" if star is not None else "PCA map"
    if _empty(segments) or not {"pc1", "pc2", "segment"} <= set(segments.columns):
        return VC.empty_figure("No PCA coordinates.", title)
    names = _names(names, segments)
    pts = sample_points(segments, n)
    faded = star is not None
    fig = go.Figure()
    for sid, name in names.items():
        d = pts[pts["segment"] == sid]
        if d.empty:
            continue
        colour = P.segment_colour(sid)
        custom = d[["customer_id"]].to_numpy() if "customer_id" in d else None
        fig.add_trace(
            go.Scatter(
                x=d["pc1"],
                y=d["pc2"],
                mode="markers",
                name=f"{sid} · {name}",
                marker={
                    "size": 4 if faded else 5,
                    "color": colour,
                    "opacity": 0.25 if faded else 0.7,
                    "line": {**P.marker_line(colour), "width": 0},
                },
                customdata=custom,
                hovertemplate=(
                    (f"Customer %{{customdata[0]}}<br>{name}" if custom is not None else name)
                    + "<br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>"
                ),
            )
        )
    if star is not None:
        colour = P.segment_colour(star_segment) if star_segment is not None else P.INK
        seg_name = names.get(int(star_segment), "") if star_segment is not None else ""
        fig.add_trace(
            go.Scatter(
                x=[float(star[0])],
                y=[float(star[1])],
                mode="markers",
                name=f"★ {star_label}",
                marker={
                    "symbol": "star",
                    "size": STAR_SIZE,
                    "color": colour,
                    "line": {"color": P.INK, "width": 1.5},
                },
                hovertemplate=f"{star_label}<br>{seg_name}<br>PC1 %{{x:.2f}} · PC2 %{{y:.2f}}"
                "<extra></extra>",
            )
        )
    shown = len(pts)
    if shown < len(segments) and star is None:
        title += f": {shown:,} of {len(segments):,} customers (seed {SEED})"
    VC.style(fig, title, _pct_axis(explained, 0), _pct_axis(explained, 1), height=height)
    fig.update_layout(
        legend={"orientation": "h", "y": -0.18, "x": 0, "yanchor": "top", "itemsizing": "constant"}
    )
    return fig


def pc_meanings(
    loadings: pd.DataFrame | None, explained: Sequence[float] | None = None, top: int = 3
) -> list[str]:
    """One plain line per PC: its share of variance and the `top` features that drive it.

    A PC's sign is arbitrary, so the line says what a HIGH score means ("higher X, lower Y").
    """
    if _empty(loadings) or "feature" not in loadings.columns:
        return []
    pcs = [c for c in ("pc1", "pc2", "pc3") if c in loadings.columns]
    lines = []
    for i, pc in enumerate(pcs):
        w = loadings.set_index("feature")[pc].astype(float)
        strongest = w.reindex(w.abs().sort_values(ascending=False).index).head(top)
        parts = [
            f"{'higher' if v > 0 else 'lower'} {FEATURE_WHAT.get(f, f.replace('_', ' '))}"
            for f, v in strongest.items()
        ]
        share = (
            f" ({_num(explained[i]):.0%} of variance)"
            if explained is not None and i < len(explained) and np.isfinite(_num(explained[i]))
            else ""
        )
        lines.append(f"**PC{i + 1}**{share}: a high score means " + ", ".join(parts) + ".")
    return lines


def pca_biplot(
    loadings: pd.DataFrame | None,
    segments: pd.DataFrame | None = None,
    explained: Sequence[float] | None = None,
    n: int = MAX_POINTS,
) -> go.Figure:
    """PC1 vs PC2 loading arrows (one per feature) over a light grey cloud of customers."""
    if _empty(loadings) or not {"feature", "pc1", "pc2"} <= set(loadings.columns):
        return VC.empty_figure("No PCA loadings.", "PCA biplot")
    fig = go.Figure()
    scale = 1.0
    y_range = None
    if not _empty(segments) and {"pc1", "pc2"} <= set(segments.columns):
        pts = sample_points(segments, n)
        fig.add_trace(
            go.Scatter(
                x=pts["pc1"],
                y=pts["pc2"],
                mode="markers",
                marker={"color": P.LIGHT_GREY, "size": 4, "opacity": 0.45},
                name=f"Customers (sample of {len(pts):,})",
                hoverinfo="skip",
            )
        )
        reach = np.nanpercentile(np.abs(pts[["pc1", "pc2"]].to_numpy()), 95)
        longest = np.nanmax(np.hypot(loadings["pc1"], loadings["pc2"]))
        if np.isfinite(reach) and longest > 0:
            scale = float(reach / longest)
        y_range = float(np.nanmax(pts["pc2"]) - np.nanmin(pts["pc2"]))
    for _, r in loadings.iterrows():
        x, y = float(r["pc1"]) * scale, float(r["pc2"]) * scale
        fig.add_annotation(
            x=x,
            y=y,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=P.BLUE,
            text="",
        )
    fig.add_trace(
        go.Scatter(
            x=loadings["pc1"] * scale,
            y=loadings["pc2"] * scale,
            mode="markers",
            text=[feature_label(f, False) for f in loadings["feature"]],
            marker={"color": P.BLUE, "size": 7, "line": P.marker_line(P.BLUE)},
            customdata=loadings[["pc1", "pc2"]].to_numpy(),
            name="Feature loadings (arrows)",
            hovertemplate=(
                "%{text}<br>loading PC1 %{customdata[0]:+.2f} · PC2 %{customdata[1]:+.2f}"
                "<extra></extra>"
            ),
        )
    )
    _label_tips(fig, loadings, scale, y_range)
    fig.add_hline(y=0, line={"color": P.GREY, "width": 1})
    fig.add_vline(x=0, line={"color": P.GREY, "width": 1})
    note = f" (arrows × {scale:.1f})" if scale != 1.0 else ""
    VC.style(
        fig,
        f"PCA biplot: what drives PC1 and PC2{note}",
        _pct_axis(explained, 0),
        _pct_axis(explained, 1),
        height=520,
    )
    fig.update_layout(legend={"orientation": "h", "y": -0.18, "x": 0})
    return fig


def _spread(values: np.ndarray, gap: float) -> np.ndarray:
    """Push sorted label positions apart so neighbours are at least `gap` apart."""
    order = np.argsort(values)
    out = values.astype(float).copy()
    for a, b in zip(order[:-1], order[1:], strict=True):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _label_tips(
    fig: go.Figure, loadings: pd.DataFrame, scale: float, y_range: float | None = None
) -> None:
    """Feature names next to the arrow tips; labels on each side are spread vertically.

    The gap is a share of the whole plotted y range (≈ one text line at the figure height),
    so labels of arrows that point the same way do not overlap.
    """
    x = loadings["pc1"].to_numpy(dtype=float) * scale
    y = loadings["pc2"].to_numpy(dtype=float) * scale
    span = max(float(np.ptp(y)) if len(y) > 1 else 1.0, float(np.abs(y).max()), 1e-9)
    if y_range is not None and np.isfinite(y_range):
        span = max(span, y_range)
    y_lab = y.copy()
    for side in (x >= 0, x < 0):
        if side.any():
            y_lab[side] = _spread(y[side], gap=0.06 * span)
    for f, xi, yl in zip(loadings["feature"], x, y_lab, strict=True):
        fig.add_annotation(
            x=xi,
            y=yl,
            text=feature_label(f, False),
            showarrow=False,
            xanchor="left" if xi >= 0 else "right",
            xshift=6 if xi >= 0 else -6,
            font={"color": P.INK, "size": 12},
            bgcolor="rgba(255,255,255,0.7)",
        )


def scree_plot(explained: Sequence[float] | None) -> go.Figure:
    """Explained variance per PC (bars) and cumulative (line) on ONE 0-100% axis."""
    ev = [_num(v) for v in (explained or [])]
    ev = [v for v in ev if np.isfinite(v)]
    if not ev:
        return VC.empty_figure("No explained variance recorded.", "Scree plot")
    pcs = [f"PC{i + 1}" for i in range(len(ev))]
    cum = np.minimum(np.cumsum(ev), 1.0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pcs,
            y=ev,
            name="Explained by this PC",
            marker={"color": P.BLUE},
            width=0.6,
            hovertemplate="%{x}: %{y:.1%} of variance<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pcs,
            y=cum,
            mode="lines+markers+text",
            name="Cumulative",
            line={"color": P.GREY, "width": 2},
            marker={"color": P.GREY, "size": 7, "line": P.marker_line(P.GREY)},
            text=[f"{c:.0%}" if i < 4 else "" for i, c in enumerate(cum)],
            textposition="top center",
            textfont={"color": P.INK, "size": 11},
            hovertemplate="PC1–%{x}: %{y:.1%} cumulative<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
    VC.style(fig, "Scree plot", "Principal component", "Variance explained (%)", height=420)
    # legend under the axis title: above the plot it ran into the title in a narrow column
    fig.update_layout(legend={"orientation": "h", "y": -0.25, "x": 0, "yanchor": "top"})
    return fig


# ================================================================== k selection
K_PANELS = (
    ("wcss", "WCSS / inertia (lower; look for the elbow)", ",.0f"),
    ("silhouette", "Silhouette (higher is better)", ".3f"),
    ("davies_bouldin", "Davies-Bouldin (lower is better)", ".2f"),
    ("calinski_harabasz", "Calinski-Harabasz (higher is better)", ",.0f"),
    ("ari_mean", "Stability: mean ARI ± sd (higher is better)", ".3f"),
    ("min_segment_share", "Smallest segment (share of customers)", ".1%"),
)
CHOSEN_COLOUR = P.VERMILLION


def k_selection_panels(
    ksel: pd.DataFrame | None,
    chosen_k: int | None = None,
    metrics: Sequence[str] | None = None,
) -> go.Figure:
    """Small multiples of every k-selection metric present; the chosen k marked on each.

    metrics: only these K_PANELS columns, in this order (v2: ("wcss", "silhouette")).
    """
    if _empty(ksel) or "k" not in ksel.columns:
        return VC.empty_figure("No k-selection results.", "Choosing k")
    k = ksel.sort_values("k")
    wanted = K_PANELS if metrics is None else [p for m in metrics for p in K_PANELS if p[0] == m]
    panels = [p for p in wanted if p[0] in k.columns and k[p[0]].notna().any()]
    if not panels:
        return VC.empty_figure("No k-selection metrics.", "Choosing k")
    cols = 3 if len(panels) > 4 else min(2, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.08,
        vertical_spacing=0.2,
    )
    legend_done = False
    for i, (col, _title, fmt) in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        d = k[["k", col] + (["ari_std"] if col == "ari_mean" and "ari_std" in k else [])].dropna(
            subset=[col]
        )
        err = (
            {"type": "data", "array": d["ari_std"].fillna(0), "color": P.GREY, "thickness": 1}
            if col == "ari_mean" and "ari_std" in d
            else None
        )
        fig.add_trace(
            go.Scatter(
                x=d["k"],
                y=d[col],
                mode="lines+markers",
                line={"color": P.BLUE, "width": 2},
                marker={"color": P.BLUE, "size": 7, "line": P.marker_line(P.BLUE)},
                error_y=err,
                name="Score per k",
                legendgroup="score",
                showlegend=i == 0,
                hovertemplate=f"k = %{{x}}<br>%{{y:{fmt}}}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        if chosen_k is not None:
            fig.add_vline(
                x=chosen_k, line={"color": CHOSEN_COLOUR, "width": 1, "dash": "dash"}, row=r, col=c
            )
            hit = d[d["k"] == chosen_k]
            if not hit.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hit["k"],
                        y=hit[col],
                        mode="markers",
                        marker={
                            "symbol": "diamond",
                            "size": 13,
                            "color": CHOSEN_COLOUR,
                            "line": P.marker_line(CHOSEN_COLOUR),
                        },
                        name=f"Chosen k = {chosen_k}",
                        legendgroup="chosen",
                        showlegend=not legend_done,
                        hovertemplate=f"chosen k = %{{x}}<br>%{{y:{fmt}}}<extra></extra>",
                    ),
                    row=r,
                    col=c,
                )
                legend_done = True
        if col == "min_segment_share":
            fig.update_yaxes(tickformat=".0%", row=r, col=c)
    fig.update_xaxes(dtick=1, range=[k["k"].min() - 0.5, k["k"].max() + 0.5])
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text="k (number of segments)", row=rows, col=c)
    k_lo, k_hi = int(k["k"].min()), int(k["k"].max())
    which = (
        "every metric"
        if metrics is None
        else " and ".join(p[0].upper() if p[0] == "wcss" else p[0] for p in panels)
    )
    height = 300 * rows + 80 if rows > 1 else 420
    VC.style(fig, f"Choosing k: {which} for k = {k_lo} … {k_hi}", height=height)
    for ann in fig.layout.annotations:
        if ann.text in {p[1] for p in panels}:
            ann.font = {"color": P.MUTED_INK, "size": 12}
    legend_y = -0.08 if rows > 1 else -0.28
    fig.update_layout(legend={"orientation": "h", "y": legend_y, "x": 0, "yanchor": "top"})
    return fig


# ================================================================== profiles
def profile_cards(profiles: pd.DataFrame | None, names: Mapping[int, str] | None = None) -> list:
    """One dict per segment (id order): id, name, colour and (label, value) lines."""
    if _empty(profiles):
        return []
    names = _names(names, profiles)
    p = profiles.set_index("segment")
    cards = []
    for sid, name in names.items():
        if sid not in p.index:
            continue
        r = p.loc[sid]
        lines = [
            ("Customers", f"{int(r['n_customers']):,} ({VC.pct(_num(r['customer_share']))})"),
            ("Revenue share", VC.pct(_num(r["revenue_share"]))),
        ]
        if "bought_again_share" in p.columns:
            lines.append(
                (f"Bought again ({HORIZON_DAYS} days)", VC.pct(_num(r["bought_again_share"])))
            )
        lines += [
            ("Mean recency", f"{_num(r['recency']):,.0f} days"),
            ("Mean frequency", f"{_num(r['frequency']):,.1f} orders"),
            ("Mean spend", VC.money(_num(r["monetary"]))),
        ]
        if "median_monetary" in p.columns:
            lines.append(("Median spend", VC.money(_num(r["median_monetary"]))))
        lines.append(("Mean basket", VC.money(_num(r["avg_basket_value"]))))
        if "uk_customer_share" in p.columns:
            lines.append(("UK customers", VC.pct(_num(r["uk_customer_share"]))))
        cards.append(
            {"segment": sid, "name": name, "colour": P.segment_colour(sid), "lines": lines}
        )
    return cards


SHARE_MEASURES = (  # (column, legend label, pattern, opacity): measure = texture, segment = hue
    ("customer_share", "Share of customers", "/", 0.45),
    ("revenue_share", "Share of revenue", "", 1.0),
)


def share_bars(profiles: pd.DataFrame | None, names: Mapping[int, str] | None = None) -> go.Figure:
    """Customer % vs revenue % per segment: grouped bars in the segment's colour; customers
    hatched + light, revenue solid (the legend names both, so colour never works alone)."""
    title = "Share of customers vs share of revenue"
    cols = {"segment", "customer_share", "revenue_share"}
    if _empty(profiles) or not cols <= set(profiles.columns):
        return VC.empty_figure("No segment profiles.", title)
    names = _names(names, profiles)
    p = profiles.assign(segment=profiles["segment"].astype(int)).set_index("segment")
    ids = [s for s in names if s in p.index]
    labels = [f"{s} · {names[s]}" for s in ids]
    colours = [P.segment_colour(s) for s in ids]
    fig = go.Figure()
    for col, label, pattern, opacity in SHARE_MEASURES:
        vals = [float(p.at[s, col]) for s in ids]
        pattern_spec = {"shape": pattern, "fgcolor": P.INK, "size": 6, "solidity": 0.2}
        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                name=label,
                legendgroup=label,
                offsetgroup=col,
                showlegend=False,  # a multi-colour bar's legend swatch shows only one colour
                marker={
                    "color": colours,
                    "opacity": opacity,
                    "pattern": pattern_spec,
                    "line": {"color": colours, "width": 1.5},
                },
                text=[f"{v:.0%}" for v in vals],
                textposition="outside",
                textfont={"color": P.INK},
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.1%}}<extra></extra>",
            )
        )
        # legend-only swatch in neutral grey: the measure is the texture + lightness (rgba,
        # not opacity: legend swatches ignore marker opacity, so both looked the same grey)
        swatch = P.GREY if opacity >= 1 else _rgba(P.GREY, opacity)
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=label,
                legendgroup=label,
                offsetgroup=col,  # same slot as its real bars: no empty gap in the groups
                marker={"color": swatch, "pattern": pattern_spec},
                hoverinfo="skip",
            )
        )
    top = max(float(p[["customer_share", "revenue_share"]].to_numpy().max()), 0.01)
    fig.update_yaxes(tickformat=".0%", range=[0, min(1.0, top * 1.18) + 0.02])
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    VC.style(fig, title, "Segment", "Share (%)", height=400)
    fig.update_layout(legend={"orientation": "h", "y": 1.02, "x": 0, "yanchor": "bottom"})
    return fig


def segment_feature_frame(
    features: pd.DataFrame | None, segments: pd.DataFrame | None
) -> pd.DataFrame:
    """features_cutoff joined to customer segments (inner join on customer_id)."""
    if _empty(features) or _empty(segments):
        return pd.DataFrame(columns=["customer_id", "segment", *C.FEATURE_COLUMNS])
    return features.merge(segments[["customer_id", "segment"]], on="customer_id", how="inner")


def segment_boxplots(
    joined: pd.DataFrame | None,
    feature: str,
    names: Mapping[int, str] | None = None,
    log: bool | None = None,
) -> go.Figure:
    """One box per segment (id order, fixed colours); log axis when the feature is skewed."""
    if _empty(joined) or feature not in joined.columns:
        return VC.empty_figure("No customer features.", feature_label(feature))
    names = _names(names, joined)
    vals = joined[feature]
    use_log = VC.is_skewed(vals) if log is None else (log and bool((vals > 0).all()))
    fig = go.Figure()
    for sid, name in names.items():
        v = joined.loc[joined["segment"] == sid, feature]
        if v.empty:
            continue
        colour = P.segment_colour(sid)
        fig.add_trace(
            go.Box(
                y=v,
                name=VC.wrap_label(f"{sid} · {name}"),
                marker={"color": colour, "size": 3, "opacity": 0.5},
                line={"color": colour, "width": 1.5},
                fillcolor=_rgba(colour, 0.25),
                boxpoints="outliers",
                boxmean=True,
                hoverinfo="y",
            )
        )
    label = feature_label(feature)
    VC.style(fig, f"{feature_label(feature, False)} by segment", "Segment", label, height=440)
    if use_log:
        VC.log_axis(fig, "y", label, dtick=1)  # decades only: "2 / 5" labels crowded it
    # no legend: the x labels already name each segment (a legend squeezed a half-width
    # column and forced rotated labels); long names wrap instead of rotating
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=0)
    return fig


def parallel_profiles(
    profiles: pd.DataFrame | None,
    names: Mapping[int, str] | None = None,
    features: Sequence[str] | None = None,
) -> go.Figure:
    """Parallel coordinates of segment means, min-max scaled per feature (0 = lowest segment,
    1 = highest); hover shows the real mean. One line per segment, legend + end labels."""
    if _empty(profiles):
        return VC.empty_figure("No segment profiles.", "Segment profiles")
    names = _names(names, profiles)
    feats = [f for f in (features or C.FEATURE_COLUMNS) if f in profiles.columns]
    if not feats:
        return VC.empty_figure("No profile features.", "Segment profiles")
    p = profiles.set_index("segment")
    means = p[feats].astype(float)
    lo, hi = means.min(), means.max()
    span = (hi - lo).replace(0, np.nan)
    scaled = ((means - lo) / span).fillna(0.5)
    xlab = [feature_label(f, False) for f in feats]
    fig = go.Figure()
    for sid, name in names.items():
        if sid not in scaled.index:
            continue
        colour = P.segment_colour(sid)
        fig.add_trace(
            go.Scatter(
                x=xlab,
                y=scaled.loc[sid],
                mode="lines+markers",
                name=f"{sid} · {name}",
                line={"color": colour, "width": 2},
                marker={"color": colour, "size": 9, "line": P.marker_line(colour)},
                customdata=means.loc[sid].to_numpy()[:, None],
                hovertemplate=f"{name}<br>%{{x}}: mean %{{customdata[0]:,.2f}}<extra></extra>",
            )
        )
    fig.update_yaxes(
        range=[-0.08, 1.08], tickvals=[0, 0.5, 1], ticktext=["lowest", "middle", "highest"]
    )
    VC.style(
        fig,
        "Segment means across features (scaled per feature)",
        "Feature",
        "Scaled segment mean",
        height=540,
        legend_title="Segment",
    )
    fig.update_xaxes(tickangle=-35)
    # legend at the bottom of the figure (a side legend squeezes the half-width plot)
    fig.update_layout(
        legend={
            "orientation": "h",
            "x": 0,
            "y": 0,
            "yref": "container",
            "yanchor": "bottom",
            "title_text": "",
        },
        margin={"b": 200},
    )
    return fig


# ================================================================== baseline + chi-square
BASELINE_ROWS = (
    ("silhouette", "Silhouette (higher = better separated)", "{:.3f}"),
    ("davies_bouldin", "Davies-Bouldin (lower = better)", "{:.2f}"),
    ("calinski_harabasz", "Calinski-Harabasz (higher = better)", "{:,.0f}"),
    ("auc_returned", "CV ROC-AUC: predicts 'bought again'", "{:.3f}"),
)


def baseline_table(baseline: Mapping | None) -> pd.DataFrame:
    """Rows = metrics, columns = K-Means / RFM quartiles (strings; '± sd' on the AUC)."""
    b = baseline or {}
    methods = [("kmeans", "K-Means segments"), ("rfm_quartile", "RFM-quartile baseline")]
    methods = [(k, lab) for k, lab in methods if isinstance(b.get(k), Mapping)]
    if not methods:
        return pd.DataFrame()
    rows = []
    for key, label, fmt in BASELINE_ROWS:
        row = {"Metric": label}
        any_value = False
        for m, mlab in methods:
            v = _num(b[m].get(key))
            if np.isfinite(v):
                any_value = True
                txt = fmt.format(v)
                sd = _num(b[m].get(f"{key}_std"))
                row[mlab] = txt + (f" ± {sd:.3f}" if np.isfinite(sd) else "")
            else:
                row[mlab] = "n/a"
        if any_value:
            rows.append(row)
    return pd.DataFrame(rows)


def uk_share_bar(chi2: Mapping | None, names: Mapping[int, str] | None = None) -> go.Figure:
    """Share of UK customers per segment (from chi2_segment_uk.uk_share_by_segment)."""
    shares = (chi2 or {}).get("uk_share_by_segment") or {}
    if not shares:
        return VC.empty_figure("UK share by segment not recorded.", "UK customers by segment")
    shares = {int(k): _num(v) for k, v in shares.items()}
    names = _names(names) or {s: f"Segment {s}" for s in shares}
    ids = [s for s in names if s in shares]
    fig = go.Figure(
        go.Bar(
            x=[f"{s} · {names[s]}" for s in ids],
            y=[shares[s] for s in ids],
            marker={"color": [P.segment_colour(s) for s in ids]},
            width=0.55,
            text=[f"{shares[s]:.1%}" for s in ids],
            textposition="outside",
            textfont={"color": P.INK},
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:.1%} UK customers<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
    return VC.style(fig, "UK customers per segment", "Segment", "UK customers (%)", height=340)


# ================================================================== stats section
QQ_LABELS = {
    "avg_basket_value": "Average basket value (£)",
    "repeat_orders": "Repeat orders (orders after the first)",
    "interpurchase_days": "Days between purchases",
}


def qq_plot(qq: pd.DataFrame | None, variable: str) -> go.Figure:
    """Q-Q plots of one variable, one panel per fitted distribution, with the 45° line."""
    if _empty(qq) or variable not in set(qq["variable"]):
        return VC.empty_figure("No Q-Q points.", "Q-Q plot")
    d = qq[qq["variable"] == variable]
    dists = list(dict.fromkeys(d["dist"]))
    fig = make_subplots(rows=1, cols=len(dists), subplot_titles=[f"vs {x}" for x in dists])
    label = QQ_LABELS.get(variable, variable.replace("_", " "))
    for i, dist in enumerate(dists, start=1):
        g = d[d["dist"] == dist].sort_values("prob")
        pos = bool((g[["theoretical", "sample"]] > 0).all().all())
        use_log = pos and VC.is_skewed(g["sample"])
        lo = float(np.nanmin(g[["theoretical", "sample"]].to_numpy()))
        hi = float(np.nanmax(g[["theoretical", "sample"]].to_numpy()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line={"color": P.GREY, "width": 1, "dash": "dash"},
                name="Perfect fit (45° line)",
                legendgroup="ref",
                showlegend=i == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=g["theoretical"],
                y=g["sample"],
                mode="markers",
                marker={"color": P.BLUE, "size": 5, "line": P.marker_line(P.BLUE)},
                name="Quantiles",
                legendgroup="pts",
                showlegend=i == 1,
                customdata=g[["prob"]].to_numpy(),
                hovertemplate=(
                    f"{dist}: quantile %{{customdata[0]:.1%}}<br>"
                    "theoretical %{x:,.2f}<br>sample %{y:,.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=i,
        )
        if use_log:
            fig.update_xaxes(type="log", dtick=1, row=1, col=i)
            fig.update_yaxes(type="log", dtick=1, row=1, col=i)
        fig.update_xaxes(
            title_text="Theoretical quantile" + (" (log scale)" if use_log else ""), row=1, col=i
        )
        fig.update_yaxes(
            title_text="Sample quantile" + (" (log scale)" if use_log else ""), row=1, col=i
        )
    VC.style(fig, f"Q-Q plot: {label}", height=400)
    for ann in fig.layout.annotations:
        ann.font = {"color": P.MUTED_INK, "size": 13}
    fig.update_layout(legend={"orientation": "h", "y": -0.25, "x": 0})
    return fig


def frequency_fit(ff: pd.DataFrame | None) -> go.Figure:
    """Observed customers per number of repeat orders (bars) vs Poisson and NB fits (lines)."""
    if _empty(ff):
        return VC.empty_figure("No frequency fit.", "Poisson vs negative binomial")
    f = ff.sort_values("k")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=f["k"],
            y=f["observed"],
            name="Observed",
            marker={"color": P.LIGHT_GREY},
            hovertemplate="%{x} repeat orders: %{y:,.0f} customers observed<extra></extra>",
        )
    )
    for col, label, colour, dash in (
        ("poisson", "Poisson fit", P.ORANGE, "dash"),
        ("nbinom", "Negative binomial fit", P.BLUE, "solid"),
    ):
        if col in f.columns:
            fig.add_trace(
                go.Scatter(
                    x=f["k"],
                    y=f[col],
                    mode="lines+markers",
                    name=label,
                    line={"color": colour, "width": 2, "dash": dash},
                    marker={"color": colour, "size": 6, "line": P.marker_line(colour)},
                    hovertemplate=f"%{{x}} repeat orders: %{{y:,.1f}} expected ({label})"
                    "<extra></extra>",
                )
            )
    VC.style(
        fig,
        "Repeat orders per customer: Poisson vs negative binomial",
        "Repeat orders (orders after the first)",
        "Customers",
        height=400,
    )
    fig.update_layout(legend={"x": 0.98, "y": 0.98, "xanchor": "right"}, bargap=0.15)
    return fig


def correlation_heatmaps(
    corr: pd.DataFrame | None, methods: Sequence[str] = ("pearson", "spearman")
) -> go.Figure:
    """Pearson vs Spearman side by side on ONE diverging scale (-1 … 1, grey at 0)."""
    if _empty(corr):
        return VC.empty_figure("No correlation matrix.", "Correlation")
    methods = [m for m in methods if m in set(corr["method"])]
    if not methods:
        return VC.empty_figure("No correlation matrix.", "Correlation")
    order = list(dict.fromkeys(corr["x"]))
    labels = [feature_label(f, False) for f in order]
    fig = make_subplots(
        rows=1,
        cols=len(methods),
        subplot_titles=[m.capitalize() for m in methods],
        horizontal_spacing=0.12,
    )
    for i, m in enumerate(methods, start=1):
        z = (
            corr[corr["method"] == m]
            .pivot_table(index="y", columns="x", values="value", aggfunc="first")
            .reindex(index=order, columns=order)
        )
        fig.add_trace(
            go.Heatmap(
                z=z.to_numpy(),
                x=labels,
                y=labels,
                coloraxis="coloraxis",
                texttemplate="%{z:.2f}",
                textfont={"size": 10},
                xgap=1,
                ygap=1,
                hovertemplate=f"{m.capitalize()}<br>%{{y}} × %{{x}}: %{{z:.2f}}<extra></extra>",
            ),
            row=1,
            col=i,
        )
        fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=i)
        fig.update_xaxes(tickangle=-40, showgrid=False, row=1, col=i)
        if i > 1:
            fig.update_yaxes(showticklabels=False, row=1, col=i)
    fig.update_layout(
        coloraxis={
            "colorscale": P.DIVERGING,
            "cmin": -1,
            "cmax": 1,
            "cmid": 0,
            "colorbar": {"title": {"text": "r"}},
        }
    )
    VC.style(fig, "Correlation of customer features: Pearson vs Spearman", height=520)
    for ann in fig.layout.annotations:
        ann.font = {"color": P.MUTED_INK, "size": 13}
    return fig


def histogram_polygon(hist: pd.DataFrame | None, feature: str) -> go.Figure:
    """Histogram bars + frequency polygon of one feature.

    Log-spaced bins (scale "log1p") are drawn at log10 positions with real-unit ticks, so
    every bar has the same visual width (plotly bars cannot span log-axis intervals).
    """
    if _empty(hist) or feature not in set(hist["feature"]):
        return VC.empty_figure("No histogram.", feature_label(feature))
    h = hist[hist["feature"] == feature].sort_values("bin")
    log = (h["scale"] == "log1p").any() and bool((h["left"] > 0).all())
    if log:
        left, right, mid = np.log10(h["left"]), np.log10(h["right"]), np.log10(h["mid"])
    else:
        left, right, mid = h["left"], h["right"], h["mid"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=(left + right) / 2,
            y=h["count"],
            width=(right - left) * 0.95,
            name="Histogram",
            marker={"color": P.LIGHT_GREY, "line": {"width": 0}},
            customdata=h[["left", "right"]].to_numpy(),
            hovertemplate="%{customdata[0]:,.1f} – %{customdata[1]:,.1f}: "
            "%{y:,.0f} customers<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mid,
            y=h["count"],
            mode="lines+markers",
            name="Frequency polygon",
            line={"color": P.BLUE, "width": 2},
            marker={"color": P.BLUE, "size": 6, "line": P.marker_line(P.BLUE)},
            customdata=h[["mid"]].to_numpy(),
            hovertemplate="bin centre %{customdata[0]:,.1f}: %{y:,.0f} customers<extra></extra>",
        )
    )
    label = feature_label(feature)
    x_title = f"{label} (log scale)" if log else label
    VC.style(
        fig, f"Distribution: {feature_label(feature, False)}", x_title, "Customers", height=380
    )
    if log:
        ticks = np.arange(np.floor(left.min()), np.ceil(right.max()) + 1)
        fig.update_xaxes(tickvals=ticks, ticktext=[f"{10**t:,.0f}" for t in ticks])
    fig.update_layout(legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"}, bargap=0)
    return fig


# ================================================================== live test picker
def compare_segments(
    joined: pd.DataFrame | None,
    seg_a: int,
    seg_b: int,
    feature: str,
    names: Mapping[int, str] | None = None,
) -> dict:
    """Welch t vs Mann-Whitney U on one feature between two segments (stats.tests)."""
    names = _names(names, joined)
    name_a, name_b = names.get(seg_a, f"segment {seg_a}"), names.get(seg_b, f"segment {seg_b}")
    if _empty(joined) or feature not in joined.columns:
        return {"conclusion": "No customer features available for this test.", "groups": {}}
    if seg_a == seg_b:
        return {
            "conclusion": f"Pick two different segments ({name_a} was chosen twice).",
            "groups": {},
        }
    a = joined.loc[joined["segment"] == seg_a, feature]
    b = joined.loc[joined["segment"] == seg_b, feature]
    if len(a.dropna()) < 2 or len(b.dropna()) < 2:
        return {
            "conclusion": f"Not enough customers to compare: {len(a)} in {name_a}, "
            f"{len(b)} in {name_b} (at least 2 each).",
            "groups": {},
        }
    _, _, unit = FEATURE_INFO.get(feature, ("", "", ""))
    return compare_groups(
        a,
        b,
        name_a=f"'{name_a}' customers",
        name_b=f"'{name_b}' customers",
        what=FEATURE_WHAT.get(feature, feature.replace("_", " ")),
        unit=unit,
    )


def result_table(result: Mapping | None) -> pd.DataFrame:
    """Welch t and Mann-Whitney U side by side: statistic, p, effect size and its label."""
    r = result or {}
    rows = []
    t = r.get("welch_t")
    if t:
        rows.append(
            {
                "Test": "Welch's t-test (means)",
                "Statistic": f"t = {t['t']:+.2f} (df {t['df']:,.0f})",
                "p-value": fmt_p(_num(t["p_value"])),
                "Effect size": f"Cohen's d = {t['cohens_d']:+.2f}",
                "Effect": t.get("effect", ""),
            }
        )
    u = r.get("mann_whitney")
    if u:
        rows.append(
            {
                "Test": "Mann-Whitney U (ranks)",
                "Statistic": f"U = {u['U']:,.0f}",
                "p-value": fmt_p(_num(u["p_value"])),
                "Effect size": f"rank-biserial r = {u['rank_biserial']:+.2f}",
                "Effect": u.get("effect", ""),
            }
        )
    return pd.DataFrame(rows)


def two_segment_distribution(
    joined: pd.DataFrame | None,
    seg_a: int,
    seg_b: int,
    feature: str,
    names: Mapping[int, str] | None = None,
) -> go.Figure:
    """Violin + box of one feature for the two compared segments (log axis if skewed)."""
    if _empty(joined) or feature not in joined.columns:
        return VC.empty_figure("No customer features.", feature_label(feature))
    names = _names(names, joined)
    sub = joined[joined["segment"].isin([seg_a, seg_b])]
    if sub.empty:
        return VC.empty_figure("No customers in these segments.", feature_label(feature))
    use_log = VC.is_skewed(sub[feature])
    fig = go.Figure()
    for sid in dict.fromkeys([seg_a, seg_b]):
        v = sub.loc[sub["segment"] == sid, feature]
        if v.empty:
            continue
        colour = P.segment_colour(sid)
        name = f"{sid} · {names.get(sid, sid)}"
        y = np.log10(v) if use_log else v
        fig.add_trace(
            go.Violin(
                y=y,
                name=name,
                line={"color": colour, "width": 1.5},
                fillcolor=_rgba(colour, 0.25),
                box={"visible": True, "width": 0.3},
                meanline={"visible": True},
                points=False,
                spanmode="hard",
                customdata=v.to_numpy()[:, None],
                hoverinfo="name",
            )
        )
    label = feature_label(feature)
    VC.style(
        fig,
        f"{feature_label(feature, False)}: the two segments compared",
        "Segment",
        f"{label} (log10 scale)" if use_log else label,
        height=420,
        legend_title="Segment",
    )
    if use_log:  # violins of log10 values, ticks back in real units
        lo, hi = np.floor(np.log10(sub[feature].min())), np.ceil(np.log10(sub[feature].max()))
        ticks = np.arange(lo, hi + 1)
        fig.update_yaxes(tickvals=ticks, ticktext=[f"{10**t:,.0f}" for t in ticks])
    return fig
