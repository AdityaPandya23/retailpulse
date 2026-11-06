"""Helpers shared by every figure module: segment colours from metrics names, log axes, titles.

Never hard-code k or segment names: build them from metrics.json `segment.segment_names`
(string keys after JSON) via segment_names() / segment_colour_map().
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from retailpulse.viz import palette as P

SKEW_LOG_THRESHOLD = 1.0  # |skewness| above this (and all values > 0) -> log axis


def segment_names(metrics_segment: Mapping | None, profiles: pd.DataFrame | None = None) -> dict:
    """{segment id (int): name}, sorted by id, from metrics["segment"]["segment_names"].

    Falls back to a profiles / segments frame with `segment` + `segment_name` columns.
    """
    raw = (metrics_segment or {}).get("segment_names") or {}
    names = {int(k): str(v) for k, v in raw.items()}
    if not names and profiles is not None and {"segment", "segment_name"} <= set(profiles):
        pairs = profiles[["segment", "segment_name"]].drop_duplicates("segment")
        names = {
            int(s): str(n) for s, n in zip(pairs["segment"], pairs["segment_name"], strict=True)
        }
    return dict(sorted(names.items()))


def segment_colour_map(names: Mapping[int, str]) -> dict[str, str]:
    """{segment name: fixed colour of its id} for plotly color_discrete_map."""
    return {str(n): P.segment_colour(int(s)) for s, n in sorted(names.items())}


def segment_order(names: Mapping[int, str]) -> list[str]:
    """Segment names in id order (0 = most valuable first) for category_orders."""
    return [str(names[s]) for s in sorted(names)]


def is_skewed(values, threshold: float = SKEW_LOG_THRESHOLD) -> bool:
    """True when a log axis is the honest choice: strictly positive and right-skewed."""
    v = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 3 or (v <= 0).any():
        return False
    return bool(v.skew() > threshold)


def log_axis(fig: go.Figure, axis: str = "y", title: str | None = None, **kwargs) -> go.Figure:
    """Switch an axis ("x", "y", "y2" or "yaxis2" ...) to log scale with readable ticks."""
    name = axis if "axis" in axis else f"{axis[0]}axis{axis[1:]}"
    update = {"type": "log", "minor": {"ticks": "", "showgrid": False}, **kwargs}
    if title is not None:
        update["title"] = {"text": f"{title} (log scale)"}
    fig.update_layout({name: update})
    return fig


def style(
    fig: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    """Title + axis labels (with units) + optional height + the RetailPulse template.

    The template is set on the figure itself: Streamlit replaces plotly's global default
    template, so render with st.plotly_chart(fig, theme=None) (components.layout.chart).
    """
    fig.update_layout(template=P.template(), title={"text": title})
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    if height is not None:
        fig.update_layout(height=height)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def empty_figure(message: str, title: str = "") -> go.Figure:
    """A blank figure with a centred note (missing data must never crash a page)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": P.MUTED_INK, "size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, title, height=300)


def wrap_label(text, width: int = 14) -> str:
    """Break a category label into <br>-joined lines of <= width characters (word breaks), so
    narrow charts can keep horizontal tick labels."""
    words, lines = str(text).split(), []
    for w in words:
        if lines and len(lines[-1]) + 1 + len(w) <= width:
            lines[-1] += " " + w
        else:
            lines.append(w)
    return "<br>".join(lines) if lines else str(text)


def md_escape(text) -> str:
    """Escape markdown / LaTeX characters (product names) before st.markdown."""
    out = str(text)
    for ch in ("\\", "*", "_", "$", "`", "[", "]", "#", "~", "|", "<", ">"):
        out = out.replace(ch, "\\" + ch)
    return out


def money(x: float, decimals: int = 0) -> str:
    """£1,234 / £1.2M style for labels and hover text."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1e6:
        return f"£{x / 1e6:.{max(decimals, 1)}f}M"
    return f"£{x:,.{decimals}f}"


def num(x, spec: str = ",.2f") -> str:
    """Format a metrics number; None (NaN is stored as null in metrics.json) / NaN -> "n/a"."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return format(v, spec) if np.isfinite(v) else "n/a"


def pct(x: float, decimals: int = 0) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100 * x:.{decimals}f}%"
