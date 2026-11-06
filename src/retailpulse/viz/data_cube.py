"""Figures + small table helpers for page 1 (Data & Cube).

Inputs are the quality JSON dicts, the EDA tables and the OLAP cube (all 8 grouping sets of
CUBE(country, year, month); a NULL dimension means "all"). Every figure function returns a
go.Figure and falls back to common.empty_figure when its input is missing or empty.

Cube rules used here:
* a grouping set is picked by its grouping_id (bit set = dimension rolled up; country = 4,
  year = 2, month = 1), never by summing finer rows;
* revenue and orders are additive (an invoice has one country and one month), distinct
  customers are NOT (one customer can buy in several months / countries), so totals of
  customers always come from the rolled-up row of the right grouping set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from retailpulse import contracts as C
from retailpulse.viz import common as VC
from retailpulse.viz import palette as P

ALL_LABEL = "All countries"
UK = "United Kingdom"
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MEASURE_LABELS = {"revenue": "Revenue (£)", "orders": "Orders", "customers": "Customers"}
ADDITIVE_MEASURES = ("revenue", "orders")

# kept (flowing on) vs removed rows in the cleaning funnel; returns go to their own table
KEPT_COLOUR = P.BLUE
REMOVED_COLOUR = P.GREY
HIGHLIGHT_COLOUR = P.BLUE  # the UK / the subject of a chart
CONTEXT_COLOUR = P.LIGHT_GREY  # everything else
FORMAT_COLOURS = {"xlsx": P.GREY, "parquet": P.BLUE}


def _empty(df) -> bool:
    return df is None or (isinstance(df, pd.DataFrame) and df.empty) or (not len(df))


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ================================================================== data quality
def _step_label(step: str) -> str:
    return str(step).replace("_", " ").strip().capitalize()


def cleaning_sankey(cleaning: dict | None) -> go.Figure:
    """Raw rows -> each step's kept rows -> final sales lines; removed rows to their destination.

    Steps that remove nothing (e.g. description imputation) are not drawn.
    """
    steps = [s for s in (cleaning or {}).get("steps", []) if _num(s.get("removed")) > 0]
    raw = _num((cleaning or {}).get("raw_rows"))
    if not steps or not np.isfinite(raw):
        return VC.empty_figure("No cleaning steps recorded.", "Cleaning funnel")

    labels, colours, links = [f"Raw rows<br>{raw:,.0f}"], [P.INK], []
    xs, ys = [0.001], [0.6]  # removed rows branch off upwards, one column per step
    dest_index: dict[str, int] = {}
    prev = 0
    for i, s in enumerate(steps):
        col_x = min(0.999, (i + 1) / len(steps))
        kept, removed = _num(s.get("rows_out")), _num(s.get("removed"))
        last = i == len(steps) - 1
        labels.append(f"Clean sales lines<br>{kept:,.0f}" if last else f"{kept:,.0f}")
        colours.append(KEPT_COLOUR)
        xs.append(col_x)
        ys.append(0.3 if last else 0.6)
        kept_idx = len(labels) - 1
        links.append((prev, kept_idx, kept, KEPT_COLOUR, _step_label(s.get("step", ""))))
        dest = str(s.get("removed_to") or "Removed")
        if dest not in dest_index:
            labels.append(dest)
            colours.append(REMOVED_COLOUR)
            xs.append(col_x)
            ys.append(0.97 if last else 0.02)  # final split: kept on top, removed below
            dest_index[dest] = len(labels) - 1
        links.append((prev, dest_index[dest], removed, REMOVED_COLOUR, _step_label(s.get("step"))))
        prev = kept_idx
    # destination totals in their labels
    for dest, idx in dest_index.items():
        total = sum(v for _, t, v, _, _ in links if t == idx)
        labels[idx] = f"{dest}: {total:,.0f}"

    src, tgt, val, col, step = zip(*links, strict=True)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": colours,
                "x": xs,
                "y": ys,
                "pad": 18,
                "thickness": 14,
                "line": {"color": P.SURFACE, "width": 1},
                "hovertemplate": "%{label}<extra></extra>",
            },
            link={
                "source": list(src),
                "target": list(tgt),
                "value": list(val),
                "color": [_rgba(c, 0.35) for c in col],
                "customdata": [[s, v / raw] for s, v in zip(step, val, strict=True)],
                "hovertemplate": (
                    "%{customdata[0]}<br>%{value:,.0f} rows "
                    "(%{customdata[1]:.1%} of raw)<extra></extra>"
                ),
            },
            textfont={"color": P.INK, "size": 12},
        )
    )
    return VC.style(fig, "Cleaning funnel: where the raw rows go", height=420)


EFFECT_ROWS = (
    ("mannwhitney_quantity", "rank_biserial", "Quantity per line<br>(rank-biserial r)"),
    ("mannwhitney_price", "rank_biserial", "Unit price<br>(rank-biserial r)"),
    ("chi2_uk", "cramers_v", "Ships to the UK<br>(Cramér's V)"),
)


def missing_id_effects(missing: dict | None) -> go.Figure:
    """Effect sizes of 'Customer ID missing vs present' (signed r, V >= 0) with size labels."""
    tests = (missing or {}).get("tests") or {}
    rows = []
    for key, stat, label in EFFECT_ROWS:
        t = tests.get(key) or {}
        if stat in t and t[stat] is not None:
            if key == "chi2_uk":
                detail = (
                    f"UK share {_num(t.get('share_uk_missing')):.1%} (missing) vs "
                    f"{_num(t.get('share_uk_present')):.1%} (present)"
                )
            else:
                detail = (
                    f"median {_num(t.get('median_missing')):,.2f} (missing) vs "
                    f"{_num(t.get('median_present')):,.2f} (present)"
                )
            rows.append((label, float(t[stat]), str(t.get("effect", "")), detail))
    if not rows:
        return VC.empty_figure("No missing-ID tests recorded.", "Missing Customer ID")
    labels, values, effects, details = zip(*rows, strict=True)
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(labels),
            orientation="h",
            marker={"color": P.BLUE},
            width=0.5,
            text=[f"{v:+.2f} ({e})" for v, e in zip(values, effects, strict=True)],
            textposition="outside",
            textfont={"color": P.INK},
            cliponaxis=False,
            customdata=np.column_stack([effects, details]),
            hovertemplate="%{y}<br>effect %{x:+.2f} (%{customdata[0]})<br>"
            "%{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line={"color": P.GREY, "width": 1})
    fig.update_xaxes(range=[-1, 1], tickformat="+.1f")
    fig.update_yaxes(autorange="reversed")
    return VC.style(
        fig,
        "Rows without a Customer ID differ from the rest",
        "Effect size (−1 … +1; |0.1| small, |0.3| medium, |0.5| large)",
        None,
        height=300,
    )


def format_comparison(fmt: dict | None) -> go.Figure:
    """xlsx vs parquet: file size and load time as two small panels (no shared y-axis)."""
    fmt = fmt or {}
    keys = ("xlsx_mb", "parquet_mb", "xlsx_load_s", "parquet_load_s")
    if not all(np.isfinite(_num(fmt.get(k))) for k in keys):
        return VC.empty_figure("No format comparison recorded.", "xlsx vs parquet")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("File size (MB)", "Load time (seconds, log scale)"),
        horizontal_spacing=0.18,
    )
    for col, (unit, fmt_str, suffix) in enumerate(
        (("mb", "{:,.1f} MB", "MB"), ("load_s", "{:,.2f} s", "s")), start=1
    ):
        vals = [_num(fmt[f"xlsx_{unit}"]), _num(fmt[f"parquet_{unit}"])]
        fig.add_trace(
            go.Bar(
                x=["xlsx", "parquet"],
                y=vals,
                marker={"color": [FORMAT_COLOURS["xlsx"], FORMAT_COLOURS["parquet"]]},
                width=0.55,
                text=[fmt_str.format(v) for v in vals],
                textposition="outside",
                textfont={"color": P.INK},
                cliponaxis=False,
                hovertemplate=f"%{{x}}: %{{y:,.3f}} {suffix}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.update_yaxes(title_text="Megabytes", row=1, col=1, rangemode="tozero")
    VC.log_axis(fig, "yaxis2")
    fig.update_yaxes(title_text="Seconds (log scale)", row=1, col=2)
    VC.style(fig, "Raw file: Excel vs Parquet", height=340)
    for ann in fig.layout.annotations:  # subplot titles in muted ink
        ann.font = {"color": P.MUTED_INK, "size": 13}
    return fig


# ================================================================== EDA
def monthly_trend(monthly: pd.DataFrame | None) -> go.Figure:
    """Monthly revenue and orders as two stacked small multiples (shared x, own y each).

    Partial months (data starts / ends mid-month) are drawn hollow on a dotted segment.
    """
    if _empty(monthly):
        return VC.empty_figure("No monthly data.", "Revenue and orders per month")
    m = monthly.sort_values("month").reset_index(drop=True)
    partial = m["is_partial"].astype(bool).to_numpy()
    near = partial.copy()
    near[1:] |= partial[:-1]
    near[:-1] |= partial[1:]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Revenue (£)", "Orders (distinct invoices)"),
    )
    specs = (
        ("revenue", "£%{y:,.0f}", 1),
        ("orders", "%{y:,.0f} orders", 2),
    )
    for measure, fmt, row in specs:
        y = m[measure].astype(float)
        hover = f"%{{x|%b %Y}}<br>{fmt}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=m["month"],
                y=y.where(~partial),
                mode="lines+markers",
                line={"color": P.BLUE, "width": 2},
                marker={"color": P.BLUE, "size": 7, "line": P.marker_line(P.BLUE)},
                name="Complete month",
                legendgroup="complete",
                showlegend=row == 1,
                hovertemplate=hover,
            ),
            row=row,
            col=1,
        )
        if partial.any():
            fig.add_trace(
                go.Scatter(
                    x=m["month"],
                    y=y.where(near),
                    mode="lines",
                    line={"color": P.BLUE, "width": 2, "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup="partial",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=m.loc[partial, "month"],
                    y=y[partial],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 10,
                        "color": P.BLUE,
                        "line": {"width": 2},
                    },
                    name="Partial month (not comparable)",
                    legendgroup="partial",
                    showlegend=row == 1,
                    hovertemplate=f"%{{x|%b %Y}} (partial month)<br>{fmt}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        full = m.loc[~partial]
        if not full.empty:  # selective direct label: the peak complete month
            peak = full.loc[full[measure].idxmax()]
            text = VC.money(peak[measure]) if measure == "revenue" else f"{peak[measure]:,.0f}"
            fig.add_annotation(
                x=peak["month"],
                y=peak[measure],
                text=f"Peak {peak['month']:%b %Y}: {text}",
                showarrow=True,
                arrowcolor=P.GREY,
                ax=-60,
                ay=-18,
                font={"color": P.INK, "size": 12},
                row=row,
                col=1,
            )
    fig.update_yaxes(title_text="Revenue (£)", tickprefix="£", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Orders", row=2, col=1, rangemode="tozero")
    fig.update_xaxes(title_text="Month", row=2, col=1)
    VC.style(fig, "Revenue and orders per month", height=520)
    fig.update_layout(legend={"orientation": "h", "y": -0.12, "x": 0, "yanchor": "top"})
    for ann in fig.layout.annotations:
        if ann.text in ("Revenue (£)", "Orders (distinct invoices)"):
            ann.font = {"color": P.MUTED_INK, "size": 13}
    return fig


def country_revenue(countries: pd.DataFrame | None, top_n: int | None = None) -> go.Figure:
    """Revenue per country on a log axis; the UK is highlighted AND labelled 'home market'."""
    if _empty(countries):
        return VC.empty_figure("No country data.", "Revenue by country")
    c = countries.sort_values("revenue", ascending=False)
    if top_n:
        c = c.head(int(top_n))
    c = c.iloc[::-1]  # largest on top of a horizontal bar chart
    is_uk = c["is_uk"].astype(bool).to_numpy()
    ylabels = [
        f"<b>{n}</b> (home market)" if uk else str(n)
        for n, uk in zip(c["country"], is_uk, strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=c["revenue"],
            y=ylabels,
            orientation="h",
            marker={"color": [HIGHLIGHT_COLOUR if uk else CONTEXT_COLOUR for uk in is_uk]},
            text=[f"{s:.1%}" if s >= 0.01 else "" for s in c["revenue_share"]],
            textposition="outside",
            textfont={"color": P.INK, "size": 11},
            cliponaxis=False,
            customdata=np.column_stack([c["revenue_share"], c["orders"], c["customers"]]),
            hovertemplate=(
                "%{y}<br>Revenue £%{x:,.0f} (%{customdata[0]:.1%} of total)<br>"
                "%{customdata[1]:,.0f} orders · %{customdata[2]:,.0f} customers<extra></extra>"
            ),
            showlegend=False,
        )
    )
    VC.log_axis(fig, "x", tickprefix="£")
    title = "Revenue by country" + (f" (top {len(c)})" if top_n else "")
    return VC.style(fig, title, "Revenue (£, log scale)", None, height=max(320, 18 * len(c) + 120))


def pareto_curve(pareto: pd.DataFrame | None, eda: dict | None = None) -> go.Figure:
    """Lorenz / Pareto curve: top x% of customers -> y% of revenue, with the equality line."""
    if _empty(pareto):
        return VC.empty_figure("No Pareto data.", "Revenue concentration")
    eda = eda or {}
    p = pareto.sort_values("customer_share")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"color": P.GREY, "width": 1, "dash": "dash"},
            name="Perfect equality",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p["customer_share"],
            y=p["revenue_share"],
            mode="lines",
            line={"color": P.BLUE, "width": 2},
            fill="tonexty",
            fillcolor=_rgba(P.BLUE, 0.08),
            name="Customers ranked by revenue",
            hovertemplate="Top %{x:.1%} of customers<br>bring %{y:.1%} of revenue<extra></extra>",
        )
    )
    for share, key in ((0.01, "top_1pct_revenue_share"), (0.1, "top_10pct_revenue_share")):
        _mark_top(fig, share, eda.get(key))
    _mark_top(fig, 0.2, eda.get("top_20pct_revenue_share"))
    gini = _num(eda.get("gini"))
    if np.isfinite(gini):
        fig.add_annotation(
            x=0.97,
            y=0.08,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            text=f"Gini = {gini:.2f}<br>(0 = everyone spends the same, 1 = one customer)",
            font={"color": P.INK, "size": 12},
            align="right",
        )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    VC.style(
        fig,
        "Revenue concentration (Pareto / Lorenz curve)",
        "Share of customers (largest spenders first)",
        "Share of revenue",
        height=420,
    )
    fig.update_layout(legend={"x": 0.98, "y": 0.35, "xanchor": "right"})
    return fig


def _mark_top(fig: go.Figure, share: float, value) -> None:
    v = _num(value)
    if not np.isfinite(v):
        return
    fig.add_trace(
        go.Scatter(
            x=[share],
            y=[v],
            mode="markers+text",
            marker={"color": P.BLUE, "size": 9, "line": P.marker_line(P.BLUE)},
            text=[f"top {share:.0%} → {v:.0%}"],
            textposition="middle right",
            textfont={"color": P.INK, "size": 12},
            showlegend=False,
            hovertemplate=f"Top {share:.0%} of customers bring %{{y:.1%}} of revenue"
            "<extra></extra>",
        )
    )


def cohort_heatmap(cohorts: pd.DataFrame | None, include_first: bool = False) -> go.Figure:
    """Share of each first-purchase cohort active N months later (sequential palette)."""
    if _empty(cohorts):
        return VC.empty_figure("No cohort data.", "Cohort retention")
    c = cohorts if include_first else cohorts[cohorts["period"] > 0]
    if c.empty:
        return VC.empty_figure("No cohort data after month 0.", "Cohort retention")
    ret = c.pivot_table(index="cohort", columns="period", values="retention", aggfunc="first")
    act = c.pivot_table(index="cohort", columns="period", values="active", aggfunc="first")
    size = c.groupby("cohort")["cohort_size"].first().reindex(ret.index)
    act = act.reindex(index=ret.index, columns=ret.columns)
    ylab = [f"{pd.Timestamp(d):%b %Y} (n={n:,})" for d, n in zip(ret.index, size, strict=True)]
    custom = np.dstack(
        [act.to_numpy(dtype=float), np.repeat(size.to_numpy()[:, None], ret.shape[1], 1)]
    )
    fig = go.Figure(
        go.Heatmap(
            z=ret.to_numpy(),
            x=[int(p) for p in ret.columns],
            y=ylab,
            colorscale=P.SEQUENTIAL,
            zmin=0,
            zmax=float(np.nanmax(ret.to_numpy())) if np.isfinite(np.nanmax(ret.to_numpy())) else 1,
            customdata=custom,
            xgap=1,
            ygap=1,
            colorbar={"title": {"text": "Active"}, "tickformat": ".0%"},
            hovertemplate=(
                "Cohort %{y}<br>Month %{x} after first purchase<br>"
                "%{z:.1%} active (%{customdata[0]:,.0f} of %{customdata[1]:,.0f})<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    fig.update_xaxes(dtick=1 if ret.shape[1] <= 12 else 3, showgrid=False)
    return VC.style(
        fig,
        "Cohort retention: share of each first-purchase month still buying",
        "Months since first purchase",
        "Cohort (first purchase month, size)",
        height=max(380, 20 * len(ylab) + 140),
    )


# ================================================================== OLAP cube
def grouping_id(dims: Iterable[str]) -> int:
    """grouping_id of the grouping set that KEEPS `dims` (bit set = dimension rolled up)."""
    dims = set(dims)
    unknown = dims - set(C.CUBE_DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown cube dimensions: {sorted(unknown)}")
    n = len(C.CUBE_DIMENSIONS)
    return sum(1 << (n - 1 - i) for i, d in enumerate(C.CUBE_DIMENSIONS) if d not in dims)


def cube_level(cube: pd.DataFrame | None, dims: Iterable[str]) -> pd.DataFrame:
    """Rows of the grouping set that keeps exactly `dims` (e.g. ("country", "year"))."""
    if _empty(cube):
        return pd.DataFrame(columns=C.CUBE_COLUMNS)
    return cube[cube["grouping_id"] == grouping_id(dims)].copy()


def cube_countries(cube: pd.DataFrame | None) -> list[str]:
    """Countries by total revenue (largest first), from the country grouping set."""
    lvl = cube_level(cube, ["country"])
    return lvl.sort_values("revenue", ascending=False)["country"].astype(str).tolist()


def cube_years(cube: pd.DataFrame | None) -> list[int]:
    lvl = cube_level(cube, ["year"])
    return sorted(int(y) for y in lvl["year"].dropna())


def grouping_sets_summary(cube: pd.DataFrame | None) -> pd.DataFrame:
    """One row per grouping set: rows, summed revenue, summed customers vs the true total.

    Revenue sums to the same total in every grouping set; summed customers over-count
    (a customer active in 3 months is counted 3 times) - the reason customer totals come
    from the rolled-up row instead.
    """
    cols = ["grouping_id", "level", "rows", "revenue_sum", "customers_sum", "customers_overcount"]
    if _empty(cube):
        return pd.DataFrame(columns=cols)
    total = cube.loc[cube["grouping_id"] == cube["grouping_id"].max(), "customers"]
    true_customers = float(total.iloc[0]) if len(total) else float("nan")
    g = (
        cube.groupby(["grouping_id", "level"], sort=True)
        .agg(
            rows=("revenue", "size"),
            revenue_sum=("revenue", "sum"),
            customers_sum=("customers", "sum"),
        )
        .reset_index()
    )
    g["customers_overcount"] = g["customers_sum"] / true_customers
    return g[cols]


def _year_filter(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    return df if year is None else df[df["year"] == int(year)]


def cube_totals(
    cube: pd.DataFrame | None, countries: Sequence[str] | None = None, year: int | None = None
) -> dict:
    """Headline numbers of a slice, each read from the right grouping set.

    countries None / empty = all countries; year None = all years. Revenue / orders of
    several countries are summed (additive); customers are exact only for one country or all
    countries, otherwise None plus the sum as an upper bound.
    """
    out = {
        "revenue": float("nan"),
        "orders": float("nan"),
        "customers": None,
        "customers_upper": None,
        "source": "",
    }
    if _empty(cube):
        return out
    countries = list(countries or [])
    if not countries:
        dims = [] if year is None else ["year"]
        rows = _year_filter(cube_level(cube, dims), year)
        out["source"] = "total" if year is None else "year"
        if len(rows):
            r = rows.iloc[0]
            out.update(
                revenue=float(r["revenue"]),
                orders=float(r["orders"]),
                customers=int(r["customers"]),
            )
        return out
    dims = ["country"] if year is None else ["country", "year"]
    rows = _year_filter(cube_level(cube, dims), year)
    rows = rows[rows["country"].isin(countries)]
    out["source"] = " × ".join(dims)
    out["revenue"] = float(rows["revenue"].sum())
    out["orders"] = float(rows["orders"].sum())
    if len(countries) == 1:
        out["customers"] = int(rows["customers"].sum()) if len(rows) else 0
    else:
        out["customers_upper"] = int(rows["customers"].sum())
    return out


def cube_slice_table(
    cube: pd.DataFrame | None, countries: Sequence[str] | None = None, year: int | None = None
) -> pd.DataFrame:
    """Per-country totals of the slice (country or country × year grouping set)."""
    cols = ["country", "revenue", "orders", "customers"]
    if _empty(cube):
        return pd.DataFrame(columns=cols)
    dims = ["country"] if year is None else ["country", "year"]
    rows = _year_filter(cube_level(cube, dims), year)
    if countries:
        rows = rows[rows["country"].isin(list(countries))]
    return rows.sort_values("revenue", ascending=False)[cols].reset_index(drop=True)


def cube_dice(
    cube: pd.DataFrame | None, countries: Sequence[str] | None = None, year: int | None = None
) -> pd.DataFrame:
    """Country × month cells of the dice: country × year × month rows of one year, or the
    country × month grouping set when year is None (all years rolled up)."""
    cols = ["country", "month", "revenue", "orders", "customers"]
    if _empty(cube):
        return pd.DataFrame(columns=cols)
    if year is None:
        rows = cube_level(cube, ["country", "month"])
    else:
        rows = _year_filter(cube_level(cube, ["country", "year", "month"]), year)
    if countries:
        rows = rows[rows["country"].isin(list(countries))]
    return rows[cols].sort_values(["country", "month"]).reset_index(drop=True)


def cube_dice_heatmap(
    dice: pd.DataFrame | None, measure: str = "revenue", title: str = ""
) -> go.Figure:
    """Country × month heatmap of one measure; colour on a log scale with real-value ticks."""
    label = MEASURE_LABELS.get(measure, measure)
    if _empty(dice):
        return VC.empty_figure("No rows for this slice.", title or label)
    order = dice.groupby("country")[measure].sum().sort_values(ascending=False).index
    z = dice.pivot_table(index="country", columns="month", values=measure, aggfunc="first")
    z = z.reindex(index=order, columns=range(1, 13))
    vals = z.to_numpy(dtype=float)
    positive = vals[np.isfinite(vals) & (vals > 0)]
    if positive.size == 0:
        return VC.empty_figure("No positive values for this slice.", title or label)
    logz = np.where(vals > 0, np.log10(np.where(vals > 0, vals, 1)), np.nan)
    lo, hi = np.floor(np.log10(positive.min())), np.ceil(np.log10(positive.max()))
    if hi <= lo:
        hi = lo + 1
    ticks = np.arange(lo, hi + 1)
    prefix = "£" if measure == "revenue" else ""
    fig = go.Figure(
        go.Heatmap(
            z=logz,
            x=list(MONTH_ABBR),
            y=[str(c) for c in z.index],
            customdata=vals,
            colorscale=P.SEQUENTIAL,
            zmin=lo,
            zmax=hi,
            xgap=1,
            ygap=1,
            colorbar={
                "title": {"text": "£" if measure == "revenue" else label},
                "tickvals": ticks,
                "ticktext": [f"{prefix}{10**t:,.0f}" for t in ticks],
            },
            hovertemplate=(
                f"%{{y}} · %{{x}}<br>{label}: {prefix}%{{customdata:,.0f}}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return VC.style(
        fig,
        title or f"{label} by country and month",
        "Month",
        "Country",
        height=max(340, 26 * len(z.index) + 140),
    )


def _values_from_children(ids: list[str], parents: list[str], values: list[float]) -> list[float]:
    """Parent value = sum of its children (leaves keep theirs); children must follow parents.

    branchvalues="total" makes plotly.js drop the whole chart when a parent is smaller than
    the sum of its children (tolerance 1e-6). Each grouping-set row is rounded to pennies on
    its own, so a parent's own row can fall a few pence short of its rounded children.
    """
    pos = {node: n for n, node in enumerate(ids)}
    total, child_sum, has_child = list(values), [0.0] * len(ids), [False] * len(ids)
    for n in range(len(ids) - 1, -1, -1):
        if has_child[n]:
            total[n] = child_sum[n]
        if parents[n]:
            p = pos[parents[n]]
            child_sum[p] += total[n]
            has_child[p] = True
    return total


def cube_sunburst(
    cube: pd.DataFrame | None,
    measure: str = "revenue",
    countries: Sequence[str] | None = None,
    year: int | None = None,
) -> go.Figure:
    """Drill-down country -> year -> month; every ring is read from its own grouping set.

    Sizes use an additive measure (revenue or orders), summed up from the finest ring so
    parents always equal their children; hover shows the exact distinct customers of each
    node from its own rolled-up row (customers are never summed). The UK branch is blue (labelled),
    other countries grey - the same highlight as the country bar chart.
    """
    if measure not in ADDITIVE_MEASURES:
        raise ValueError(f"sunburst sizes need an additive measure, not {measure!r}")
    label = MEASURE_LABELS[measure]
    lvl_c = cube_level(cube, ["country"])
    if lvl_c.empty:
        return VC.empty_figure("No cube rows.", "OLAP drill-down")
    lvl_cy = _year_filter(cube_level(cube, ["country", "year"]), year)
    lvl_cym = _year_filter(cube_level(cube, ["country", "year", "month"]), year)
    if year is not None:  # country ring = that year's country × year rows
        lvl_c = lvl_cy.copy()
    if countries:
        keep = list(countries)
        lvl_c, lvl_cy, lvl_cym = (d[d["country"].isin(keep)] for d in (lvl_c, lvl_cy, lvl_cym))
    lvl_c = lvl_c[lvl_c[measure] > 0]
    if lvl_c.empty:
        return VC.empty_figure("No rows for this slice.", "OLAP drill-down")

    def colour(country: str, depth: int) -> str:
        base = HIGHLIGHT_COLOUR if country == UK else P.GREY
        return _rgba(base, (1.0, 0.7, 0.45)[depth])

    root = "root"
    ids, labels, parents, values, custs, colours = [root], [ALL_LABEL], [""], [], [], [P.SURFACE]
    values.append(float(lvl_c[measure].sum()))
    totals = cube_totals(cube, countries, year)
    custs.append(totals["customers"] if totals["customers"] is not None else None)
    for _, r in lvl_c.iterrows():
        cid = f"{root}/{r['country']}"
        ids.append(cid)
        labels.append(str(r["country"]))
        parents.append(root)
        values.append(float(r[measure]))
        custs.append(int(r["customers"]))
        colours.append(colour(r["country"], 0))
    for _, r in lvl_cy.iterrows():
        if f"{root}/{r['country']}" not in ids:
            continue
        ids.append(f"{root}/{r['country']}/{int(r['year'])}")
        labels.append(str(int(r["year"])))
        parents.append(f"{root}/{r['country']}")
        values.append(float(r[measure]))
        custs.append(int(r["customers"]))
        colours.append(colour(r["country"], 1))
    year_ids = set(ids)
    for _, r in lvl_cym.iterrows():
        parent = f"{root}/{r['country']}/{int(r['year'])}"
        if parent not in year_ids:
            continue
        ids.append(f"{parent}/{int(r['month'])}")
        labels.append(MONTH_ABBR[int(r["month"]) - 1])
        parents.append(parent)
        values.append(float(r[measure]))
        custs.append(int(r["customers"]))
        colours.append(colour(r["country"], 2))
    values = _values_from_children(ids, parents, values)

    fmt = "£%{value:,.0f}" if measure == "revenue" else "%{value:,.0f} orders"
    text_colours = [P.SURFACE if c == _rgba(HIGHLIGHT_COLOUR, 1.0) else P.INK for c in colours]
    cust_text = [f"{c:,.0f}" if c is not None else "not additive across countries" for c in custs]
    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            maxdepth=3,
            customdata=cust_text,
            marker={"colors": colours, "line": {"color": P.SURFACE, "width": 1}},
            insidetextorientation="radial",
            insidetextfont={"color": text_colours},
            hovertemplate=(
                f"%{{currentPath}}<b>%{{label}}</b><br>{label}: {fmt}<br>"
                "%{percentParent:.1%} of parent<br>"
                "Distinct customers: %{customdata}<extra></extra>"
            ),
        )
    )
    yr = "all years" if year is None else str(year)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 60, "b": 10})
    return VC.style(fig, f"Drill-down: {label} by country → year → month ({yr})", height=560)
