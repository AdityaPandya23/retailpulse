"""Page 1 figures (viz/data_cube.py) on the MOCK artifacts and tiny hand-made frames."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from retailpulse import contracts as C
from retailpulse.viz import data_cube as D
from retailpulse.viz import palette as P


def _mock_parquet(path):
    file = C.mock_path(path)
    if not file.exists():
        pytest.skip("artifacts/mock/ not generated")
    return pd.read_parquet(file)


def _mock_json(path):
    file = C.mock_path(path)
    if not file.exists():
        pytest.skip("artifacts/mock/ not generated")
    return json.loads(file.read_text(encoding="utf-8"))


def _is_empty_figure(fig: go.Figure) -> bool:
    return len(fig.data) == 0 and len(fig.layout.annotations) == 1


@pytest.fixture
def cube():
    return _mock_parquet(C.CUBE_PARQUET)


# ------------------------------------------------------------------ quality
def test_cleaning_sankey_conserves_rows():
    steps = _mock_json(C.CLEANING_STEPS_JSON)
    fig = D.cleaning_sankey(steps)
    assert len(fig.data) == 1 and fig.data[0].type == "sankey"
    link = fig.data[0].link
    out_of_raw = sum(v for s, v in zip(link.source, link.value, strict=True) if s == 0)
    assert out_of_raw == steps["raw_rows"]  # raw rows = kept + removed at step 1
    labels = list(fig.data[0].node.label)
    final = [s for s in steps["steps"] if s["removed"] > 0][-1]["rows_out"]
    assert any("Clean sales lines" in lab and f"{final:,}" in lab for lab in labels)


def test_cleaning_sankey_skips_zero_steps_and_handles_missing():
    steps = {
        "raw_rows": 100,
        "steps": [
            {"step": "a", "rows_in": 100, "rows_out": 90, "removed": 10, "removed_to": "X"},
            {"step": "b", "rows_in": 90, "rows_out": 90, "removed": 0, "removed_to": None},
        ],
    }
    fig = D.cleaning_sankey(steps)
    assert len(fig.data[0].link.value) == 2  # kept + removed of step a only
    assert _is_empty_figure(D.cleaning_sankey({}))
    assert _is_empty_figure(D.cleaning_sankey(None))


def test_missing_id_effects_has_three_bars():
    fig = D.missing_id_effects(_mock_json(C.MISSING_ID_JSON))
    assert fig.data[0].type == "bar" and len(fig.data[0].x) == 3
    assert all(-1 <= v <= 1 for v in fig.data[0].x)
    assert _is_empty_figure(D.missing_id_effects({"tests": {}}))


def test_format_comparison_two_panels_no_dual_axis():
    fig = D.format_comparison(_mock_json(C.FORMAT_COMPARISON_JSON))
    assert [t.type for t in fig.data] == ["bar", "bar"]
    assert fig.layout.yaxis2.type == "log"  # load seconds span 3 orders of magnitude
    assert fig.layout.yaxis2.overlaying is None  # separate panels, not a dual axis
    assert _is_empty_figure(D.format_comparison({"xlsx_mb": 1}))


# ------------------------------------------------------------------ EDA
def test_monthly_trend_marks_partial_months():
    monthly = _mock_parquet(C.EDA_MONTHLY_PARQUET)
    fig = D.monthly_trend(monthly)
    names = {t.name for t in fig.data if t.showlegend}
    assert "Partial month (not comparable)" in names
    hollow = [t for t in fig.data if t.marker and t.marker.symbol == "circle-open"]
    assert len(hollow) == 2  # revenue and orders panels
    assert len(hollow[0].x) == int(monthly["is_partial"].sum())
    assert fig.layout.yaxis2.overlaying is None


def test_monthly_trend_without_partial_and_empty():
    m = pd.DataFrame(
        {
            "month": pd.date_range("2010-01-01", periods=3, freq="MS"),
            "revenue": [1.0, 2.0, 3.0],
            "orders": [1, 2, 3],
            "is_partial": [False, False, False],
        }
    )
    fig = D.monthly_trend(m)
    assert len(fig.data) == 2  # one line per panel, no partial traces
    assert _is_empty_figure(D.monthly_trend(None))
    assert _is_empty_figure(D.monthly_trend(m.iloc[0:0]))


def test_country_revenue_highlights_and_labels_uk():
    countries = _mock_parquet(C.EDA_COUNTRIES_PARQUET)
    fig = D.country_revenue(countries)
    bar = fig.data[0]
    assert fig.layout.xaxis.type == "log"
    uk = [i for i, y in enumerate(bar.y) if "home market" in y]
    assert len(uk) == 1  # label, not colour alone
    assert bar.marker.color[uk[0]] == D.HIGHLIGHT_COLOUR
    assert sum(c == D.HIGHLIGHT_COLOUR for c in bar.marker.color) == 1
    assert len(D.country_revenue(countries, top_n=2).data[0].y) == 2


def test_pareto_curve_has_equality_line_and_gini():
    metrics = _mock_json(C.METRICS_JSON)
    fig = D.pareto_curve(_mock_parquet(C.EDA_PARETO_PARQUET), metrics["eda"])
    names = [t.name for t in fig.data]
    assert "Perfect equality" in names and "Customers ranked by revenue" in names
    assert any("Gini" in a.text for a in fig.layout.annotations)
    assert len(D.pareto_curve(_mock_parquet(C.EDA_PARETO_PARQUET), {}).data) == 2
    assert _is_empty_figure(D.pareto_curve(pd.DataFrame(columns=C.PARETO_COLUMNS)))


def test_cohort_heatmap_sequential_without_month_zero():
    cohorts = _mock_parquet(C.EDA_COHORTS_PARQUET)
    fig = D.cohort_heatmap(cohorts)
    hm = fig.data[0]
    assert hm.type == "heatmap" and 0 not in list(hm.x)
    assert [list(c) for c in hm.colorscale] == [list(c) for c in P.SEQUENTIAL]
    assert 0 in list(D.cohort_heatmap(cohorts, include_first=True).data[0].x)
    assert _is_empty_figure(D.cohort_heatmap(None))


# ------------------------------------------------------------------ cube
def test_grouping_id_matches_levels(cube):
    assert D.grouping_id(["country", "year", "month"]) == 0
    assert D.grouping_id([]) == 7
    for gid, level in cube.drop_duplicates("grouping_id")[["grouping_id", "level"]].itertuples(
        index=False
    ):
        kept = [] if level == "total" else [d.strip() for d in level.split("×")]
        assert D.grouping_id(kept) == gid
    with pytest.raises(ValueError):
        D.grouping_id(["colour"])


def test_cube_totals_come_from_rolled_up_rows(cube):
    total = cube[cube["grouping_id"] == 7].iloc[0]
    t = D.cube_totals(cube)
    assert t["customers"] == total["customers"] and t["revenue"] == pytest.approx(total["revenue"])
    countries = D.cube_countries(cube)
    one = D.cube_totals(cube, countries[:1])
    row = cube[(cube["grouping_id"] == 3) & (cube["country"] == countries[0])].iloc[0]
    assert one["customers"] == row["customers"]
    year = D.cube_years(cube)[-1]
    yrow = cube[(cube["grouping_id"] == 5) & (cube["year"] == year)].iloc[0]
    assert D.cube_totals(cube, None, year)["customers"] == yrow["customers"]
    many = D.cube_totals(cube, countries[:3])
    assert many["customers"] is None and many["customers_upper"] >= one["customers"]
    assert many["revenue"] == pytest.approx(
        cube[(cube["grouping_id"] == 3) & cube["country"].isin(countries[:3])]["revenue"].sum()
    )


def test_grouping_sets_summary_revenue_additive_customers_not(cube):
    s = D.grouping_sets_summary(cube)
    assert len(s) == 8
    assert s["revenue_sum"].round(2).nunique() == 1  # every set sums to the same revenue
    assert s.loc[s["grouping_id"] == 7, "customers_overcount"].iloc[0] == pytest.approx(1)
    assert (s["customers_overcount"] >= 1).all() and (s["customers_overcount"] > 1).any()


def test_cube_sunburst_ids_unique_and_parents_exist(cube):
    fig = D.cube_sunburst(cube)
    sb = fig.data[0]
    assert sb.type == "sunburst" and sb.branchvalues == "total"
    assert len(set(sb.ids)) == len(sb.ids)
    ids = set(sb.ids)
    assert all(p == "" or p in ids for p in sb.parents)
    # country ring sums to the cube total (revenue is additive)
    root_value = sb.values[0]
    assert root_value == pytest.approx(cube[cube["grouping_id"] == 7]["revenue"].iloc[0])


def test_cube_sunburst_slice_dice_and_measure(cube):
    countries = D.cube_countries(cube)[:2]
    year = D.cube_years(cube)[-1]
    sb = D.cube_sunburst(cube, "orders", countries, year).data[0]
    country_nodes = [lab for lab, p in zip(sb.labels, sb.parents, strict=True) if p == "root"]
    assert set(country_nodes) <= set(countries)
    years = {lab for lab, i in zip(sb.labels, sb.ids, strict=True) if i.count("/") == 2}
    assert years <= {str(year)}
    with pytest.raises(ValueError):
        D.cube_sunburst(cube, "customers")  # not additive -> not a size
    assert _is_empty_figure(D.cube_sunburst(cube, "revenue", ["Atlantis"]))
    assert _is_empty_figure(D.cube_sunburst(None))


def test_cube_dice_reads_the_right_grouping_set(cube):
    countries = D.cube_countries(cube)[:2]
    year = D.cube_years(cube)[-1]
    dice = D.cube_dice(cube, countries, year)
    assert set(dice["country"]) <= set(countries)
    expected = cube[
        (cube["grouping_id"] == 0) & cube["country"].isin(countries) & (cube["year"] == year)
    ]
    assert len(dice) == len(expected)
    all_years = D.cube_dice(cube, countries, None)
    assert len(all_years) == len(cube[(cube["grouping_id"] == 2) & cube["country"].isin(countries)])
    fig = D.cube_dice_heatmap(dice, "revenue")
    assert fig.data[0].type == "heatmap" and len(fig.data[0].x) == 12
    assert _is_empty_figure(D.cube_dice_heatmap(dice.iloc[0:0]))
    table = D.cube_slice_table(cube, countries, year)
    assert list(table["country"]) == sorted(
        table["country"], key=lambda c: -table.set_index("country").loc[c, "revenue"]
    )


def _rounded_cube() -> pd.DataFrame:
    """One country, one year, 12 months of £0.004 each: every row rounded to pennies on its
    own (like cube.py) -> months 12 x £0.01 = £0.12 but the year / country rows only £0.05."""
    rows = []
    for gid, country, year, month, revenue in [(0, "UK", 2010, m, 0.01) for m in range(1, 13)] + [
        (1, "UK", 2010, None, 0.05),
        (3, "UK", None, None, 0.05),
    ]:
        rows.append(
            {
                "country": country,
                "year": year,
                "month": month,
                "grouping_id": gid,
                "level": "x",
                "revenue": revenue,
                "orders": 1,
                "customers": 1,
            }
        )
    return pd.DataFrame(rows, columns=C.CUBE_COLUMNS)


def test_cube_sunburst_parents_never_smaller_than_children():
    """Regression (P4 review): branchvalues="total" breaks when a parent < sum(children)."""
    sb = D.cube_sunburst(_rounded_cube(), "revenue").data[0]
    value = dict(zip(sb.ids, sb.values, strict=True))
    kids: dict[str, float] = {}
    for node, parent in zip(sb.ids, sb.parents, strict=True):
        if parent:
            kids[parent] = kids.get(parent, 0.0) + value[node]
    for parent, total in kids.items():
        assert value[parent] >= total * (1 - 1e-9), parent
    assert value["root/UK"] == pytest.approx(0.12)
