"""Data & Cube: data quality (cleaning funnel, missing IDs, file format), EDA and the OLAP cube."""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.viz import data_cube as D
from retailpulse.viz.common import money, pct

layout.setup_page("Data & Cube")
layout.page_header(
    "Data & Cube",
    "A · Who are our customers? First: what data do we have, and can we trust it?",
    "From 1 million raw invoice lines to clean sales of registered customers, then the "
    "shape of the business over time, across customers and countries, and an OLAP cube "
    "you can slice and drill.",
)

cleaning = data.json_file(C.CLEANING_STEPS_JSON)
missing = data.json_file(C.MISSING_ID_JSON)
fmt = data.json_file(C.FORMAT_COMPARISON_JSON)
wh = data.json_file(C.WAREHOUSE_SUMMARY_JSON)
eda = data.metrics("eda")
feats = data.metrics("features")
cube = data.table(C.CUBE_PARQUET)

tab_quality, tab_time, tab_customers, tab_cube = st.tabs(
    ["Data quality", "Sales over time", "Customers & countries", "OLAP cube"]
)

# ------------------------------------------------------------------ data quality
with tab_quality:
    steps = cleaning.get("steps") or []
    final_rows = steps[-1].get("rows_out") if steps else None
    layout.kpi_row(
        [
            {"label": "Raw rows (2 Excel sheets)", "value": f"{cleaning.get('raw_rows', 0):,}"},
            {
                "label": "Clean sales lines",
                "value": f"{final_rows:,}" if final_rows is not None else "n/a",
                "help": "Valid sales lines of registered customers after every cleaning step.",
            },
            {
                "label": "Customers",
                "value": f"{(wh.get('row_counts') or {}).get('dim_customer', 0):,}",
            },
            {
                "label": "Lines without a Customer ID",
                "value": pct(missing.get("share_missing", float("nan")), 1),
                "help": "Share of valid lines (after duplicates, returns and invalid rows).",
            },
        ]
    )
    tour.callout(
        1,
        "Follow the rows from left to right: duplicates, returns (kept in their own table) "
        "and invalid quantities leave first; the biggest loss is lines without a Customer ID.",
        "Cleaning funnel",
    )
    layout.chart(D.cleaning_sankey(cleaning), key="p1_sankey")

    col_missing, col_format = st.columns(2)
    with col_missing:
        st.subheader("Missing Customer ID: MCAR or not?")
        layout.chart(D.missing_id_effects(missing), key="p1_missing")
        if missing.get("verdict"):
            layout.conclusion(missing["verdict"], "Verdict")
        st.caption(
            "Rank-biserial r compares the two groups' distributions (Mann-Whitney U); "
            "Cramér's V measures the association in the UK × missing table (chi-square). "
            "Any clear difference means the IDs are not missing completely at random."
        )
    with col_format:
        st.subheader("Excel vs Parquet")
        layout.chart(D.format_comparison(fmt), key="p1_format")
        layout.kpi_row(
            [
                {
                    "label": "Parquet is smaller by",
                    "value": f"{fmt['size_ratio']:,.1f}×" if fmt.get("size_ratio") else "n/a",
                },
                {
                    "label": "… and loads faster by",
                    "value": f"{fmt['speedup']:,.0f}×" if fmt.get("speedup") else "n/a",
                    "help": "Best of 3 loads of the same 1M-row table.",
                },
            ]
        )
        st.caption(
            "Columnar, compressed and typed: the raw Excel file is converted once and every "
            "later step reads Parquet."
        )

# ------------------------------------------------------------------ sales over time
with tab_time:
    tour.callout(
        2,
        "Revenue and orders move together and peak every November (Christmas gift stock). "
        "Hollow points are partial months: the data starts or ends mid-month.",
        "Seasonality",
    )
    layout.chart(D.monthly_trend(data.table(C.EDA_MONTHLY_PARQUET)), key="p1_monthly")
    st.subheader("Do new customers come back?")
    layout.chart(D.cohort_heatmap(data.table(C.EDA_COHORTS_PARQUET)), key="p1_cohorts")
    st.caption(
        "Each row is the group of customers whose first purchase was in that month; month 0 "
        "(100% by definition) is left out so the colour scale shows the later months. "
        "December 2009 also holds customers who bought before the data starts."
    )

# ------------------------------------------------------------------ customers & countries
with tab_customers:
    tour.callout(
        3,
        "Spend is concentrated: a small share of customers brings most of the revenue. "
        "That is why segments and 'who comes back' matter more than averages.",
        "Pareto",
    )
    col_pareto, col_country = st.columns(2)
    with col_pareto:
        layout.chart(D.pareto_curve(data.table(C.EDA_PARETO_PARQUET), eda), key="p1_pareto")
    with col_country:
        layout.chart(D.country_revenue(data.table(C.EDA_COUNTRIES_PARQUET)), key="p1_countries")
    layout.kpi_row(
        [
            {"label": "Gini of customer revenue", "value": f"{eda.get('gini', float('nan')):.2f}"},
            {
                "label": "Top 1% of customers",
                "value": pct(eda.get("top_1pct_revenue_share", float("nan"))),
            },
            {"label": "Top 10%", "value": pct(eda.get("top_10pct_revenue_share", float("nan")))},
            {"label": "Top 20%", "value": pct(eda.get("top_20pct_revenue_share", float("nan")))},
            {
                "label": "UK share of revenue",
                "value": pct(eda.get("uk_revenue_share", float("nan"))),
            },
        ]
    )

# ------------------------------------------------------------------ OLAP cube
with tab_cube:
    if cube is None or cube.empty:
        st.info("The OLAP cube has not been built yet (pipeline stage `cube`).")
    else:
        countries = D.cube_countries(cube)
        years = D.cube_years(cube)
        tour.callout(
            4,
            "Slice by year, dice by a few countries, then click a country in the sunburst to "
            "drill down to its years and months. Every number comes from its own grouping set.",
            "OLAP cube",
        )
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            chosen = st.multiselect(
                "Countries (dice; empty = all countries)",
                countries,
                default=countries[:5],
                key="cube_countries",
            )
        with c2:
            year = st.selectbox(
                "Year (slice)",
                [None, *years],
                format_func=lambda y: "All years" if y is None else str(y),
                key="cube_year",
            )
        with c3:
            measure = st.radio(
                "Size by",
                list(D.ADDITIVE_MEASURES),
                format_func=lambda m: D.MEASURE_LABELS[m],
                key="cube_measure",
                horizontal=True,
            )

        totals = D.cube_totals(cube, chosen, year)
        if totals["customers"] is not None:
            cust_value, cust_help = (
                f"{totals['customers']:,}",
                (f"Exact: read from the '{totals['source']}' grouping set."),
            )
        else:
            cust_value = f"≤ {totals['customers_upper']:,}"
            cust_help = (
                "Distinct customers are not additive: a customer who bought in two of the "
                "chosen countries is counted in both rows. The sum is only an upper bound."
            )
        if not chosen:
            scope = "all countries"
        else:
            scope = chosen[0] if len(chosen) == 1 else f"{len(chosen)} countries"
        layout.kpi_row(
            [
                {"label": f"Revenue · {scope}", "value": money(totals["revenue"])},
                {"label": "Orders", "value": f"{totals['orders']:,.0f}"},
                {"label": "Distinct customers", "value": cust_value, "help": cust_help},
            ]
        )

        layout.chart(D.cube_sunburst(cube, measure, chosen, year), key="p1_sunburst")
        st.caption(
            "Click a country to drill down to years and months; click the centre to go back."
        )

        dice = D.cube_dice(cube, chosen, year)
        yr = "all years" if year is None else str(year)
        layout.chart(
            D.cube_dice_heatmap(
                dice, measure, f"{D.MEASURE_LABELS[measure]} by country and month ({yr})"
            ),
            key="p1_dice",
        )
        st.markdown("**Slice totals per country**")
        st.dataframe(
            D.cube_slice_table(cube, chosen, year),
            hide_index=True,
            width="stretch",
            column_config={
                "country": "Country",
                "revenue": st.column_config.NumberColumn("Revenue (£)", format="£%,.0f"),
                "orders": st.column_config.NumberColumn("Orders", format="%,d"),
                "customers": st.column_config.NumberColumn("Distinct customers", format="%,d"),
            },
            key="cube_table",
        )

        with st.expander("How CUBE works (grouping sets, roll-up, NULL = all)"):
            st.markdown(
                "`GROUP BY CUBE(country, year, month)` computes **all 2³ = 8 grouping sets** in "
                "one query: every combination of the three dimensions, from the finest "
                "(country × year × month) up to the grand total. A dimension that is rolled "
                "up is stored as **NULL, which means 'all'**; `grouping_id` encodes which "
                "dimensions are rolled up (country = 4, year = 2, month = 1).\n\n"
                "- **Slice** = fix one dimension (a year); **dice** = pick a sub-cube (some "
                "countries × one year); **drill-down / roll-up** = move between grouping sets.\n"
                "- Revenue and orders are additive, so every grouping set sums to the same "
                "total. **Distinct customers are not**: one customer buying in 5 months is "
                "1 customer for the year but 5 in the month rows. So totals of customers are "
                "always read from the rolled-up row, never summed."
            )
            summary = D.grouping_sets_summary(cube)
            st.dataframe(
                summary,
                hide_index=True,
                width="stretch",
                column_config={
                    "grouping_id": "grouping_id",
                    "level": "Grouping set (kept dimensions)",
                    "rows": "Rows",
                    "revenue_sum": st.column_config.NumberColumn("Σ revenue (£)", format="£%,.0f"),
                    "customers_sum": st.column_config.NumberColumn("Σ customers", format="%,d"),
                    "customers_overcount": st.column_config.NumberColumn(
                        "Σ customers ÷ true total", format="%.2f×"
                    ),
                },
                key="cube_sets",
            )

st.divider()
layout.conclusion(
    [eda.get("conclusion"), feats.get("conclusion")],
    "What this tells us",
)
