"""Data: the dataset card, its 8 columns, sample rows, the cleaning funnel and the shape of the
business (monthly sales, countries, revenue concentration)."""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.viz import data_cube as D

layout.setup_page("Data")
layout.page_header(
    "Data",
    "What data do we have, and can we trust it?",
    "From the raw Excel file to clean sales lines of registered customers.",
)

fmt = data.json_file(C.FORMAT_COMPARISON_JSON)
cleaning = data.json_file(C.CLEANING_STEPS_JSON)
eda = data.metrics("eda")

# ------------------------------------------------------------------ dataset card
st.subheader("The dataset")
with st.container(border=True):
    st.markdown(
        "**[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)** "
        "(Chen, 2012) · licence **CC BY 4.0**  \n"
        "Every invoice line of a UK-based online shop that sells gifts, mostly to wholesalers."
    )
    sheets = fmt.get("sheets") or {}
    layout.kpi_row(
        [
            {
                "label": "Raw rows",
                "value": f"{fmt['rows']:,}" if fmt.get("rows") else "n/a",
                "help": " + ".join(f"{k}: {v:,}" for k, v in sheets.items()) or None,
            },
            {
                "label": "First invoice",
                "value": str(fmt.get("date_min", "n/a"))[:10],
            },
            {
                "label": "Last invoice",
                "value": str(fmt.get("date_max", "n/a"))[:10],
            },
            {"label": "Excel sheets", "value": str(len(sheets)) if sheets else "n/a"},
        ]
    )
st.markdown(f"**The {len(C.RAW_COLUMNS)} columns**")
st.dataframe(D.raw_columns_table(), hide_index=True, width="stretch", key="data_columns")

sample = data.table(C.EDA_SAMPLE_ROWS_PARQUET)
st.markdown(
    f"**{0 if sample is None else len(sample)} sample rows** "
    "(after cleaning; line_value = quantity × price)"
)
if sample is None or sample.empty:
    st.info("Sample rows are not available.")
else:
    st.dataframe(
        sample,
        hide_index=True,
        width="stretch",
        column_config={
            "price": st.column_config.NumberColumn(format="£%.2f"),
            "line_value": st.column_config.NumberColumn(format="£%.2f"),
            "customer_id": st.column_config.NumberColumn(format="%d"),
        },
        key="data_sample",
    )

# ------------------------------------------------------------------ cleaning
st.subheader("Cleaning")
tour.callout(
    1,
    "Follow the rows from left to right: duplicates, returns (kept in their own table) and "
    "invalid quantities leave first; the biggest loss is lines without a Customer ID.",
    "Cleaning funnel",
)
layout.chart(D.cleaning_sankey(cleaning), key="data_sankey")
st.caption(
    "Lines without a Customer ID (guest checkouts) cannot be linked to a customer, so they "
    "are dropped: the results describe registered customers only."
)

# ------------------------------------------------------------------ the business
st.subheader("Sales over time")
tour.callout(
    2,
    "Revenue peaks every November (Christmas gift stock). Hollow points are partial months.",
    "Seasonality",
)
layout.chart(D.monthly_trend(data.table(C.EDA_MONTHLY_PARQUET)), key="data_monthly")

st.subheader("Customers and countries")
tour.callout(
    3,
    "A small share of customers brings most of the revenue, and most revenue is from the UK. "
    "That is why we segment customers instead of using averages.",
    "Concentration",
)
col_pareto, col_country = st.columns(2)
with col_pareto:
    layout.chart(D.pareto_curve(data.table(C.EDA_PARETO_PARQUET), eda), key="data_pareto")
with col_country:
    layout.chart(D.country_revenue(data.table(C.EDA_COUNTRIES_PARQUET)), key="data_countries")
layout.conclusion(eda.get("conclusion"))

st.divider()
st.page_link("Home.py", label="← Back to the live demo", icon=":material/home:")
