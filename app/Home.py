"""RetailPulse v2 home = the OVERVIEW: headline numbers, the pipeline and the dataset (card,
columns, sample rows, cleaning funnel, sales over time, countries). Run: streamlit run app/Home.py

The live demo (score one customer with the saved models) is the next page, pages/2_Demo.py.
All logic is in retailpulse.demo / viz; this page only lays it out.
"""

from __future__ import annotations

import streamlit as st
from components import data, layout

from retailpulse import contracts as C
from retailpulse import demo
from retailpulse.viz import data_cube as D
from retailpulse.viz.common import money

DEMO_LABEL = "Next: the live demo →"

layout.setup_page("Overview")
layout.page_header(
    "RetailPulse",
    "Who are our customers, and will they come back?",
    "Overview: the data, how it was cleaned and the pipeline behind the models. "
    "The live demo on the next page scores one customer with the saved models.",
)

fmt = data.json_file(C.FORMAT_COMPARISON_JSON)
cleaning = data.json_file(C.CLEANING_STEPS_JSON)
eda = data.metrics("eda")

# ------------------------------------------------------------------ overview
st.subheader("Overview")
facts = demo.kpi_facts(data.json_file(C.WAREHOUSE_SUMMARY_JSON), cleaning, fmt)
layout.kpi_row(
    [
        {
            "label": "Clean sales lines",
            "value": f"{facts['clean_rows']:,}" if facts["clean_rows"] else "n/a",
            "help": f"From {facts['raw_rows']:,} raw rows." if facts["raw_rows"] else None,
        },
        {
            "label": "Customers",
            "value": f"{facts['customers']:,}" if facts["customers"] else "n/a",
            "help": "Registered customers (guest checkouts have no Customer ID).",
        },
        {"label": "Revenue", "value": money(facts["revenue"] or float("nan"), 2)},
        {
            "label": "Date range",
            "value": f"{facts['date_min']} → {facts['date_max']}" if facts["date_min"] else "n/a",
        },
    ]
)

# ------------------------------------------------------------------ pipeline
st.subheader("How it works: the pipeline")
st.graphviz_chart(demo.pipeline_dot())
st.caption(
    "An offline pipeline builds every model and table once; this app only reads the results."
)
st.page_link("pages/2_Demo.py", label=DEMO_LABEL, icon=":material/play_circle:")

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
            {"label": "First invoice", "value": str(fmt.get("date_min", "n/a"))[:10]},
            {"label": "Last invoice", "value": str(fmt.get("date_max", "n/a"))[:10]},
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
layout.chart(D.cleaning_sankey(cleaning), key="data_sankey")
st.caption(
    "Lines without a Customer ID (guest checkouts) cannot be linked to a customer, so they "
    "are dropped: the results describe registered customers only."
)

# ------------------------------------------------------------------ the business
st.subheader("Sales over time")
layout.chart(D.monthly_trend(data.table(C.EDA_MONTHLY_PARQUET)), key="data_monthly")

st.subheader("Customers and countries")
col_pareto, col_country = st.columns(2)
with col_pareto:
    layout.chart(D.pareto_curve(data.table(C.EDA_PARETO_PARQUET), eda), key="data_pareto")
with col_country:
    layout.chart(D.country_revenue(data.table(C.EDA_COUNTRIES_PARQUET)), key="data_countries")
layout.conclusion(eda.get("conclusion"))

st.divider()
st.page_link("pages/2_Demo.py", label=DEMO_LABEL, icon=":material/play_circle:")
st.caption(layout.DATA_CREDIT + " Built with free, open-source tools only.")
