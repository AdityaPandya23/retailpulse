"""Page chrome shared by every page: config + sidebar, header, KPI tiles, conclusion boxes."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from components import data, tour

DATA_CREDIT = (
    "Data: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) "
    "(Chen, 2012), CC BY 4.0. Registered customers only; guest checkouts are excluded."
)


def setup_page(title: str) -> None:
    """First call on every page: page config, Guided Tour toggle, mock badge, data credit."""
    st.set_page_config(page_title=f"{title} · RetailPulse", layout="wide")
    tour.sidebar_toggle()
    mock = data.mock_files()
    if mock:
        st.sidebar.warning(
            f"**Mock data** in use for {len(mock)} input(s): run "
            "`python -m retailpulse.pipeline` to build the real artifacts.",
            icon=":material/science:",
        )
        with st.sidebar.expander("Which files are mock?"):
            st.markdown("\n".join(f"- `{m}`" for m in mock))
    st.sidebar.caption(DATA_CREDIT)


def page_header(title: str, question: str | None = None, intro: str | None = None) -> None:
    st.title(title)
    if data.mock_files():
        st.badge("Mock data", icon=":material/science:", color="orange")
    if question:
        st.markdown(f"**{question}**")
    if intro:
        st.caption(intro)


def kpi_row(items: Iterable[dict]) -> None:
    """Row of stat tiles: dicts with label, value and optional help / delta."""
    items = list(items)
    for col, item in zip(st.columns(len(items)), items, strict=True):
        with col:
            st.metric(
                item["label"],
                item["value"],
                delta=item.get("delta"),
                delta_color=item.get("delta_color", "off"),
                help=item.get("help"),
                border=True,
            )


def conclusion(texts: str | Iterable[str | None] | None, title: str = "What this tells us") -> None:
    """Box with the plain-English metrics.json "conclusion" strings (None / empty skipped)."""
    if isinstance(texts, str):
        texts = [texts]
    lines = [t for t in (texts or []) if t]
    if not lines:
        return
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {t}" for t in lines))


def section_conclusion(section: str, *keys: str, title: str = "What this tells us") -> None:
    """Conclusion box from metrics.json: section["conclusion"] or section[key]["conclusion"]."""
    m = data.metrics(section)
    if not keys:
        texts = [m.get("conclusion")]
    else:
        texts = [(m.get(k) or {}).get("conclusion") for k in keys]
    conclusion(texts, title)


def chart(fig, key: str | None = None) -> None:
    """Render a figure with OUR template (theme=None stops Streamlit restyling it)."""
    st.plotly_chart(fig, theme=None, width="stretch", key=key)
