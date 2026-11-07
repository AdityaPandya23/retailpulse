"""Page chrome shared by every page: config + sidebar (light / dark button), header, KPI
tiles, conclusion boxes and charts (recoloured for the dark theme)."""

from __future__ import annotations

import re
from collections.abc import Iterable

import plotly.io as pio
import streamlit as st

from components import data

THEME_OPTION = "theme.base"
# light template colour -> dark theme colour (text, grid, surface; series colours unchanged)
_DARK = {
    "#222222": "#FAFAFA",  # ink
    "#595959": "#C9CDD4",  # muted ink
    "#2A3F5F": "#FAFAFA",  # plotly_white text
    "#000000": "#FAFAFA",  # black ticks / outlines
    "BLACK": "#FAFAFA",
    "#E5E5E5": "#31333F",  # grid
    "#EBF0F8": "#31333F",  # plotly_white grid
    "#DFE8F3": "#31333F",
    "#FFFFFF": "#0E1117",  # surface
    "WHITE": "#0E1117",
}
_COLOUR_TOKEN = re.compile(r'"(#[0-9A-Fa-f]{6}|white|black)"')

DATA_CREDIT = (
    "Data: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) "
    "(Chen, 2012), CC BY 4.0. Registered customers only; guest checkouts are excluded."
)


def setup_page(title: str) -> None:
    """First call on every page: page config, light / dark button, mock badge, data credit."""
    st.set_page_config(page_title=f"{title} · RetailPulse", layout="wide")
    theme_button()
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


def is_dark() -> bool:
    return st.get_option(THEME_OPTION) == "dark"


def theme_button() -> None:
    """Sidebar button that flips the app between the light and the dark theme."""
    dark = is_dark()
    if st.sidebar.button(
        "Light mode" if dark else "Dark mode",
        key="theme_toggle",
        icon=":material/light_mode:" if dark else ":material/dark_mode:",
        width="stretch",
    ):
        st._config.set_option(THEME_OPTION, "light" if dark else "dark")
        st.rerun()


def dark_figure(fig):
    """Copy of a figure with the template's text / grid / surface colours swapped for dark."""
    text = pio.to_json(fig, validate=False)
    text = _COLOUR_TOKEN.sub(lambda m: f'"{_DARK.get(m.group(1).upper(), m.group(1))}"', text)
    text = text.replace("rgba(255,255,255,0.8)", "rgba(14,17,23,0.8)")
    return pio.from_json(text, skip_invalid=True)


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
    if is_dark():
        fig = dark_figure(fig)
    st.plotly_chart(fig, theme=None, width="stretch", key=key)
