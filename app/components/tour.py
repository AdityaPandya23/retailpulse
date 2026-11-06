"""Guided Tour: a sidebar toggle (kept across pages) + numbered callouts next to key charts."""

from __future__ import annotations

import streamlit as st

STATE_KEY = "tour_on"  # plain session-state value: survives page switches
WIDGET_KEY = "tour_toggle"


def _sync() -> None:
    st.session_state[STATE_KEY] = bool(st.session_state.get(WIDGET_KEY, False))


def sidebar_toggle() -> bool:
    """Render the toggle in the sidebar; returns whether the tour is on."""
    st.session_state.setdefault(STATE_KEY, False)
    st.sidebar.toggle(
        "Guided Tour",
        value=st.session_state[STATE_KEY],
        key=WIDGET_KEY,
        on_change=_sync,
        help="Shows numbered notes next to the key charts, in demo order.",
    )
    return enabled()


def enabled() -> bool:
    return bool(st.session_state.get(STATE_KEY, False))


def callout(step: int, text: str, title: str | None = None) -> None:
    """A numbered note shown only while the tour is on (put it right above its chart)."""
    if not enabled():
        return
    head = f"**Tour {step}" + (f" · {title}**" if title else "**")
    st.info(f"{head}  \n{text}", icon=":material/tour:")
