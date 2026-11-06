"""Colour-blind-safe palette (Okabe-Ito) and the one Plotly template every figure uses.

Fixed colour per segment id (0 = most valuable) and per model, so a colour always means the
same thing on every page; colour follows the entity, never its rank on a chart.

Validated with the dataviz palette checker (light surface):
  * segments 0-2 (k = 3): PASS lightness / chroma / CVD separation (worst dE 11.4).
  * all 7 segment slots: #F0E442 (slot 6) is too light for marks on white and #CC79A7 vs
    #009E73 is a 7.6 deutan pair -> segment colours always come with the segment NAME
    (legend or label) and yellow markers get a dark outline (marker_line()).
  * classifiers: worst CVD pair 7.6 (qda vs naive_bayes) -> every model also has its own
    line dash (CLASSIFIER_DASH), so identity is never colour alone.
  * #E69F00 / #CC79A7 / #56B4E9 are below 3:1 contrast on white -> visible labels or a
    table next to the chart; text never uses series colours.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from retailpulse import contracts as C

# Okabe & Ito (2008), "Color Universal Design"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PINK = "#CC79A7"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
YELLOW = "#F0E442"
BLACK = "#000000"

# recessive greys for chrome, "all customers" / reference lines and segments beyond slot 6
GREY = "#7F7F7F"
LIGHT_GREY = "#BDBDBD"
GRID = "#E5E5E5"
INK = "#222222"  # primary text
MUTED_INK = "#595959"  # secondary text
SURFACE = "#FFFFFF"

SEGMENT_COLOURS: tuple[str, ...] = (BLUE, ORANGE, GREEN, PINK, SKY, VERMILLION, YELLOW)
OTHER_COLOUR = GREY  # segment id >= 7 (only with a config override k > 7): never a new hue

CLASSIFIER_COLOURS: dict[str, str] = dict(
    zip(C.CLASSIFIERS, (BLUE, ORANGE, GREEN, PINK, VERMILLION), strict=True)
)
CLASSIFIER_DASH: dict[str, str] = dict(
    zip(C.CLASSIFIERS, ("solid", "dash", "dot", "dashdot", "longdash"), strict=True)
)
REGRESSOR_COLOURS: dict[str, str] = dict(zip(C.REGRESSORS, (SKY, BLUE, VERMILLION), strict=True))
REGRESSOR_DASH: dict[str, str] = dict(zip(C.REGRESSORS, ("dot", "solid", "longdash"), strict=True))
REFERENCE_COLOUR = GREY  # chance line, mean-only baseline, "all customers"

# before / after the cutoff (timeline, time machine): blue vs vermillion (warm/cool pair)
PERIOD_COLOURS: dict[str, str] = {"before": BLUE, "after": VERMILLION}

# sequential (magnitude): one hue light -> dark; diverging (correlation): two poles + grey mid
SEQUENTIAL: list[list] = [[0.0, "#F2F7FB"], [0.5, "#6BAED6"], [1.0, "#08306B"]]
DIVERGING: list[list] = [[0.0, VERMILLION], [0.5, "#F2F2F2"], [1.0, BLUE]]

FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
TEMPLATE_NAME = "retailpulse"


def segment_colour(segment_id: int) -> str:
    """Fixed colour of a segment id (0 = most valuable); ids >= 7 share the neutral grey."""
    i = int(segment_id)
    return SEGMENT_COLOURS[i] if 0 <= i < len(SEGMENT_COLOURS) else OTHER_COLOUR


def model_colour(model: str) -> str:
    return CLASSIFIER_COLOURS.get(model) or REGRESSOR_COLOURS.get(model) or GREY


def marker_line(colour: str) -> dict:
    """2px surface ring on markers; light yellow gets a dark ring so it stays visible."""
    if colour.upper() == YELLOW:
        return {"color": INK, "width": 1}
    return {"color": SURFACE, "width": 1}


def _template() -> go.layout.Template:
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": LIGHT_GREY,
        "ticks": "outside",
        "tickcolor": LIGHT_GREY,
        "title": {"font": {"color": MUTED_INK, "size": 13}},
        "tickfont": {"color": MUTED_INK, "size": 12},
        "automargin": True,
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT_FAMILY, "color": INK, "size": 13},
            "title": {"font": {"size": 16, "color": INK}, "x": 0.0, "xanchor": "left"},
            "colorway": list(SEGMENT_COLOURS[:6]),
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "xaxis": axis,
            "yaxis": axis,
            "legend": {"bgcolor": "rgba(255,255,255,0.8)", "font": {"color": INK}},
            "hoverlabel": {"bgcolor": SURFACE, "font": {"color": INK}, "bordercolor": GREY},
            "hovermode": "closest",
            "margin": {"l": 60, "r": 20, "t": 60, "b": 50},
            "colorscale": {"sequential": SEQUENTIAL, "diverging": DIVERGING},
        },
        data={
            "scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})],
            "bar": [go.Bar(marker={"line": {"color": SURFACE, "width": 1}})],
        },
    )


def register_template() -> str:
    """Register the template and make it the default; returns the template string."""
    pio.templates[TEMPLATE_NAME] = _template()
    pio.templates.default = f"plotly_white+{TEMPLATE_NAME}"
    return pio.templates.default


def template() -> go.layout.Template:
    """plotly_white + the RetailPulse layer, as one Template object to set on a figure."""
    if TEMPLATE_NAME not in pio.templates:
        pio.templates[TEMPLATE_NAME] = _template()
    base = go.layout.Template(pio.templates["plotly_white"])
    base.update(pio.templates[TEMPLATE_NAME])
    return base
