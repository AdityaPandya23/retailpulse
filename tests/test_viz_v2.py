"""v2 figures + helpers (viz/segments, viz/predictions, viz/rules, viz/data_cube additions) on
tiny hand-made frames (never real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from retailpulse import contracts as C
from retailpulse.viz import data_cube as D
from retailpulse.viz import palette as P
from retailpulse.viz import predictions as VP
from retailpulse.viz import rules as VR
from retailpulse.viz import segments as S

NAMES = {0: "Champions", 1: "Regulars", 2: "Lost"}


def _is_empty(fig: go.Figure) -> bool:
    return len(fig.data) == 0 and len(fig.layout.annotations) == 1


@pytest.fixture
def segments() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 3000
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "segment": rng.integers(0, 3, n),
            "pc1": rng.normal(size=n),
            "pc2": rng.normal(size=n),
            "pc3": rng.normal(size=n),
        }
    )


@pytest.fixture
def profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": [0, 1, 2],
            "segment_name": list(NAMES.values()),
            "customer_share": [0.4, 0.31, 0.29],
            "revenue_share": [0.88, 0.075, 0.045],
        }
    )


# ------------------------------------------------------------------ segments
def test_pca_scatter_2d_star_and_sampling(segments):
    fig = S.pca_scatter_2d(
        segments, NAMES, [0.5, 0.2, 0.1], n=2000, star=[1.0, -2.0, 0.3], star_segment=2
    )
    seg_traces, star = fig.data[:-1], fig.data[-1]
    assert [t.name for t in seg_traces] == ["0 · Champions", "1 · Regulars", "2 · Lost"]
    assert sum(len(t.x) for t in seg_traces) == 2000  # at most n real customers
    assert all(t.marker.opacity == 0.25 for t in seg_traces)  # faded behind the star
    assert [t.marker.color for t in seg_traces] == [P.segment_colour(i) for i in range(3)]
    assert star.marker.symbol == "star" and star.marker.size == S.STAR_SIZE
    assert star.marker.color == P.segment_colour(2) and (star.x[0], star.y[0]) == (1.0, -2.0)
    assert "50% of variance" in fig.layout.xaxis.title.text
    # without a star: not faded, one trace per segment
    plain = S.pca_scatter_2d(segments, NAMES)
    assert len(plain.data) == 3 and plain.data[0].marker.opacity == 0.7
    assert _is_empty(S.pca_scatter_2d(segments.drop(columns="pc2"), NAMES))


def test_k_selection_panels_subset(segments):
    ksel = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "wcss": [100.0, 70.0, 60.0],
            "silhouette": [0.25, 0.29, 0.22],
            "davies_bouldin": [1.4, 1.3, 1.5],
            "calinski_harabasz": [2000.0, 2300.0, 2100.0],
        }
    )
    fig = S.k_selection_panels(ksel, 3, ("wcss", "silhouette"))
    titles = [a.text for a in fig.layout.annotations]
    assert any(t.startswith("WCSS") for t in titles) and any("Silhouette" in t for t in titles)
    assert not any("Davies" in t for t in titles)
    assert len(fig.data) == 4  # 2 score lines + 2 chosen-k markers
    assert "WCSS and silhouette" in fig.layout.title.text
    assert len(S.k_selection_panels(ksel, 3).data) == 8  # v1 default: every metric


def test_pc_meanings_sign_aware():
    load = pd.DataFrame(
        {
            "feature": ["recency", "frequency", "monetary"],
            "pc1": [-0.1, 0.7, 0.6],
            "pc2": [0.9, 0.1, -0.2],
        }
    )
    lines = S.pc_meanings(load, [0.52, 0.21], top=2)
    assert lines[0].startswith("**PC1** (52% of variance)")
    assert "higher order count, higher total spend" in lines[0]
    assert "higher recency" in lines[1] and "lower total spend" in lines[1]
    assert S.pc_meanings(None) == []
    assert "of variance" not in S.pc_meanings(load)[0]


def test_share_bars(profiles):
    fig = S.share_bars(profiles, NAMES)
    bars = [t for t in fig.data if t.x[0] is not None]
    legend = [t for t in fig.data if t.x[0] is None]
    assert [t.name for t in bars] == ["Share of customers", "Share of revenue"]
    assert list(bars[1].y) == [0.88, 0.075, 0.045]
    assert list(bars[0].marker.color) == [P.segment_colour(i) for i in range(3)]
    assert bars[0].marker.pattern.shape == "/" and bars[1].marker.pattern.shape == ""
    assert not any(t.showlegend for t in bars)
    # legend swatches are neutral grey: texture + lightness (not the hue) name the measure;
    # lightness via rgba because legend swatches ignore marker opacity (P6c visual pass)
    assert [t.name for t in legend] == ["Share of customers", "Share of revenue"]
    light, solid = (t.marker.color for t in legend)
    assert light.startswith("rgba(") and light.endswith(",0.45)") and solid == P.GREY
    assert all(t.marker.opacity is None for t in legend)
    assert fig.layout.barmode == "group"
    assert _is_empty(S.share_bars(profiles.drop(columns="revenue_share")))


def test_feature_meaning_covers_every_feature():
    assert set(C.FEATURE_COLUMNS) <= set(S.FEATURE_MEANING)


# ------------------------------------------------------------------ predictions
def _probs():
    return pd.DataFrame(
        {
            "model": ["logistic", "naive_bayes", "knn"],
            "p_return": [0.3, 0.5, 0.6],
            "threshold": [0.35, 0.4, 0.35],
            "chosen": [False, False, True],
        }
    )


def test_model_probability_bars():
    fig = VP.model_probability_bars(_probs(), 90)
    bars, ticks = fig.data
    assert list(bars.y) == ["knn ★ chosen", "naive_bayes", "logistic"]  # first model on top
    assert list(bars.marker.color) == [
        P.model_colour(m) for m in ("knn", "naive_bayes", "logistic")
    ]
    assert list(ticks.x) == [0.35, 0.4, 0.35] and ticks.marker.symbol == "line-ns"
    assert "90 days" in fig.layout.title.text
    no_thr = VP.model_probability_bars(_probs().assign(threshold=None))
    assert len(no_thr.data) == 1
    assert _is_empty(VP.model_probability_bars(None))


def test_simple_model_table():
    cls = {
        "best_model": "knn",
        "models": {
            "logistic": {
                "cv_auc_mean": 0.8,
                "cv_auc_std": 0.01,
                "test_auc": 0.79,
                "test_ap": 0.7,
                "accuracy": 0.7,
                "threshold": 0.3,
            },
            "knn": {
                "cv_auc_mean": 0.82,
                "cv_auc_std": 0.01,
                "test_auc": 0.78,
                "test_ap": 0.7,
                "accuracy": 0.71,
                "threshold": 0.35,
                "k": 75,
            },
        },
    }
    t = VP.simple_model_table(cls)
    assert list(t.columns) == [
        "Model",
        "Chosen",
        "CV ROC-AUC",
        "Test ROC-AUC",
        "Accuracy",
        "Threshold",
    ]
    assert list(t["Model"]) == ["logistic", "knn (k = 75)"] and list(t["Chosen"]) == ["", "★"]
    assert VP.simple_model_table({}).empty


def test_split_timeline_from_metrics_dates():
    cls = {
        "train_cutoff": "2011-06-11",
        "train_label_window": ["2011-06-11", "2011-09-09"],
        "test_cutoff": "2011-09-09",
        "test_label_window": ["2011-09-09", "2011-12-08"],
    }
    fig = VP.split_timeline(cls, "2009-12-01", 90)
    assert len(fig.data) == 4
    assert {t.y[0] for t in fig.data} == {"Train (model learns)", "Test (model is judged)"}
    assert sum(t.showlegend for t in fig.data) == 2  # history + outcome
    texts = " ".join(a.text for a in fig.layout.annotations)
    assert "train snapshot 2011-06-11" in texts and "cutoff 2011-09-09" in texts
    outcome = [t for t in fig.data if t.marker.color == P.PERIOD_COLOURS["after"]]
    assert all(t.x[0] == 89 * 86_400_000 or t.x[0] == 90 * 86_400_000 for t in outcome)
    assert _is_empty(VP.split_timeline({}))
    # label windows missing -> derived from the cutoffs + horizon
    assert (
        len(VP.split_timeline({"train_cutoff": "2011-06-11", "test_cutoff": "2011-09-09"}).data)
        == 4
    )


@pytest.fixture
def curve() -> pd.DataFrame:
    k = np.arange(1, 101)
    return pd.DataFrame(
        {"k": k, "train_error": 0.25 * (1 - np.exp(-k / 8)), "test_error": 0.3 + 0.1 / np.sqrt(k)}
    )


def test_knn_k_options(curve):
    assert VP.knn_k_options(curve, 75) == [1, 5, 20, 75, 100]
    assert VP.knn_k_options(curve, 20) == [1, 5, 20, 100]  # de-duplicated
    assert VP.knn_k_options(curve[curve["k"] <= 50], 75) == [1, 5, 20]  # only k in the curve
    assert VP.knn_k_options(curve, None) == [1, 5, 20, 100]
    assert VP.knn_k_options(None, 75) == []


def test_knn_accuracies_and_regime(curve):
    acc = VP.knn_accuracies(curve, 1)
    row = curve.iloc[0]
    assert acc["k"] == 1
    assert acc["train_accuracy"] == pytest.approx(1 - row["train_error"])
    assert acc["gap"] == pytest.approx(row["test_error"] - row["train_error"])
    assert VP.knn_accuracies(None, 5) == {}
    assert VP.OVERFIT in VP.knn_regime(1, 75)
    assert VP.BALANCE in VP.knn_regime(75, 75)
    assert VP.UNDERFIT in VP.knn_regime(100, 75)
    assert VP.knn_regime(5, None) == "k = 5."


def test_knn_simple_figure(curve):
    fig = VP.knn_simple_figure(curve, 20, 75)
    assert fig.layout.xaxis.type == "log"
    lines, dots = fig.data[:2], fig.data[2:]
    assert [t.mode for t in lines] == ["lines", "lines"]
    assert len(dots) == 2 and all(t.x[0] == 20 and t.marker.size == 16 for t in dots)
    vline = [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert len(vline) == 1 and vline[0].x0 == 75
    assert any(a.text == "best k = 75" for a in fig.layout.annotations)
    assert not VP.knn_simple_figure(curve, 20, None).layout.shapes
    assert _is_empty(VP.knn_simple_figure(None, 1, 75))


# ------------------------------------------------------------------ rules
@pytest.fixture
def rules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["overall", "overall", "overall", "segment:X"],
            "antecedents": ["CUP", "CUP + PLATE", "BOWL", "TIN"],
            "consequents": ["SAUCER", "MUG", "SPOON", "LID"],
            "support": [0.02, 0.011, 0.015, 0.05],
            "confidence": [0.7, 0.5, 0.4, 0.9],
            "lift": [50.0, 10.0, 3.0, 99.0],
            "antecedent_codes": ["1", "1,2", "3", "9"],
            "consequent_codes": ["4", "5", "6", "8"],
        }
    )


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["1", "2", "3", "4", "5", "6"],
            "description": ["CUP", "PLATE", "BOWL", "SAUCER", "MUG", "SPOON"],
            "n_invoices": [60, 50, 40, 30, 20, 10],
        }
    )


def test_overall_rules_and_top_table(rules):
    r = VR.overall_rules(rules)
    assert set(r["scope"]) == {"overall"} and list(r["lift"]) == [50.0, 10.0, 3.0]
    t = VR.top_rules_table(rules, 2)
    assert list(t.columns) == VR.RULE_TABLE_COLUMNS
    assert list(t["If a basket has"]) == ["CUP", "CUP + PLATE"]
    assert VR.top_rules_table(None).empty


def test_rule_explainer_derives_base_rates():
    top = {"antecedents": "A", "consequents": "B", "support": 0.02, "confidence": 0.7, "lift": 35.0}
    e = VR.rule_explainer(top, 20_000)
    assert set(e) == {"support", "confidence", "lift"}
    assert "2.0%" in e["support"] and "about 400 of 20,000" in e["support"]
    assert "70%" in e["confidence"]
    assert "2.0% of all baskets" in e["lift"] and "35.0×" in e["lift"]  # P(B) = 0.7 / 35
    assert "about" not in VR.rule_explainer(top)["support"]
    assert VR.rule_explainer(None) == {} and VR.rule_explainer({"support": 0.1}) == {}


def test_rules_bubble(rules):
    fig = VR.rules_bubble(rules)
    (t,) = fig.data
    assert len(t.x) == 3  # overall scope only
    assert list(t.marker.color) == [50.0, 10.0, 3.0] and t.marker.showscale
    assert max(t.marker.size) > min(t.marker.size)
    assert _is_empty(VR.rules_bubble(None))


def test_antecedent_products_and_rules_for_product(rules, products):
    ap = VR.antecedent_products(rules, products)
    assert list(ap.columns) == ["stock_code", "description", "n_rules"]
    assert ap.iloc[0].tolist() == ["1", "CUP", 2]  # in 2 overall rules
    assert set(ap["stock_code"]) == {"1", "2", "3"}  # segment rule "9" excluded
    assert dict(zip(ap["stock_code"], ap["description"], strict=True))["2"] == "PLATE"
    assert list(VR.rules_for_product(rules, "1")["it often also has"]) == ["SAUCER", "MUG"]
    assert list(VR.rules_for_product(rules, "2")["it often also has"]) == ["MUG"]
    assert VR.rules_for_product(rules, None).empty
    # without products.parquet: single-item rules name the product, multi-item fall back to code
    ap2 = VR.antecedent_products(rules)
    assert dict(zip(ap2["stock_code"], ap2["description"], strict=True)) == {
        "1": "CUP",
        "2": "2",
        "3": "BOWL",
    }
    assert VR.antecedent_products(None).empty


def test_hit_rate_bar():
    fig = VR.hit_rate_bar(
        {"hit_rate_at_5": 0.224, "popularity_hit_rate_at_5": 0.161, "n_eval_customers": 2194}
    )
    (t,) = fig.data
    assert list(t.y) == [0.224, 0.161] and list(t.text) == ["22.4%", "16.1%"]
    assert list(t.marker.color) == [VR.RULES_COLOUR, VR.BEST_SELLER_COLOUR]
    assert "2,194" in fig.layout.yaxis.title.text
    assert _is_empty(VR.hit_rate_bar({}))


# ------------------------------------------------------------------ data
def test_raw_columns_table():
    t = D.raw_columns_table()
    assert len(t) == 8 and list(t["Column (here)"]) == C.RAW_COLUMNS
    assert t.loc[0, "Column (Excel)"] == "Invoice" and "Customer ID" in set(t["Column (Excel)"])
    assert t["Meaning"].str.len().gt(5).all()


# ------------------------------------------------------------------ review fixes
def test_overall_rules_never_fall_back_to_segment_rules(rules):
    only_segments = rules[rules["scope"] != "overall"]
    assert VR.overall_rules(only_segments).empty  # same rule set as recommend_from_basket
    assert VR.top_rules_table(only_segments).empty
    assert VR.antecedent_products(only_segments).empty
    no_scope = rules.drop(columns="scope")  # no scope column: every rule counts
    assert len(VR.overall_rules(no_scope)) == 4


def test_rule_tables_are_in_percent(rules):
    t = VR.top_rules_table(rules, 1)
    assert t.loc[0, "Support (%)"] == pytest.approx(2.0)
    assert t.loc[0, "Confidence (%)"] == pytest.approx(70.0)
    assert VR.rules_for_product(rules, "3").loc[0, "Support (%)"] == pytest.approx(1.5)


def test_rule_text_escapes_product_names():
    top = {
        "antecedents": "*BOOMBOX IPOD",
        "consequents": "$5 TIN_BOX",
        "support": 0.02,
        "confidence": 0.5,
        "lift": 10.0,
    }
    e = VR.rule_explainer(top)
    assert r"\*BOOMBOX IPOD" in e["support"] and r"\$5 TIN\_BOX" in e["lift"]
    assert VR.rule_title(top) == r"**\*BOOMBOX IPOD → \$5 TIN\_BOX**"
    assert VR.rule_title({}) is None


def test_overall_invoices_and_default_product():
    assert VR.overall_invoices({"n_invoices": {"overall": 29962, "segment:X": 10}}) == 29962
    assert VR.overall_invoices({"n_invoices": 500}) == 500
    assert VR.overall_invoices({}) is None and VR.overall_invoices(None) is None
    assert VR.default_product({"antecedent_codes": "3,1"}, ["1", "3"]) == "3"
    assert VR.default_product({"antecedent_codes": "9"}, ["1", "3"]) == "1"
    assert VR.default_product(None, []) is None


def test_knn_best_k():
    assert VP.knn_best_k({"knn_k": 75}) == 75
    assert VP.knn_best_k({"models": {"knn": {"k": 20.0}}}) == 20
    assert VP.knn_best_k({}) is None and VP.knn_best_k(None) is None


# ------------------------------------------------------------------ P6c: narrow columns
# Visual pass at a 1280 px window: Home cards ~230 px, half columns ~400 px, [3, 2] ~480 px.
def test_home_cards_fit_a_narrow_column(segments):
    pca = S.pca_scatter_2d(segments, NAMES, [0.5, 0.2, 0.1], star=[0.0, 0.0], star_segment=0)
    assert len(pca.layout.title.text) <= 28  # the old title was clipped at ~230 px
    probs = VP.model_probability_bars(_probs(), 90)
    assert probs.layout.showlegend is False  # legend collided with the axis title
    assert list(probs.layout.xaxis.tickvals) == [0, 0.5, 1]  # 5 ticks were rotated
    assert len(VR.rules_bubble(_bubble_rules()).layout.title.text) <= 40


def _bubble_rules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["overall"] * 3,
            "antecedents": ["A", "B", "C"],
            "consequents": ["D", "E", "F"],
            "support": [0.01, 0.012, 0.02],
            "confidence": [0.4, 0.6, 0.9],
            "lift": [5.0, 20.0, 50.0],
        }
    )


def test_knn_axis_does_not_move_with_the_slider(curve):
    figs = [VP.knn_simple_figure(curve, k, 75) for k in (1, 20, int(curve["k"].max()))]
    ranges = {tuple(f.layout.xaxis.range) for f in figs}
    assert len(ranges) == 1  # same x range whatever k is selected
    ax = figs[0].layout.xaxis
    assert all(t in VP.KNN_TICKS for t in ax.tickvals) and 1 in ax.tickvals
    lo, hi = ax.range
    assert lo < 0 < np.log10(curve["k"].max()) < hi  # k = 1 and k max inside, not on the edge
    dots = figs[0].data[2:]
    assert all(t.cliponaxis is False for t in dots)  # the k = 1 train dot sits on 0%


def test_legends_leave_labels_and_titles_alone(segments, profiles):
    scree = S.scree_plot([0.5, 0.2, 0.1])
    assert scree.layout.legend.y < 0  # under the plot, not beside the title
    pareto = pd.DataFrame({"customer_share": [0, 0.01, 0.2, 1], "revenue_share": [0, 0.3, 0.8, 1]})
    p = D.pareto_curve(pareto, {"top_1pct_revenue_share": 0.3})
    assert p.layout.legend.y < 0  # inside the plot it covered the "top 1%" label
    p3 = S.pca_scatter_3d(segments, NAMES, [0.5, 0.2, 0.1])
    assert p3.layout.legend.orientation == "h"  # a right legend squeezed the 3D scene


def test_boxplots_wrap_names_instead_of_a_legend():
    rng = np.random.default_rng(42)
    joined = pd.DataFrame(
        {"segment": np.repeat([0, 1, 2], 50), "monetary": rng.lognormal(6, 1.5, 150)}
    )
    names = {0: "Champions", 1: "Regulars", 2: "Lost / Hibernating"}
    fig = S.segment_boxplots(joined, "monetary", names, log=True)
    assert fig.layout.showlegend is False
    assert fig.layout.xaxis.tickangle == 0
    assert [t.name for t in fig.data] == [
        "0 · Champions",
        "1 · Regulars",
        "2 · Lost /<br>Hibernating",
    ]
    assert fig.layout.yaxis.type == "log" and fig.layout.yaxis.dtick == 1
