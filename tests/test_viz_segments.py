"""Page 2 figures (viz/segments.py) on the MOCK artifacts and tiny hand-made frames."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from retailpulse import contracts as C
from retailpulse.viz import common as VC
from retailpulse.viz import palette as P
from retailpulse.viz import segments as S


def _mock_parquet(path):
    file = C.mock_path(path)
    if not file.exists():
        pytest.skip("artifacts/mock/ not generated")
    return pd.read_parquet(file)


def _is_empty_figure(fig: go.Figure) -> bool:
    return len(fig.data) == 0 and len(fig.layout.annotations) == 1


@pytest.fixture
def seg_metrics():
    file = C.mock_path(C.METRICS_JSON)
    if not file.exists():
        pytest.skip("artifacts/mock/ not generated")
    return json.loads(file.read_text(encoding="utf-8"))["segment"]


@pytest.fixture
def segments():
    return _mock_parquet(C.SEGMENTS_PARQUET)


@pytest.fixture
def profiles():
    return _mock_parquet(C.SEGMENT_PROFILES_PARQUET)


@pytest.fixture
def names(seg_metrics, profiles):
    return VC.segment_names(seg_metrics, profiles)


@pytest.fixture
def joined(segments):
    return S.segment_feature_frame(_mock_parquet(C.FEATURES_CUTOFF_PARQUET), segments)


# ------------------------------------------------------------------ PCA
def test_sample_points_is_stratified_seeded_and_capped():
    df = pd.DataFrame({"segment": np.repeat([0, 1, 2], [6000, 3000, 1000]), "v": range(10000)})
    a, b = S.sample_points(df, 3000), S.sample_points(df, 3000)
    assert len(a) == 3000 and a.equals(b)  # seed 42 -> reproducible
    shares = a["segment"].value_counts(normalize=True).sort_index()
    assert shares.round(2).tolist() == [0.6, 0.3, 0.1]
    assert len(S.sample_points(df.head(10), 3000)) == 10


def test_pca_scatter_one_trace_per_segment(segments, names, seg_metrics):
    fig = S.pca_scatter_3d(segments, names, seg_metrics["pca_explained_variance"])
    assert [t.type for t in fig.data] == ["scatter3d"] * len(names)
    for sid, trace in zip(names, fig.data, strict=True):
        assert trace.marker.color == P.segment_colour(sid)
        assert names[sid] in trace.name
    assert "% of variance" in fig.layout.scene.xaxis.title.text
    assert sum(len(t.x) for t in fig.data) == len(segments)
    assert _is_empty_figure(S.pca_scatter_3d(segments.drop(columns="pc3"), names))


def test_pca_scatter_samples_large_frames(names):
    rng = np.random.default_rng(0)
    big = pd.DataFrame(
        {
            "customer_id": range(5000),
            "segment": rng.integers(0, len(names), 5000),
            "pc1": rng.normal(size=5000),
            "pc2": rng.normal(size=5000),
            "pc3": rng.normal(size=5000),
        }
    )
    fig = S.pca_scatter_3d(big, names)
    assert sum(len(t.x) for t in fig.data) <= S.MAX_POINTS


def test_biplot_and_scree(segments, seg_metrics):
    loadings = _mock_parquet(C.PCA_LOADINGS_PARQUET)
    fig = S.pca_biplot(loadings, segments, seg_metrics["pca_explained_variance"])
    arrows = [a for a in fig.layout.annotations if a.showarrow]
    assert len(arrows) == len(loadings)
    labels = {a.text for a in fig.layout.annotations if not a.showarrow}
    assert S.feature_label("monetary", False) in labels
    assert len(S.pca_biplot(loadings).data) == 1  # arrows alone, no point cloud
    scree = S.scree_plot(seg_metrics["pca_explained_variance"])
    assert [t.type for t in scree.data] == ["bar", "scatter"]
    assert max(scree.data[1].y) <= 1 and scree.layout.yaxis.range[1] <= 1.1  # one 0-100% axis
    assert _is_empty_figure(S.scree_plot(None))
    assert _is_empty_figure(S.pca_biplot(None))


# ------------------------------------------------------------------ k selection
def test_k_selection_marks_chosen_k_on_every_panel(seg_metrics):
    ksel = _mock_parquet(C.K_SELECTION_PARQUET)
    k = seg_metrics["k"]
    fig = S.k_selection_panels(ksel, k)
    chosen = [t for t in fig.data if t.name == f"Chosen k = {k}"]
    assert len(chosen) == 4  # mock: WCSS, silhouette, DB, CH
    assert all(list(t.x) == [k] for t in chosen)
    assert sum(t.showlegend for t in chosen) == 1
    assert len(fig.layout.shapes) == 4  # dashed vline per panel


def test_k_selection_uses_optional_ari_with_error_bars():
    ksel = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "wcss": [3.0, 2.0, 1.0],
            "silhouette": [0.3, 0.4, 0.2],
            "davies_bouldin": [1.2, 1.0, 1.3],
            "calinski_harabasz": [10.0, 12.0, 9.0],
            "ari_mean": [np.nan, 0.9, 0.8],
            "ari_std": [np.nan, 0.01, 0.05],
            "min_segment_share": [0.4, 0.2, 0.1],
        }
    )
    fig = S.k_selection_panels(ksel, 3)
    scores = [t for t in fig.data if t.name == "Score per k"]
    assert len(scores) == 6
    ari = [t for t in scores if t.error_y and t.error_y.array is not None]
    assert len(ari) == 1 and list(ari[0].x) == [3, 4]
    assert _is_empty_figure(S.k_selection_panels(None, 3))
    assert len(S.k_selection_panels(ksel, None).data) == 6  # no chosen k -> no markers


# ------------------------------------------------------------------ profiles
def test_profile_cards_optional_columns(profiles, names):
    cards = S.profile_cards(profiles, names)
    assert [c["segment"] for c in cards] == list(names)
    assert cards[0]["colour"] == P.segment_colour(0)
    labels = [k for k, _ in cards[0]["lines"]]
    assert "Bought again (90 days)" not in labels  # mock has no bought_again_share
    richer = profiles.assign(bought_again_share=0.5, median_monetary=100.0, uk_customer_share=0.9)
    labels = [k for k, _ in S.profile_cards(richer, names)[0]["lines"]]
    assert {"Bought again (90 days)", "Median spend", "UK customers"} <= set(labels)
    assert S.profile_cards(None, names) == []


def test_segment_boxplots_one_box_per_segment(joined, names):
    fig = S.segment_boxplots(joined, "monetary", names)
    assert [t.type for t in fig.data] == ["box"] * len(names)
    assert [t.marker.color for t in fig.data] == [P.segment_colour(s) for s in names]
    log = VC.is_skewed(joined["monetary"])
    assert (fig.layout.yaxis.type == "log") == log
    lin = S.segment_boxplots(joined, "monetary", names, log=False)
    assert lin.layout.yaxis.type != "log"
    assert _is_empty_figure(S.segment_boxplots(joined, "no_such_feature", names))
    assert _is_empty_figure(S.segment_boxplots(None, "monetary", names))


def test_parallel_profiles_scaled_0_1(profiles, names):
    fig = S.parallel_profiles(profiles, names)
    assert len(fig.data) == len(names)
    ys = np.concatenate([np.asarray(t.y, dtype=float) for t in fig.data])
    assert ys.min() >= 0 and ys.max() <= 1
    assert all(len(t.x) == len(C.FEATURE_COLUMNS) for t in fig.data)
    assert _is_empty_figure(S.parallel_profiles(None, names))


def test_names_fallback_when_metrics_have_none(profiles):
    fig = S.parallel_profiles(profiles, None)  # names from the profiles frame
    assert len(fig.data) == profiles["segment"].nunique()


# ------------------------------------------------------------------ baseline + chi-square
def test_baseline_table_with_and_without_optional_metrics(seg_metrics):
    mock = S.baseline_table(seg_metrics["baseline"])
    assert list(mock.columns) == ["Metric", "K-Means segments", "RFM-quartile baseline"]
    assert len(mock) == 2  # mock: silhouette + AUC only
    full = S.baseline_table(
        {
            "kmeans": {
                "silhouette": 0.28,
                "davies_bouldin": 1.3,
                "calinski_harabasz": 2335,
                "auc_returned": 0.718,
                "auc_returned_std": 0.013,
            },
            "rfm_quartile": {"silhouette": 0.17, "auc_returned": 0.745},
        }
    )
    assert len(full) == 4 and "± 0.013" in full.iloc[3]["K-Means segments"]
    assert full.iloc[1]["RFM-quartile baseline"] == "n/a"
    assert S.baseline_table({}).empty


def test_uk_share_bar(names):
    chi2 = {"uk_share_by_segment": {str(s): 0.9 for s in names}}
    fig = S.uk_share_bar(chi2, names)
    assert len(fig.data[0].x) == len(names)
    assert list(fig.data[0].marker.color) == [P.segment_colour(s) for s in names]
    assert _is_empty_figure(S.uk_share_bar({}, names))  # mock has no shares


# ------------------------------------------------------------------ stats
def test_qq_plot_panels_and_reference_line():
    qq = _mock_parquet(C.QQ_PARQUET)
    var = qq["variable"].iloc[0]
    fig = S.qq_plot(qq, var)
    n_dist = qq.loc[qq["variable"] == var, "dist"].nunique()
    assert len(fig.data) == 2 * n_dist
    refs = [t for t in fig.data if t.name == "Perfect fit (45° line)"]
    assert all(list(t.x) == list(t.y) for t in refs)
    assert _is_empty_figure(S.qq_plot(qq, "nothing"))


def test_frequency_fit_bars_and_two_fits():
    fig = S.frequency_fit(_mock_parquet(C.FREQUENCY_FIT_PARQUET))
    assert [t.type for t in fig.data] == ["bar", "scatter", "scatter"]
    assert {t.name for t in fig.data[1:]} == {"Poisson fit", "Negative binomial fit"}
    assert _is_empty_figure(S.frequency_fit(None))


def test_correlation_heatmaps_share_one_diverging_scale():
    corr = _mock_parquet(C.CORRELATION_PARQUET)
    fig = S.correlation_heatmaps(corr)
    assert [t.type for t in fig.data] == ["heatmap", "heatmap"]
    ca = fig.layout.coloraxis
    assert (ca.cmin, ca.cmax, ca.cmid) == (-1, 1, 0)
    assert all(t.coloraxis == "coloraxis" for t in fig.data)
    assert len(S.correlation_heatmaps(corr, ["pearson"]).data) == 1
    assert _is_empty_figure(S.correlation_heatmaps(corr, ["kendall"]))


def test_histogram_polygon_log_and_linear():
    hist = _mock_parquet(C.EDA_HISTOGRAMS_PARQUET)
    for feature in hist["feature"].unique():
        fig = S.histogram_polygon(hist, feature)
        assert [t.type for t in fig.data] == ["bar", "scatter"]
    assert _is_empty_figure(S.histogram_polygon(hist, "nothing"))


# ------------------------------------------------------------------ live test picker
def test_compare_segments_runs_both_tests(joined, names):
    a, b = list(names)[:2]
    r = S.compare_segments(joined, a, b, "monetary", names)
    assert {"welch_t", "mann_whitney", "conclusion"} <= set(r)
    assert names[a] in r["conclusion"]
    table = S.result_table(r)
    assert list(table["Test"]) == ["Welch's t-test (means)", "Mann-Whitney U (ranks)"]


def test_compare_segments_edge_cases(joined, names):
    a = list(names)[0]
    same = S.compare_segments(joined, a, a, "monetary", names)
    assert "two different segments" in same["conclusion"] and S.result_table(same).empty
    tiny = joined[joined["segment"] == a].head(1)
    few = S.compare_segments(pd.concat([tiny, joined]), a, 999, "monetary", names)
    assert "Not enough customers" in few["conclusion"]
    assert "No customer features" in S.compare_segments(None, a, 1, "monetary")["conclusion"]


def test_two_segment_distribution(joined, names):
    a, b = list(names)[:2]
    fig = S.two_segment_distribution(joined, a, b, "monetary", names)
    assert [t.type for t in fig.data] == ["violin", "violin"]
    assert [t.line.color for t in fig.data] == [P.segment_colour(a), P.segment_colour(b)]
    assert _is_empty_figure(S.two_segment_distribution(joined, 998, 999, "monetary", names))


def test_biplot_labels_one_line_apart(segments):
    """Visual pass: feature labels of arrows pointing the same way overlapped."""
    load = pd.DataFrame(
        {
            "feature": ["monetary", "frequency", "n_distinct_products"],
            "pc1": [0.5] * 3,
            "pc2": [0.10, 0.11, 0.12],
        }
    )
    fig = S.pca_biplot(load, segments)
    ys = sorted(a.y for a in fig.layout.annotations if a.text and not a.showarrow)
    y_range = segments["pc2"].max() - segments["pc2"].min()
    assert min(b - a for a, b in zip(ys[:-1], ys[1:], strict=True)) >= 0.05 * y_range


def test_parallel_profiles_legend_not_beside_plot(profiles):
    """Visual pass: a side legend squeezed the half-width plot to ~120 px."""
    assert S.parallel_profiles(profiles).layout.legend.orientation == "h"
