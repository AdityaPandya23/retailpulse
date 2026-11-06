"""K-Means customer segmentation on features_cutoff (pure functions: frames in, frames out).

Preprocessing = log1p on the clustering features that are also in contracts.SKEWED_FEATURES,
then StandardScaler. Both live inside one sklearn Pipeline with KMeans, so the saved model
accepts RAW feature values (the P4 What-if sliders feed raw numbers).

Segment ids are ordered by revenue share (0 = most valuable) so colours stay fixed.
k and the names come from the automatic rules below (choose_k, name_segments) unless set in
config.yaml (segment.k / segment.names); the evidence is k_selection.parquet, metrics.json
segment.k_candidates / k_choice_reason and profiles.parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from retailpulse import contracts as C
from retailpulse.config import ALPHA, SEED, SEGMENT, SegmentConfig
from retailpulse.features import log1p_skewed
from retailpulse.segment import baseline as B
from retailpulse.segment import validate as V
from retailpulse.segment.baseline import cluster_quality

MIN_SEGMENT_SHARE = 0.03  # every segment must hold >= 3% of customers to be eligible
SILHOUETTE_WINDOW = 0.02  # k within this of the best silhouette compete on stability (ARI)
NAME_THRESHOLD = 0.5  # |z| of a segment mean (in customer-level SD units) that counts as high/low


# ---------------- preprocessing (module-level so the fitted pipeline pickles) ----------------
def log_columns(features) -> list[str]:
    """Clustering features that get log1p (in SKEWED_FEATURES), in clustering order."""
    return [c for c in features if c in C.SKEWED_FEATURES]


def log1p_frame(X, features: list[str], log_cols: list[str]) -> pd.DataFrame:
    """Raw clustering features (DataFrame or array in `features` order) -> log1p'd frame."""
    if isinstance(X, pd.DataFrame):
        df = X[list(features)].astype("float64")
    else:
        df = pd.DataFrame(np.asarray(X, dtype="float64"), columns=list(features))
    return log1p_skewed(df, columns=list(log_cols))


def preprocessor(features) -> Pipeline:
    features = list(features)
    log1p = FunctionTransformer(
        log1p_frame,
        kw_args={"features": features, "log_cols": log_columns(features)},
        feature_names_out="one-to-one",
    )
    return Pipeline([("log1p", log1p), ("scale", StandardScaler())])


def segment_pipeline(features, k: int, seed: int = SEED) -> Pipeline:
    """log1p -> StandardScaler -> KMeans(k, n_init=10, random_state=seed)."""
    steps = preprocessor(features).steps
    return Pipeline([*steps, ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed))])


def predict_segment(model: dict, raw: pd.DataFrame) -> np.ndarray:
    """Saved model dict + raw feature rows -> segment ids (0 = most valuable)."""
    labels = model["pipeline"].predict(raw[model["features"]])
    return np.array([model["label_to_segment"][int(lab)] for lab in labels], dtype="int64")


# ---------------- choosing k ----------------
def min_share(labels: np.ndarray, k: int) -> float:
    return float(np.bincount(labels, minlength=k).min() / len(labels))


def scan_k(X: np.ndarray, k_range: tuple[int, int], seed: int = SEED):
    """K-Means for every k in k_range (inclusive) -> (k_selection frame, {k: labels})."""
    rows, labels_by_k = [], {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels_by_k[k] = km.labels_
        rows.append(
            {
                "k": k,
                "wcss": float(km.inertia_),
                **cluster_quality(X, km.labels_),
                "min_segment_share": min_share(km.labels_, k),
            }
        )
    return pd.DataFrame(rows), labels_by_k


def choose_k(
    candidates: pd.DataFrame,
    min_segment_share: float = MIN_SEGMENT_SHARE,
    window: float = SILHOUETTE_WINDOW,
) -> tuple[int, str, pd.DataFrame]:
    """Automatic k rule on a table with k, silhouette, min_segment_share, ari_mean.

    1. eligible = every segment holds >= min_segment_share of customers;
    2. best = highest silhouette among eligible k;
    3. candidates = eligible k with silhouette >= best - window;
    4. pick the highest mean ARI; ties -> the smaller k.
    No eligible k -> highest silhouette in the range (stated in the reason).
    Returns (k, plain-English reason, table with eligible / in_window / chosen flags).
    """
    t = candidates.sort_values("k").reset_index(drop=True).copy()
    t["eligible"] = t["min_segment_share"] >= min_segment_share
    lo, hi = int(t["k"].min()), int(t["k"].max())
    ari = t["ari_mean"].fillna(-np.inf).round(6)
    pool = t[t["eligible"]]
    if pool.empty:
        k = int(t.loc[t["silhouette"].idxmax(), "k"])
        t["in_window"] = False
        t["chosen"] = t["k"] == k
        sil = float(t.loc[t["chosen"], "silhouette"].iloc[0])
        reason = (
            f"No k in {lo}-{hi} keeps every segment at >= {min_segment_share:.0%} of customers, "
            f"so the rule falls back to the highest silhouette: k = {k} ({sil:.3f})."
        )
        return k, reason, t
    best = pool.loc[pool["silhouette"].idxmax()]
    t["in_window"] = t["eligible"] & (t["silhouette"] >= best["silhouette"] - window - 1e-12)
    window_rows = t[t["in_window"]]
    top_ari = ari[t["in_window"]].max()
    tied = window_rows[ari[t["in_window"]] == top_ari]
    k = int(tied["k"].min())
    t["chosen"] = t["k"] == k
    elig = ", ".join(str(int(v)) for v in pool["k"])
    reason = (
        f"Among k = {lo}-{hi}, every segment holds >= {min_segment_share:.0%} of customers for "
        f"k = {elig}; the best silhouette is k = {int(best['k'])} ({best['silhouette']:.3f})."
    )
    if len(window_rows) == 1:
        reason += f" No other eligible k is within {window} of it, so k = {k}."
    else:
        ks = ", ".join(str(int(v)) for v in window_rows["k"])
        chosen_ari = float(t.loc[t["chosen"], "ari_mean"].iloc[0])
        reason += (
            f" k = {ks} are within {window} silhouette of it; k = {k} has the highest mean "
            f"stability ARI ({chosen_ari:.3f})"
            + (", tie broken by the smaller k." if len(tied) > 1 else ".")
        )
    return k, reason, t


# ---------------- ids and names ----------------
def order_by_revenue(labels: np.ndarray, monetary) -> dict[int, int]:
    """{kmeans label: segment id}, segment 0 = largest revenue share (ties: smaller label)."""
    rev = pd.Series(np.asarray(monetary, dtype="float64")).groupby(np.asarray(labels)).sum()
    order = sorted(rev.index, key=lambda lab: (-rev[lab], lab))
    return {int(lab): i for i, lab in enumerate(order)}


def name_one(z: pd.Series, t: float = NAME_THRESHOLD) -> str:
    """One segment's name from its mean z-scores (missing features count as average)."""
    g = {c: float(z.get(c, 0.0)) for c in C.FEATURE_COLUMNS}
    value = (g["frequency"] + g["monetary"]) / 2
    if g["recency"] < -t and value > t:
        return "Champions"
    if g["monetary"] > t and g["recency"] > t:
        return "At Risk High-Value"
    if g["tenure"] < -t and g["recency"] < -t:
        return "New Customers"
    if g["recency"] > t and value < 0:
        return "Lost / Hibernating"
    if g["avg_basket_value"] > t and g["frequency"] < 0:
        return "Big-Basket Occasionals"
    return "Regulars"


NAMING_RULE = (
    "Segment means of the log1p + standardised clustering features (z = SD units vs all "
    f"customers; high/low = |z| > {NAME_THRESHOLD}), first match wins: low recency + high "
    "frequency/monetary -> Champions; high monetary + high recency -> At Risk High-Value; low "
    "tenure + low recency -> New Customers; high recency + below-average value -> Lost / "
    "Hibernating; high avg_basket_value + below-average frequency -> Big-Basket Occasionals; "
    "otherwise Regulars. Duplicates get ' 2', ' 3' in segment-id (revenue) order."
)


def unique_names(names: dict[int, str]) -> dict[int, str]:
    """Make names unique: the lowest segment id keeps a name, later ones get ' 2', ' 3', ..."""
    out, used = {}, set()
    for seg in sorted(names):
        name, n = names[seg], 1
        while name in used:
            n += 1
            name = f"{names[seg]} {n}"
        out[seg] = name
        used.add(name)
    return out


def name_segments(z_means: pd.DataFrame, overrides: dict[int, str] | None = None) -> dict:
    """{segment id: unique name}; `overrides` (config.yaml segment.names) win."""
    names = {int(s): name_one(z_means.loc[s]) for s in z_means.index}
    for seg, name in (overrides or {}).items():
        if int(seg) in names:
            names[int(seg)] = name
    return unique_names(names)


# ---------------- end-to-end (pure) ----------------
@dataclass
class SegmentationResult:
    tables: dict[Path, pd.DataFrame]
    metrics: dict
    model: dict
    extras: dict = field(default_factory=dict)


def _profiles(f: pd.DataFrame, segment: np.ndarray, names: dict, bought: np.ndarray):
    g = f.assign(segment=segment, uk=(f["non_uk_share"] < 0.5), bought=bought).groupby("segment")
    prof = g[C.FEATURE_COLUMNS].mean()
    n = g.size()
    prof.insert(0, "segment_name", [names[s] for s in prof.index])
    prof.insert(1, "n_customers", n.astype("int64"))
    prof.insert(2, "customer_share", n / n.sum())
    prof.insert(3, "revenue_share", g["monetary"].sum() / f["monetary"].sum())
    prof["median_monetary"] = g["monetary"].median()
    prof["uk_customer_share"] = g["uk"].mean()
    prof["bought_again_share"] = g["bought"].mean()  # outcome in the label window (descriptive)
    return prof.rename_axis("segment").reset_index()


def segment_customers(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: SegmentConfig = SEGMENT,
    seed: int = SEED,
    n_subsamples: int = 10,
    subsample_frac: float = 0.8,
    cv_folds: int = 5,
    alpha: float = ALPHA,
) -> SegmentationResult:
    """features_cutoff (+ labels for the baseline AUC) -> every segment artifact + metrics."""
    C.require_columns(features, C.FEATURES_COLUMNS, "features_cutoff")
    C.require_columns(labels, C.LABELS_COLUMNS, "labels")
    f = features.reset_index(drop=True)
    feats = list(cfg.features)
    raw = f[feats]
    X = preprocessor(feats).fit_transform(raw)

    k_sel, labels_by_k = scan_k(X, cfg.k_range, seed)
    k_sel["ari_mean"] = np.nan
    k_sel["ari_std"] = np.nan
    lo, hi = cfg.k_choice_range
    cand_rows, ari_by_k = [], {}
    for k in range(lo, hi + 1):
        ref = labels_by_k.get(k)
        if ref is None:
            ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_
            labels_by_k[k] = ref
        q = cluster_quality(X, ref)
        ari_by_k[k] = V.ari_summary(
            V.stability_ari(X, k, n_subsamples, subsample_frac, seed, reference=ref)
        )
        k_sel.loc[k_sel["k"] == k, ["ari_mean", "ari_std"]] = [
            ari_by_k[k]["mean"],
            ari_by_k[k]["std"],
        ]
        cand_rows.append(
            {
                "k": k,
                "silhouette": q["silhouette"],
                "min_segment_share": min_share(ref, k),
                "ari_mean": ari_by_k[k]["mean"],
            }
        )
    auto_k, auto_reason, cand = choose_k(pd.DataFrame(cand_rows))
    if cfg.k is None:
        k, k_reason, k_source = auto_k, auto_reason, "automatic rule"
    else:
        k, k_source = int(cfg.k), "config.yaml"
        k_reason = f"k = {k} set in config.yaml (the automatic rule would pick k = {auto_k})."

    pipe = segment_pipeline(feats, k, seed).fit(raw)
    km_labels = pipe.named_steps["kmeans"].labels_
    label_to_segment = order_by_revenue(km_labels, f["monetary"])
    segment = np.array([label_to_segment[int(lab)] for lab in km_labels], dtype="int64")
    if k not in ari_by_k:
        ari_by_k[k] = V.ari_summary(
            V.stability_ari(X, k, n_subsamples, subsample_frac, seed, reference=km_labels)
        )

    # names from segment means in the scaled space (= z vs the customer-level mean / sd)
    z_means = pd.DataFrame(X, columns=feats).groupby(segment).mean()
    names = name_segments(z_means, cfg.names)
    names_source = "config.yaml" if cfg.names else "automatic rule"

    rfm = B.rfm_scores(f)
    rfm_group = B.rfm_groups(rfm["rfm_score"], k)
    y = (
        f[["customer_id"]]
        .merge(labels[["customer_id", "returned_flag"]], on="customer_id", how="left")[
            "returned_flag"
        ]
        .fillna(0)
        .astype("int64")
        .to_numpy()
    )
    baseline = B.compare_baseline(X, segment, rfm_group, y, seed=seed, folds=cv_folds)
    quality = {
        m: baseline["kmeans"][m] for m in ("silhouette", "davies_bouldin", "calinski_harabasz")
    }

    coords, explained, loadings = V.pca_summary(X, feats, seed)
    chi2 = V.chi2_segment_uk(segment, f["non_uk_share"], alpha)

    seg_names = np.array([names[s] for s in segment], dtype=object)
    segments = pd.DataFrame(
        {
            "customer_id": f["customer_id"].astype("int64"),
            "segment": segment,
            "segment_name": seg_names,
            "r_score": rfm["r_score"].to_numpy(),
            "f_score": rfm["f_score"].to_numpy(),
            "m_score": rfm["m_score"].to_numpy(),
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "pc3": coords[:, 2],
            "rfm_score": rfm["rfm_score"].to_numpy(),
            "rfm_group": rfm_group,
        }
    )
    profiles = _profiles(f, segment, names, y)

    top = profiles.iloc[0]
    ari = ari_by_k[k]
    conclusion = (
        f"K-Means on {len(feats)} log1p + standardised features splits the {len(f):,} customers "
        f"into k = {k} segments ({k_source}; silhouette {quality['silhouette']:.2f}, mean "
        f"subsample ARI {ari['mean']:.2f}); '{top['segment_name']}' is "
        f"{top['customer_share']:.0%} of customers but {top['revenue_share']:.0%} of revenue."
    )
    metrics = {
        "k": k,
        "k_source": k_source,
        "k_choice_reason": k_reason,
        "k_rule": (
            f"within k = {lo}-{hi}: keep k whose smallest segment >= {MIN_SEGMENT_SHARE:.0%} "
            f"of customers; best silhouette; eligible k within {SILHOUETTE_WINDOW} of it "
            "compete on mean ARI (ties -> smaller k); none eligible -> best silhouette."
        ),
        "k_candidates": cand[
            ["k", "silhouette", "min_segment_share", "ari_mean", "eligible", "in_window", "chosen"]
        ].to_dict(orient="records"),
        **quality,
        "wcss": float(pipe.named_steps["kmeans"].inertia_),
        "stability_ari": ari,
        "stability_ari_by_k": {
            kk: {"mean": v["mean"], "std": v["std"]} for kk, v in ari_by_k.items()
        },
        "stability_method": (
            f"{n_subsamples} random {subsample_frac:.0%} subsamples (np.random.default_rng("
            f"{seed})); K-Means fit on each subsample predicts all customers; ARI vs the full fit."
        ),
        "pca_explained_variance": explained,
        "pca_cumulative_variance": [float(v) for v in np.cumsum(explained)],
        "baseline": baseline,
        "chi2_segment_uk": chi2,
        "segment_names": names,
        "names_source": names_source,
        "naming_rule": NAMING_RULE,
        "segment_sizes": {
            int(s): int(n)
            for s, n in zip(profiles["segment"], profiles["n_customers"], strict=True)
        },
        "features_used": feats,
        "log1p_features": log_columns(feats),
        "excluded_from_clustering": [c for c in C.FEATURE_COLUMNS if c not in feats],
        "scaling": (
            f"log1p on {', '.join(log_columns(feats))}, then StandardScaler (z-scores) fitted on "
            f"all {len(f):,} features_cutoff customers; the same steps sit in the saved pipeline."
        ),
        "n_customers": len(f),
        "seed": seed,
        "override": (
            "k and names are chosen by the rules above; to override, set segment.k and/or "
            "segment.names ({id: name}) in config.yaml and rerun the pipeline."
        ),
        "conclusion": conclusion,
    }
    model = {
        "pipeline": pipe,
        "features": feats,
        "label_to_segment": label_to_segment,
        "segment_names": names,
    }
    tables = {
        C.SEGMENTS_PARQUET: segments,
        C.SEGMENT_PROFILES_PARQUET: profiles,
        C.K_SELECTION_PARQUET: k_sel,
        C.PCA_LOADINGS_PARQUET: loadings,
    }
    return SegmentationResult(tables=tables, metrics=metrics, model=model, extras={"X": X})
