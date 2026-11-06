"""P3b prediction tests on synthetic data only (tests/conftest.py fixtures, hand-made frames)."""

from __future__ import annotations

import io
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from retailpulse import contracts as C
from retailpulse.config import CUTOFF, HORIZON_DAYS
from retailpulse.features import build_features, build_labels, log1p_skewed
from retailpulse.predict import classify, dataset, regress
from retailpulse.predict.backtest import backtest, decile_table
from retailpulse.predict.stage import compute

K_MAX = 20  # small KNN scan keeps the tests fast
OFFSET = 90


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variables are collinear")
        yield


@pytest.fixture(scope="module")
def frames(synthetic_shop):
    sales, returns = synthetic_shop.sales, synthetic_shop.returns
    f = build_features(sales, returns, CUTOFF)
    lab = build_labels(sales, CUTOFF, HORIZON_DAYS)
    train = dataset.build_train(sales, returns, CUTOFF, OFFSET, HORIZON_DAYS)
    test = dataset.build_test(f, lab)
    return train, test, f, lab


@pytest.fixture(scope="module")
def computed(synthetic_shop, frames):
    _, _, f, lab = frames
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variables are collinear")
        return compute(synthetic_shop.sales, synthetic_shop.returns, f, lab, knn_k_max=K_MAX)


# ------------------------------------------------------------------ datasets / leakage
def _perturb_from(df: pd.DataFrame, cutoff: pd.Timestamp, extra_rows: pd.DataFrame):
    df = df.copy()
    late = df["invoice_date"] >= cutoff
    df.loc[late, "quantity"] = df.loc[late, "quantity"] * 7
    df.loc[late, "line_value"] = df.loc[late, "quantity"] * df.loc[late, "price"]
    return pd.concat([df, extra_rows], ignore_index=True)


def test_train_snapshot_never_touches_dates_from_cutoff(synthetic_shop):
    sales, returns = synthetic_shop.sales, synthetic_shop.returns
    base = dataset.build_train(sales, returns, CUTOFF, OFFSET, HORIZON_DAYS)

    # new rows exactly at and after the cutoff: an existing customer and a brand-new one
    extra = sales.head(2).copy()
    extra["invoice_date"] = [CUTOFF, CUTOFF + pd.Timedelta(days=3)]
    extra["customer_id"] = pd.array([int(sales["customer_id"].iloc[0]), 99999], dtype="Int64")
    extra["quantity"] = 500
    extra["line_value"] = extra["quantity"] * extra["price"]
    extra_ret = returns.head(1).copy()
    extra_ret["invoice_date"] = CUTOFF
    perturbed = dataset.build_train(
        _perturb_from(sales, CUTOFF, extra),
        _perturb_from(returns, CUTOFF, extra_ret),
        CUTOFF,
        OFFSET,
        HORIZON_DAYS,
    )
    pd.testing.assert_frame_equal(base, perturbed)
    assert 99999 not in set(perturbed["customer_id"])


def test_train_snapshot_shapes(frames):
    train, test, f, lab = frames
    t_cut = dataset.train_cutoff(CUTOFF, OFFSET)
    assert t_cut == pd.Timestamp("2011-06-11")
    assert list(train.columns) == [*C.FEATURES_COLUMNS, "returned_flag", "next_spend"]
    assert len(test) == len(f) and not test.isna().any().any()
    assert set(train["returned_flag"].unique()) <= {0, 1}
    assert ((train["next_spend"] > 0) == (train["returned_flag"] == 1)).all()


def test_join_labels_fills_missing_and_keeps_rows(feature_case):
    f = build_features(feature_case.sales, feature_case.returns, feature_case.cutoff)
    lab = build_labels(feature_case.sales, feature_case.cutoff, feature_case.horizon)
    out = dataset.join_labels(f, lab.iloc[1:])  # drop the first customer's label row
    assert len(out) == len(f)
    assert out.loc[0, "returned_flag"] == 0 and out.loc[0, "next_spend"] == 0.0
    assert not out.isna().any().any()
    with pytest.raises(ValueError):
        dataset.join_labels(f, pd.concat([lab, lab.head(1)]))


def test_short_offset_rejected(synthetic_shop):
    with pytest.raises(ValueError):
        dataset.build_train(synthetic_shop.sales, synthetic_shop.returns, CUTOFF, 30, 90)


# ------------------------------------------------------------------ preprocessing / pipelines
def test_pipeline_takes_raw_frame_and_pickles(frames):
    train, test, _, _ = frames
    pipe = classify.make_classifier("logistic").fit(train[dataset.FEATURES], train["returned_flag"])
    raw = test[dataset.FEATURES]
    # the pipeline logs internally: same transform as log1p_skewed -> scaler by hand
    by_hand = pipe.named_steps["scale"].transform(log1p_skewed(raw.astype("float64")))
    np.testing.assert_allclose(pipe[:-1].transform(raw), by_hand)
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    again = joblib.load(buf)
    np.testing.assert_allclose(again.predict_proba(raw), pipe.predict_proba(raw))
    # array input in FEATURES order gives the same answer
    np.testing.assert_allclose(pipe.predict_proba(raw.to_numpy()), pipe.predict_proba(raw))


def test_max_f1_threshold_by_hand():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.7])
    # cut 0.3 -> predict 4 (3 TP): F1 = 6/7 is the best of all cut-offs
    assert classify.max_f1_threshold(y, p) == pytest.approx(0.3)


def test_fast_knn_scan_matches_sklearn(frames):
    train, test, _, _ = frames
    prep = dataset.preprocessor().fit(train[dataset.FEATURES])
    Xtr, Xte = prep.transform(train[dataset.FEATURES]), prep.transform(test[dataset.FEATURES])
    y = train["returned_flag"].to_numpy()
    votes = classify._knn_votes(Xtr, y, Xte, 15)
    for k in (1, 4, 15):
        ref = KNeighborsClassifier(n_neighbors=k).fit(Xtr, y).predict_proba(Xte)[:, 1]
        np.testing.assert_allclose(votes[:, k - 1], ref)


# ------------------------------------------------------------------ classification
def test_choices_use_train_only(frames):
    train, test, _, _ = frames
    a = classify.run_classification(train, test, knn_k_max=K_MAX)
    flipped = test.assign(returned_flag=1 - test["returned_flag"])
    b = classify.run_classification(train, flipped, knn_k_max=K_MAX)
    assert a.best_model == b.best_model
    assert a.threshold == b.threshold
    assert a.metrics["knn_k"] == b.metrics["knn_k"]
    for m in C.CLASSIFIERS:
        assert a.metrics["models"][m]["cv_aucs"] == b.metrics["models"][m]["cv_aucs"]
        assert a.metrics["models"][m]["threshold"] == b.metrics["models"][m]["threshold"]
        assert a.metrics["models"][m]["test_auc"] == pytest.approx(
            1 - b.metrics["models"][m]["test_auc"]
        )
    best = a.metrics["models"][a.best_model]["cv_auc_mean"]
    assert best == max(v["cv_auc_mean"] for v in a.metrics["models"].values())


def test_classification_outputs(computed, frames):
    train, test, _, _ = frames
    tables, metrics, _ = computed
    cls = tables[C.CLASSIFICATION_PARQUET]
    assert len(cls) == len(C.CLASSIFIERS) * (len(train) + len(test))
    assert set(cls["split"]) == {"train", "test"}
    assert cls["p_return"].between(0, 1).all()
    c = metrics["classify"]
    for m in C.CLASSIFIERS:
        mm = c["models"][m]
        assert len(mm["cv_aucs"]) == 5
        rows = cls[(cls["model"] == m) & (cls["split"] == "test")]
        assert ((rows["p_return"] >= mm["threshold"]).astype(int) == rows["pred_return"]).all()
        assert sum(mm["confusion"].values()) == len(test)
    pt = c["paired_t_auc"]
    assert pt["model_a"] == c["best_model"] and pt["model_b"] != pt["model_a"]
    assert c["reference"]["auc"] == 0.5
    assert set(tables[C.ROC_PARQUET]["model"]) == set(C.CLASSIFIERS)


def test_knn_bias_variance_curve(computed):
    curve = computed[0][C.KNN_CURVE_PARQUET]
    assert curve["k"].tolist() == list(range(1, K_MAX + 1))
    assert curve.loc[0, "train_error"] <= curve.loc[0, "test_error"]
    assert curve[["train_error", "test_error"]].stack().between(0, 1).all()


# ------------------------------------------------------------------ regression / backtest
def test_regression_on_returners(computed, frames):
    train, test, _, _ = frames
    tables, metrics, models = computed
    reg = tables[C.REGRESSION_PARQUET]
    n_tr, n_te = (train["next_spend"] > 0).sum(), (test["next_spend"] > 0).sum()
    assert len(reg) == len(C.REGRESSORS) * (n_tr + n_te)
    te = reg[(reg["split"] == "test") & (reg["model"] == "simple_lr")]
    spend = test.set_index("customer_id").loc[te["customer_id"], "next_spend"].to_numpy()
    np.testing.assert_allclose(te["y_true"].to_numpy(), np.log1p(spend))
    r = metrics["regress"]
    assert r["best_model"] == min(r["models"], key=lambda m: r["models"][m]["cv_rmse_mean"])
    assert r["models"]["simple_lr"]["features"] == ["frequency"]
    assert models[C.REGRESSOR_JOBLIB]["target"] == "log1p(next_spend)"


def test_backtest_by_hand():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.3, 0.2, 0.1, 0.05])
    b = backtest(y, p, 0.5, "logistic")
    assert b["n_predicted_return"] == 4  # 0.9, 0.8, 0.7, 0.6
    assert b["share_actually_returned"] == pytest.approx(0.75)
    assert b["base_rate"] == pytest.approx(0.4)
    assert b["lift"] == pytest.approx(0.75 / 0.4)
    assert b["recall"] == pytest.approx(0.75)
    d = decile_table(y, p, 5)
    assert d["n"].tolist() == [2] * 5 and d.loc[0, "return_rate"] == 1.0


# ------------------------------------------------------------------ contracts
def test_contracts_and_metrics_keys(computed):
    tables, metrics, models = computed
    for path, df in tables.items():
        C.validate(df, path)
    for section in ("classify", "regress", "backtest"):
        assert set(C.P3_METRICS_KEYS[section]) <= set(metrics[section])
        assert isinstance(metrics[section]["conclusion"], str)
    assert set(models[C.CLASSIFIER_JOBLIB]) == {"pipeline", "features", "model", "threshold"}
    assert set(models[C.REGRESSOR_JOBLIB]) == {"pipeline", "features", "model", "target", "note"}
    assert models[C.CLASSIFIER_JOBLIB]["features"] == C.FEATURE_COLUMNS
    assert metrics["backtest"]["model"] == metrics["classify"]["best_model"]
    C.to_jsonable(metrics)  # JSON-serialisable


def test_regressor_unknown_name():
    with pytest.raises(ValueError):
        regress.make_regressor("forest")
    with pytest.raises(ValueError):
        classify.make_classifier("forest")
