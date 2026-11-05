import pandas as pd
import pytest

from retailpulse.clean import (
    UNKNOWN_DESCRIPTION,
    clean,
    description_lookup,
    missing_id_analysis,
)
from retailpulse.contracts import RAW_COLUMNS, SALES_COLUMNS, ContractError


@pytest.fixture
def result(raw):
    return clean(raw)


def test_standardize_schema(raw):
    assert list(raw.columns) == RAW_COLUMNS
    assert raw["customer_id"].dtype == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(raw["invoice_date"])
    assert raw["invoice"].map(type).eq(str).all()


def test_step_counts(result, raw):
    removed = {s["step"]: s["removed"] for s in result.steps}
    assert len(raw) == 18
    assert removed == {
        "remove_duplicates": 1,
        "split_returns": 2,
        "drop_nonpositive_qty_price": 3,
        "drop_missing_customer_id": 2,
        "impute_description": 0,
    }
    assert len(result.sales) == 10
    assert len(result.returns) == 2


def test_funnel_is_consistent(result, raw):
    steps = result.steps
    assert steps[0]["rows_in"] == len(raw)
    for prev, nxt in zip(steps, steps[1:], strict=False):
        assert prev["rows_out"] == nxt["rows_in"]
    assert len(raw) - sum(s["removed"] for s in steps) == len(result.sales)


def test_returns_split(result):
    assert result.returns["invoice"].str.startswith("C").all()
    assert not result.sales["invoice"].str.startswith("C").any()
    assert (result.returns["quantity"] < 0).all()


def test_sales_valid(result):
    s = result.sales
    assert (s["quantity"] > 0).all() and (s["price"] > 0).all()
    assert s["customer_id"].notna().all()
    assert not s.duplicated().any()
    assert not s["invoice"].str.startswith("A").any()


def test_description_imputed_with_mode(result):
    s = result.sales
    assert s["description"].notna().all()
    inv = s[s["invoice"] == "100013"].set_index("stock_code")["description"]
    assert inv["85123A"] == "WHITE HEART"  # mode over HEART
    assert inv["99999"] == UNKNOWN_DESCRIPTION
    assert result.returns["description"].notna().all()
    assert result.returns.set_index("invoice").loc["C100008", "description"] == "CAKESTAND"
    imp = result.steps[-1]
    assert imp["imputed_sales"] == 2 and imp["imputed_returns"] == 1


def test_junk_description_not_in_lookup(result):
    # "?" lives on a zero-price row, which is dropped before the lookup is built
    assert "?" not in set(result.sales["description"])


def test_line_value(result):
    for df in (result.sales, result.returns):
        assert (df["line_value"] == df["quantity"] * df["price"]).all()
    assert list(result.sales.columns) == SALES_COLUMNS


def test_input_not_mutated(raw):
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_missing_columns_raise(raw):
    with pytest.raises(ContractError):
        clean(raw.drop(columns="price"))


def test_description_lookup_tie_break():
    df = pd.DataFrame(
        {"stock_code": ["X", "X", "Y"], "description": ["B", "A", pd.NA]}, dtype="string"
    )
    lk = description_lookup(df)
    assert lk["X"] == "A"
    assert "Y" not in lk.index


def test_missing_id_analysis(result):
    m = result.missing_id
    assert m["n_missing"] == 2 and m["n_rows"] == 12
    assert m["share_missing"] == pytest.approx(2 / 12, abs=1e-4)
    assert m["share_of_those_fully_missing"] == 1.0
    for key in ("mannwhitney_quantity", "mannwhitney_price", "chi2_uk"):
        assert 0.0 <= m["tests"][key]["p_value"] <= 1.0
    assert -1 <= m["tests"]["mannwhitney_quantity"]["rank_biserial"] <= 1
    assert m["tests"]["chi2_uk"]["share_uk_missing"] == 1.0
    assert isinstance(m["verdict"], str) and m["verdict"]


def test_missing_id_analysis_without_missing(raw):
    df = raw[raw["customer_id"].notna()]
    m = missing_id_analysis(df)
    assert m["n_missing"] == 0
    assert m["tests"] == {}
