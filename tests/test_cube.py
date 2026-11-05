import pytest

from retailpulse import contracts as C
from retailpulse.cube import build_cube, level_name
from retailpulse.warehouse import connect


def test_level_names():
    assert level_name(0) == "country × year × month"
    assert level_name(3) == "country"
    assert level_name(4) == "year × month"
    assert level_name(7) == "total"


@pytest.fixture(scope="module")
def cube(synthetic_shop, tmp_path_factory):
    d = tmp_path_factory.mktemp("cube")
    synthetic_shop.sales.to_parquet(d / "sales.parquet", index=False)
    synthetic_shop.returns.to_parquet(d / "returns.parquet", index=False)
    con = connect(d / "sales.parquet", d / "returns.parquet")
    try:
        yield build_cube(con)
    finally:
        con.close()


def test_cube_has_all_grouping_sets(cube, synthetic_shop):
    assert list(cube.columns) == C.CUBE_COLUMNS
    assert sorted(cube["grouping_id"].unique()) == list(range(8))
    sales = synthetic_shop.sales
    finest = cube[cube["grouping_id"] == 0]
    combos = sales.groupby(
        ["country", sales["invoice_date"].dt.year, sales["invoice_date"].dt.month]
    ).ngroups
    assert len(finest) == combos
    # a rolled-up dimension is NULL ("all"), a grouped one never is
    for dim, bit in zip(C.CUBE_DIMENSIONS, (4, 2, 1), strict=True):
        rolled = (cube["grouping_id"] & bit) > 0
        assert cube.loc[rolled, dim].isna().all() and cube.loc[~rolled, dim].notna().all()


def test_cube_totals_match_sales(cube, synthetic_shop):
    sales = synthetic_shop.sales
    total = cube[cube["level"] == "total"].iloc[0]
    assert total["revenue"] == pytest.approx(sales["line_value"].sum(), abs=0.01)
    assert total["orders"] == sales["invoice"].nunique()
    assert total["customers"] == sales["customer_id"].nunique()

    by_country = cube[cube["level"] == "country"].set_index("country")
    expected = sales.groupby("country")["line_value"].sum()
    assert by_country["revenue"].sort_index().tolist() == pytest.approx(
        expected.sort_index().round(2).tolist(), abs=0.01
    )
    # distinct customers are not additive across months -> the cube must precompute roll-ups
    by_month = cube[cube["level"] == "year × month"]
    assert by_month["customers"].sum() > total["customers"]
