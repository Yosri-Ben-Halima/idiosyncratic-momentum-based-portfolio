import pytest

from halal_screener import refresh
from utils.halal_helper import filter_halal

GOOD = dict(sector="Technology", industry="Software - Infrastructure", total_debt=1.0,
            cash_and_st=1.0, interest_income=0.0, total_revenue=100.0,
            market_cap=100.0, avg_market_cap_12m=100.0, total_assets=100.0, receivables=1.0)


def fake_fetch(t):
    if t == "LEV":  # fails AAOIFI (debt 40% of market cap), passes Wahed (20% of assets)
        return dict(GOOD, total_debt=40.0, total_assets=200.0)
    if t == "BANK":
        return dict(GOOD, sector="Financial Services", industry="Banks - Regional")
    return dict(GOOD)


@pytest.fixture
def data_dir(tmp_path):
    refresh(tmp_path, tickers=["AAA", "BANK", "LEV", "ZZZ"], fetch=fake_fetch, pause=0)
    return tmp_path


def test_keeps_only_compliant_in_original_order(data_dir):
    assert filter_halal(["ZZZ", "BANK", "AAA", "LEV"], "aaoifi", data_dir) == ["ZZZ", "AAA"]
    assert filter_halal(["ZZZ", "BANK", "AAA", "LEV"], "wahed", data_dir) == ["ZZZ", "AAA", "LEV"]


def test_ticker_not_in_snapshot_is_dropped(data_dir):
    assert filter_halal(["AAA", "NEW"], "aaoifi", data_dir) == ["AAA"]


def test_bad_standard_raises(data_dir):
    with pytest.raises(ValueError):
        filter_halal(["AAA"], "msci", data_dir)


def test_no_snapshot_raises_with_refresh_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="refresh"):
        filter_halal(["AAA"], "aaoifi", tmp_path)
