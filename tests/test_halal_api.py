import pytest

from halal_screener.api import refresh, screen, halal_tickers, compare

GOOD = dict(sector="Technology", industry="Software - Infrastructure", total_debt=1.0,
            cash_and_st=1.0, interest_income=0.0, total_revenue=100.0,
            market_cap=100.0, avg_market_cap_12m=100.0, total_assets=100.0, receivables=1.0)


def fake_fetch(t):
    if t == "LEV":  # debt 40% of market cap but 20% of assets: fails AAOIFI, passes Wahed
        return dict(GOOD, total_debt=40.0, total_assets=200.0)
    if t == "REC":  # receivables + cash = 60% of assets: passes AAOIFI, fails Wahed
        return dict(GOOD, receivables=59.0)
    if t == "BOOM":
        raise TimeoutError("down")
    return dict(GOOD, industry="Banks - Regional") if t == "BANK" else dict(GOOD)


def snap(tmp_path):
    return refresh(tmp_path, tickers=["AAA", "BANK", "BOOM"], fetch=fake_fetch, pause=0, max_error_share=0.5)


def test_refresh_and_halal_tickers(tmp_path):
    path = snap(tmp_path)
    assert path.name.startswith("snapshot_") and path.exists()
    assert halal_tickers(data_dir=tmp_path) == ["AAA"]


def test_refresh_survives_fetch_error(tmp_path):
    snap(tmp_path)
    row = screen(data_dir=tmp_path).set_index("ticker").loc["BOOM"]
    for std in ("aaoifi", "wahed"):
        assert not row[f"passed_{std}"] and row[f"reasons_{std}"] == "error:TimeoutError"


def test_allowlist_adds_reviewed_ticker(tmp_path):
    snap(tmp_path)
    (tmp_path / "allowlist.txt").write_text("# reviewed\nbank  # ok per scholar\n")
    assert halal_tickers(data_dir=tmp_path) == ["AAA", "BANK"]


def test_allowlist_ignores_tickers_outside_snapshot(tmp_path):
    snap(tmp_path)
    (tmp_path / "allowlist.txt").write_text("ZZZ\n")
    assert halal_tickers(data_dir=tmp_path) == ["AAA"]


def test_no_snapshot_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="refresh"):
        halal_tickers(data_dir=tmp_path)


def test_limited_run_is_partial_and_not_served(tmp_path):
    snap(tmp_path)
    p = refresh(tmp_path, limit=1, tickers=["BANK"], fetch=fake_fetch, pause=0)
    assert p.name.endswith("_partial.csv")
    assert halal_tickers(data_dir=tmp_path) == ["AAA"]


def test_error_heavy_run_is_partial(tmp_path):
    p = refresh(tmp_path, tickers=["AAA", "BOOM"], fetch=fake_fetch, pause=0)
    assert p.name.endswith("_partial.csv")


def test_snapshot_has_both_standards(tmp_path):
    refresh(tmp_path, tickers=["AAA", "LEV", "REC"], fetch=fake_fetch, pause=0)
    df = screen(data_dir=tmp_path)
    assert {"passed_aaoifi", "reasons_aaoifi", "passed_wahed", "reasons_wahed", "total_assets"} <= set(df.columns)
    assert halal_tickers(data_dir=tmp_path) == ["AAA", "REC"]
    assert halal_tickers(data_dir=tmp_path, standard="wahed") == ["AAA", "LEV"]


def test_compare_splits_lists(tmp_path):
    refresh(tmp_path, tickers=["AAA", "LEV", "REC", "BANK"], fetch=fake_fetch, pause=0)
    assert compare(data_dir=tmp_path) == {"both": ["AAA"], "aaoifi_only": ["REC"], "wahed_only": ["LEV"]}


def test_unknown_standard_rejected(tmp_path):
    snap(tmp_path)
    with pytest.raises(ValueError):
        halal_tickers(data_dir=tmp_path, standard="msci")
