from halal_screener.rules import evaluate


def ok(**kw):
    row = dict(sector="Technology", industry="Software - Infrastructure",
               total_debt=10.0, cash_and_st=10.0, interest_income=1.0,
               total_revenue=100.0, market_cap=100.0, avg_market_cap_12m=100.0,
               total_assets=100.0, receivables=10.0)
    row.update(kw)
    return row


def test_clean_stock_passes():
    r = evaluate(ok())
    assert r["passed"] and r["reasons"] == ""
    assert r["purification_pct"] == 0.01


def test_ratio_at_limit_fails():
    assert "debt" in evaluate(ok(total_debt=30.0))["reasons"].split(";")
    assert "income" in evaluate(ok(interest_income=5.0))["reasons"].split(";")


def test_avg_market_cap_must_also_pass():
    r = evaluate(ok(total_debt=25.0, avg_market_cap_12m=80.0))  # 25/80 = 31%
    assert not r["passed"] and r["reasons"] == "debt_avg"


def test_cash_fails():
    assert evaluate(ok(cash_and_st=40.0))["reasons"] == "cash;cash_avg"


def test_prohibited_and_grey_industries_fail():
    assert evaluate(ok(industry="Banks - Regional"))["reasons"] == "prohibited:Banks - Regional"
    assert evaluate(ok(industry="Restaurants"))["reasons"] == "grey:Restaurants"


def test_missing_field_fails():
    r = evaluate(ok(interest_income=float("nan")))
    assert not r["passed"] and "missing:interest_income" in r["reasons"]
    assert r["purification_pct"] is None
    assert "missing:industry" in evaluate(ok(industry=None))["reasons"]


def test_zero_denominator_fails():
    assert "missing:market_cap" in evaluate(ok(market_cap=0.0))["reasons"]
    assert "missing:total_revenue" in evaluate(ok(total_revenue=-5.0))["reasons"]


def test_financial_sector_fails_whatever_the_industry():
    r = evaluate(ok(sector="Financial Services", industry="Asset Management"))
    assert not r["passed"] and "prohibited:Financial Services" in r["reasons"]


def test_dash_variants_of_labels_still_match():
    assert "prohibited:Banks - Regional" in evaluate(ok(industry="Banks—Regional"))["reasons"]
    assert "grey:Aerospace & Defense" in evaluate(ok(industry="aerospace & defense"))["reasons"]


def test_missing_sector_fails():
    assert "missing:sector" in evaluate(ok(sector=None))["reasons"]


import pytest


def test_wahed_clean_passes():
    r = evaluate(ok(), standard="wahed")
    assert r["passed"] and r["reasons"] == "" and r["purification_pct"] == 0.01


def test_wahed_uses_total_assets_not_market_cap():
    row = ok(total_debt=40.0, total_assets=200.0)  # 40% of market cap, 20% of assets
    assert not evaluate(row)["passed"]
    assert evaluate(row, standard="wahed")["passed"]


def test_wahed_limits():
    assert evaluate(ok(total_debt=33.334), standard="wahed")["reasons"] == "debt"
    assert evaluate(ok(cash_and_st=33.334), standard="wahed")["reasons"] == "cash"
    assert evaluate(ok(receivables=40.0), standard="wahed")["reasons"] == "receivables_cash"  # 40+10 = 50%
    assert evaluate(ok(interest_income=5.0), standard="wahed")["passed"]  # "should not exceed" 5%
    assert evaluate(ok(interest_income=5.01), standard="wahed")["reasons"] == "income"


def test_each_standard_requires_only_its_fields():
    no_rec = ok(receivables=None)
    assert evaluate(no_rec)["passed"]
    assert evaluate(no_rec, standard="wahed")["reasons"] == "missing:receivables"
    no_avg = ok(avg_market_cap_12m=None)
    assert evaluate(no_avg, standard="wahed")["passed"]
    assert "missing:avg_market_cap_12m" in evaluate(no_avg)["reasons"]


def test_wahed_shares_business_screen():
    assert "prohibited:Banks - Regional" in evaluate(ok(industry="Banks - Regional"), standard="wahed")["reasons"]


def test_unknown_standard_raises():
    with pytest.raises(ValueError):
        evaluate(ok(), standard="msci")
