"""Change metric tests (synthetic units)."""

import pandas as pd
import pytest

from sitios_arq import change as C


@pytest.fixture
def df():
    # unit A: valid area constant; unit B: valid area changes (mean in the denominator)
    return pd.DataFrame({
        "unit": ["A", "A", "A", "B", "B", "B"],
        "year": [2000, 2010, 2020] * 2,
        "area_anthropic_m2": [10.0, 30.0, 25.0, 0.0, 40.0, 60.0],
        "area_with_data_m2": [100.0, 100.0, 100.0, 100.0, 100.0, 300.0],
    })


def test_delta_and_rate(df):
    out = C.period_change(df, [(2000, 2020), (2000, 2010)]).set_index(["unit", "period"])
    a = out.loc[("A", "2000–2020")]
    assert a["delta_pp"] == pytest.approx(15.0)
    assert a["rate_pp_per_yr"] == pytest.approx(0.75)
    assert a["delta_pp"] == pytest.approx(a["anthropic_pct_end"] - a["anthropic_pct_start"])
    b = out.loc[("B", "2000–2020")]
    assert b["delta_pp"] == pytest.approx(100 * 60 / 200)  # mean valid area
    assert out.loc[("A", "2000–2010"), "rate_m2_per_yr"] == pytest.approx(2.0)


def test_additivity(df):
    periods = [(2000, 2020), (2000, 2010), (2010, 2020)]
    out = C.period_change(df, periods)
    C.check_additivity(out, (2000, 2020), [(2000, 2010), (2010, 2020)])
    with pytest.raises(ValueError, match="do not tile"):
        C.check_additivity(out, (2000, 2020), [(2000, 2010)])


def test_errors(df):
    with pytest.raises(ValueError, match="Year 1999"):
        C.period_change(df, [(1999, 2020)])
    with pytest.raises(ValueError, match="different units"):
        C.period_change(df.drop(index=5), [(2000, 2020)])
    with pytest.raises(ValueError, match="Duplicated"):
        C.period_change(pd.concat([df, df.iloc[[0]]]), [(2000, 2020)])


def test_periods_additivity(df):
    periods = [(2000, 2020), (2000, 2010), (2010, 2020)]
    C.check_periods_additivity(C.period_change(df, periods), periods)
    assert C.full_span(periods) == (2000, 2020)
    with pytest.raises(ValueError, match="full span"):
        C.full_span([(2000, 2010), (2010, 2020)])
