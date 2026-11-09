"""Thematic aggregation tests: legend mapping, class 0 rule, closure and checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sitios_arq import thematic as T

ROOT = Path(__file__).resolve().parents[1]
LEGEND_PATH = ROOT / "config/mapbiomas_legend_c11.csv"

APPROVED = {
    "Natural": {3, 4, 5, 6, 7, 11, 12, 23, 29, 32, 49, 50, 77, 84},
    "Anthropic": {9, 15, 20, 21, 24, 25, 30, 31, 35, 39, 40, 41, 46, 47, 48, 62, 75, 91},
    "Water": {33},
    "No data": {0},
}


@pytest.fixture(scope="module")
def legend():
    return T.load_legend(LEGEND_PATH)


def test_legend_matches_approved_mapping(legend):
    assert len(legend) == 34
    got = legend.groupby("category")["code"].apply(set).to_dict()
    assert got == APPROVED


def test_each_code_in_exactly_one_category(legend):
    for code in set().union(*APPROVED.values()):
        hits = [cat for cat, codes in APPROVED.items() if code in codes]
        assert len(hits) == 1
        assert T.category_of(legend)[code] == T.CATEGORIES[hits[0]]


def _long(rows):
    return pd.DataFrame(rows, columns=["id_bem", "radius_m", "year", "mapbiomas_class", "area_m2"])


def test_class0_left_out_of_denominator(legend):
    df = _long([(1, 200, 1985, 3, 30.0), (1, 200, 1985, 15, 50.0), (1, 200, 1985, 33, 20.0),
                (1, 200, 1985, 0, 100.0)])
    out = T.aggregate_categories(df, legend, ["id_bem", "radius_m", "year"]).iloc[0]
    assert out["area_total_m2"] == 200.0
    assert out["area_with_data_m2"] == 100.0
    assert out["area_nodata_m2"] == 100.0
    assert (out["prop_natural"], out["prop_anthropic"], out["prop_water"]) == (0.3, 0.5, 0.2)
    assert T.closure_rel(pd.DataFrame([out])).abs().max() == 0


def test_all_nodata_gives_nan_proportions(legend):
    df = _long([(1, 1000, 1985, 0, 10.0)])
    out = T.aggregate_categories(df, legend, ["id_bem", "radius_m", "year"]).iloc[0]
    assert np.isnan(out["prop_natural"]) and out["area_natural_m2"] == 0.0


def test_unmapped_code_raises(legend):
    df = _long([(1, 100, 1985, 3, 1.0), (1, 100, 1985, 26, 1.0)])
    with pytest.raises(KeyError, match="26"):
        T.aggregate_categories(df, legend, ["id_bem", "radius_m", "year"])


def test_national_vs_biomes_per_category(legend):
    df = pd.DataFrame({"unit": ["NATIONAL", "NATIONAL", "Cerrado", "Pampa", "Pampa"],
                       "radius_m": 100, "year": 1985, "mapbiomas_class": [3, 15, 3, 3, 15],
                       "area_m2": [5.0, 2.0, 4.0, 1.0, 2.0]})
    out = T.aggregate_categories(df, legend, ["unit", "radius_m", "year"])
    nb = T.national_vs_biomes(out)
    assert nb["rel_diff"].max() == 0
    out.loc[out["unit"] == "Pampa", "area_anthropic_m2"] = 3.0
    nb = T.national_vs_biomes(out)
    assert nb.set_index("category").loc["area_anthropic_m2", "rel_diff"] == pytest.approx(0.5)


def test_class0_site_summary(legend):
    df = _long([(1, 500, 1985, 3, 9.0), (1, 500, 1985, 0, 1.0),
                (2, 500, 1985, 3, 10.0),
                (1, 500, 2025, 3, 10.0), (2, 500, 2025, 3, 10.0)])
    out = T.aggregate_categories(df, legend, ["id_bem", "radius_m", "year"])
    s = T.class0_site_summary(out).set_index("year")
    assert s.loc[1985, "n_sites_nodata"] == 1
    assert s.loc[1985, "nodata_frac_max"] == pytest.approx(0.1)
    assert s.loc[2025, "n_sites_nodata"] == 0
    assert (s["n_sites_any_year"] == 1).all()
