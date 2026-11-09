"""Extraction tests: tasks, helpers and exact area closure."""

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from sitios_arq import buffers as B
from sitios_arq import extraction as E

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_tasks_from_config():
    cfg = E.load_config(ROOT / "config/extraction.toml")
    tasks = E.build_tasks(cfg)
    assert len(tasks) == 53
    assert tasks[0] == (100, 1985) and tasks[40] == (100, 2025)
    assert tasks[41:45] == [(200, 1985), (200, 2004), (200, 2019), (200, 2025)]
    assert tasks[-1] == (1000, 2025)


def test_format_array():
    assert E.format_array([7, 8, 9, 19]) == "7-9,19"
    assert E.format_array([3]) == "3"
    assert E.format_array([]) == ""


def test_explode_classes_with_empty_feature():
    o1 = pd.DataFrame({"_fid": [0, 1], "unique": [np.array([3, 15]), np.array([])],
                       "weighted_frac": [np.array([0.25, 0.75]), np.array([])]})
    out = E.explode_classes(o1)
    assert out["_fid"].tolist() == [0, 0]
    assert out["mapbiomas_class"].tolist() == [3, 15]


def _write_tif(path, data, transform, dtype):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                       count=1, dtype=dtype, crs="EPSG:4326", transform=transform) as dst:
        dst.write(data.astype(dtype), 1)


def test_extract_closes_on_geodesic_area(tmp_path):
    pa = _load_script("05_pixel_area")
    t = from_origin(-50.05, -9.95, 0.001, 0.001)
    n = 100
    cls = np.where(np.arange(n)[None, :] < 50, 3, 15) * np.ones((n, 1), dtype=int)
    cls[30:40, :] = 33
    _write_tif(tmp_path / "cls.tif", cls, t, "uint8")
    area = np.repeat(pa.row_areas(t, n)[:, None], n, axis=1)
    _write_tif(tmp_path / "area.tif", area, t, "float32")

    geoms = B.geodesic_buffers([-50.0, -49.97], [-10.0, -9.98], 1000, 256)
    gdf = E.prepare(gpd.GeoDataFrame({"id_bem": [1, 2]}, geometry=geoms, crs=4674))
    classes, totals = E.extract(tmp_path / "cls.tif", tmp_path / "area.tif", gdf)

    geod = B.geodesic_areas(gdf.geometry)
    summed = classes.groupby("_fid")["area_m2"].sum().reindex(gdf["_fid"]).to_numpy()
    assert np.abs(summed / geod - 1).max() < 1e-6
    assert np.abs(totals.set_index("_fid").loc[gdf["_fid"], "extracted_area_m2"].to_numpy()
                  / geod - 1).max() < 1e-6
    assert set(classes["mapbiomas_class"]) == {3, 15, 33}


def test_prepare_rejects_projected_crs():
    g = gpd.GeoDataFrame({"id_bem": [1]}, geometry=B.geodesic_buffers([-50], [-10], 100, 16),
                         crs=4674).to_crs(5880)
    with pytest.raises(ValueError):
        E.prepare(g)


def test_national_vs_biomes_and_class0():
    agg = pd.DataFrame({"unit": ["NATIONAL", "NATIONAL", "A", "B"], "radius_m": 100, "year": 1985,
                        "mapbiomas_class": [0, 3, 3, 0], "area_m2": [1.0, 3.0, 3.0, 1.0]})
    assert E.national_vs_biomes(agg)["rel_diff"].iloc[0] == 0
    sites = pd.DataFrame({"id_bem": [1, 1, 2], "radius_m": 100, "year": 1985,
                          "mapbiomas_class": [0, 3, 3], "area_m2": [1.0, 3.0, 4.0]})
    c0 = E.class0_diagnostic(sites)
    assert c0["id_bem"].tolist() == [1] and c0["class0_frac"].iloc[0] == 0.25


def test_class0_review_table_sorts_and_aggregates():
    diagnostic = pd.DataFrame({
        "id_bem": [1, 1, 2, 3],
        "radius_m": [100, 100, 100, 200],
        "year": [1985, 1986, 1985, 1985],
        "class0_frac": [0.2, 0.4, 0.9, 0.5],
    })
    sites = pd.DataFrame({"id_bem": [1, 2, 3], "lon": [-40.0, -41.0, -42.0], "lat": [-10.0, -11.0, -12.0]})

    review = E.class0_review_table(diagnostic, sites, radius_m=100)

    assert review["id_bem"].tolist() == [2, 1]  # sorted by frac_max desc; site 3 excluded (radius 200)
    row1 = review[review["id_bem"] == 1].iloc[0]
    assert row1["n_years_present"] == 2 and row1["frac_mean"] == pytest.approx(0.3)
    assert row1["frac_max"] == pytest.approx(0.4) and row1["frac_min"] == pytest.approx(0.2)
    assert review[review["id_bem"] == 2].iloc[0]["lon"] == pytest.approx(-41.0)


def test_merge_report_fails_on_partial_coverage():
    # a closure deficit "explained" by partial coverage can hide a broken geometry (P-026)
    merge = _load_script("07_merge_extraction")
    df = pd.DataFrame({"radius_m": [100, 100], "year": [1985, 1985],
                       "rel_geodesic": [0.0, -0.078], "coverage_frac": [1.0, 0.922],
                       "orphan": [False, False], "rel_ngon": [np.nan, np.nan]})
    assert merge.report(df, "aggregate", [], check_ngon=False) == 1
    assert merge.report(df, "aggregate", [], check_ngon=False, allow_partial=True) == 0
