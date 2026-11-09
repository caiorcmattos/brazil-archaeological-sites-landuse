"""Reservoir association and water status (sitios_arq.reservoirs), synthetic geometries."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from sitios_arq import reservoirs as R

ROOT = Path(__file__).resolve().parents[1]
CRS = "EPSG:4674"


def cfg():
    return R.load_config(ROOT / "config/reservoirs.toml")


def test_config():
    c = cfg()
    assert c["years"]["value"] == [1985, 2025]
    assert c["water_body_type"]["value"] == "Artificial"


def test_water_status():
    start = np.array([1.0, 0.2, 0.2, 0.5, 0.5, 1.0])
    end = np.array([1.0, 1.0, 0.6, 0.5, 0.1, 0.999999])
    assert list(R.water_status(start, end)) == [
        "entirely_water_both", "entirely_water_end", "partial_gain", "no_gain", "no_gain", "no_gain"]
    with pytest.raises(ValueError):
        R.water_status([np.nan], [1.0])


def test_water_status_min_share():
    start = np.array([0.98, 0.2, 0.96, 0.5])
    end = np.array([0.99, 0.97, 0.98, 0.94])
    assert list(R.water_status(start, end, 0.97)) == [
        "entirely_water_both", "entirely_water_end", "entirely_water_end", "partial_gain"]
    assert list(R.water_status(start, end)) == ["partial_gain"] * 4
    with pytest.raises(ValueError):
        R.water_status(start, end, 0)


def bodies_raw():
    return gpd.GeoDataFrame({
        "gid": [1, 2, 3],
        "detipomass": ["Artificial", "Artificial", "Natural"],
        "nmoriginal": ["UHE A", None, "Lagoa"],
        "nmespecifi": [None, "  Açude B ", None],
        "usoprinc": ["Hidrelétrica", None, None],
    }, geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6), box(10, 10, 11, 11)], crs=CRS)


def test_prepare_and_link():
    bodies = R.prepare_water_bodies(bodies_raw(), cfg(), CRS)
    assert list(bodies["res_id"]) == [1, 2]                  # natural dropped
    assert list(bodies["res_name"]) == ["UHE A", "Açude B"]   # fallback to nmespecifi, stripped
    assert list(bodies["is_hydro"]) == [True, False]

    buffers = gpd.GeoDataFrame({"id_bem": [10, 11, 12, 13], "biome": ["Cerrado"] * 4},
                               geometry=[Point(0.5, 0.5).buffer(0.1), Point(1.05, 0.5).buffer(0.1),
                                         Point(3, 3).buffer(0.1), Point(10.5, 10.5).buffer(0.1)],
                               crs=CRS)
    links = R.link_sites(buffers, bodies)
    assert list(links["id_bem"]) == [10, 11]                 # 12 far away, 13 only touches natural
    assert links["is_hydro"].all()

    water = pd.DataFrame({"id_bem": [10, 10, 11, 11], "year": [1985, 2025] * 2,
                          "prop_water": [0.0, 1.0, 0.1, 0.4]})
    sites = R.site_table(links, water, [1985, 2025])
    assert list(sites["status"]) == ["entirely_water_end", "partial_gain"]
    assert sites["delta_pp"].tolist() == pytest.approx([100, 30])

    with pytest.raises(ValueError, match="without water share"):
        R.site_table(links, water[water["id_bem"] == 10], [1985, 2025])
