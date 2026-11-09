"""Buffer tests on synthetic geometries.

Scenario (lon/lat, SIRGAS 2000): biome A = box lon -52..-50, biome B = box
lon -50..-48, both lat -12..-8; east of lon -48 lies outside every biome.
"""

import math

import geopandas as gpd
import numpy as np
import pytest
import shapely
from pyproj import Geod
from shapely.geometry import Point, box

from sitios_arq import buffers as B

GEOD = Geod(ellps="WGS84")


@pytest.fixture
def biomes_ll():
    return gpd.GeoDataFrame(
        {"Bioma": ["A", "B"], "CD_Bioma": [1, 2]},
        # densified: a straight lon/lat edge is curved in EPSG:5880
        geometry=shapely.segmentize([box(-52, -12, -50, -8), box(-50, -12, -48, -8)], 0.001),
        crs=4674,
    )


@pytest.mark.parametrize("lat", [-33.0, -15.0, 5.0])
@pytest.mark.parametrize("r", [100, 1000])
def test_geodesic_buffer_vertices_and_area(lat, r):
    n = 256
    poly = B.geodesic_buffers([-50.0], [lat], r, n)[0]
    xy = shapely.get_coordinates(poly)[:-1]
    assert len(xy) == n
    _, _, dist = GEOD.inv(np.full(n, -50.0), np.full(n, lat), xy[:, 0], xy[:, 1])
    assert np.abs(dist - r).max() < 1e-3
    area = B.geodesic_areas([poly])[0]
    assert abs(area / (math.pi * r ** 2 * B.polygon_area_factor(n)) - 1) < 1e-6


def test_polygon_area_factor():
    assert B.polygon_area_factor(64) == pytest.approx(0.998394, abs=1e-6)
    assert B.polygon_area_factor(256) == pytest.approx(0.999900, abs=1e-6)


def test_fill_nearest_biome(biomes_ll):
    sites = gpd.GeoDataFrame(
        {"id_bem": [1, 2], "biome": ["A", None]},
        geometry=[Point(-51, -10), Point(-47.99, -10)], crs=4674,
    ).to_crs(5880)
    out = B.fill_nearest_biome(sites, biomes_ll.to_crs(5880))
    assert out["biome"].tolist() == ["A", "B"]
    assert out["biome_dist_m"].iloc[0] == 0
    assert 1000 < out["biome_dist_m"].iloc[1] < 1200


def test_dissolve_overlap():
    geoms = B.geodesic_buffers([-50.0, -49.999], [-10.0, -10.0], 100, 256)
    ind = B.geodesic_areas(geoms).sum()
    diss = B.geodesic_areas([shapely.union_all(geoms)])[0]
    assert diss < ind
    assert diss > B.geodesic_areas(geoms[:1])[0]


def test_disjoint_biomes_removes_overlap():
    b = gpd.GeoDataFrame(
        {"Bioma": ["A", "B"], "CD_Bioma": [1, 2]},
        geometry=[box(0, 0, 2, 1), box(1, 0, 3, 1)], crs=4674,
    )
    d = B.disjoint_biomes(b)
    assert d.geometry.iloc[0].intersection(d.geometry.iloc[1]).area == 0
    assert d.geometry.iloc[1].area == pytest.approx(1.0)


def test_clip_by_biome_partition(biomes_ll):
    # crossing A|B at lon -50; half outside B at lon -47.995; fully outside at -47.9
    geoms = np.concatenate([
        B.geodesic_buffers([-50.0], [-10.0], 1000, 256),
        B.geodesic_buffers([-47.995, -47.9], [-10.0, -10.0], 1000, 256),
    ])
    dissolved = shapely.union_all(geoms)
    disjoint = B.disjoint_biomes(biomes_ll)
    out, outside = B.clip_by_biome(dissolved, disjoint, biomes_ll.to_crs(5880))

    areas = dict(zip(out["biome"], B.geodesic_areas(out.geometry)))
    total = B.geodesic_areas([dissolved])[0]
    assert sum(areas.values()) == pytest.approx(total, rel=1e-9)
    one = B.geodesic_areas(geoms[:1])[0]
    assert areas["A"] == pytest.approx(one / 2, rel=1e-3)
    # B holds half of the crossing buffer plus both eastern buffers entirely
    assert areas["B"] == pytest.approx(total - one / 2, rel=1e-3)
    assert outside["A"] == 0
    assert outside["B"] > B.geodesic_areas(geoms[2:3])[0]
