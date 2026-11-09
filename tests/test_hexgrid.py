"""Hexagonal grid tests (synthetic sites and buffers, real EPSG:5880 lattice)."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from pyproj import Transformer

from sitios_arq import buffers as B
from sitios_arq import hexgrid as H

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 100_000.0
ORIGIN = (2_700_000.0, 6_200_000.0)
CRS = "EPSG:5880"


@pytest.fixture
def lattice():
    return H.HexLattice(WIDTH, ORIGIN)


def test_config_loads_and_periods():
    cfg = H.load_config(ROOT / "config/hex.toml")
    assert H.periods(cfg) == [(1985, 2025), (1985, 2004), (2004, 2019), (2019, 2025)]
    assert H.period_years(cfg) == [1985, 2004, 2019, 2025]


def test_config_rejects_bad_period(tmp_path):
    text = (ROOT / "config/hex.toml").read_text().replace("[2019, 2025]]",
                                                          "[2025, 2019]]")
    p = tmp_path / "hex.toml"
    p.write_text(text)
    with pytest.raises(ValueError, match="invalid period"):
        H.load_config(p)


def test_cell_shape_and_area(lattice):
    poly = lattice.polygon(10, 20)
    assert math.isclose(poly.area, 3 * math.sqrt(3) / 8 * WIDTH ** 2, rel_tol=1e-12)
    cx, cy = lattice.center(10, 20)
    xy = shapely.get_coordinates(poly)[:-1]
    assert np.allclose(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy), WIDTH / 2)


def test_grid_tiles_without_gaps_or_overlaps(lattice):
    grid = H.build_grid(lattice, (4.0e6, 8.0e6, 4.6e6, 8.6e6), CRS)
    inner = shapely.box(4.1e6, 8.1e6, 4.5e6, 8.5e6)
    pieces = [shapely.intersection(g, inner) for g in grid.geometry]
    assert math.isclose(sum(p.area for p in pieces), inner.area, rel_tol=1e-12)


def test_ids_stable_across_bounds(lattice):
    a = H.build_grid(lattice, (4.0e6, 8.0e6, 4.3e6, 8.3e6), CRS).set_index("hex_id")
    b = H.build_grid(lattice, (3.5e6, 7.5e6, 4.6e6, 8.6e6), CRS).set_index("hex_id")
    common = a.index.intersection(b.index)
    assert len(common) == len(a)
    assert all(a.geometry[i].equals_exact(b.geometry[i], 0) for i in common)


def test_neighbours_share_edges_bitwise_after_reprojection(lattice):
    grid = H.build_grid(lattice, (4.0e6, 8.0e6, 4.3e6, 8.3e6), CRS)
    ll = H.grid_to_lonlat(lattice, grid, 1000.0).set_index("hex_id")
    for c, r in [(20, 22), (21, 22)]:  # even and odd column
        me = {tuple(p) for p in shapely.get_coordinates(ll.geometry[H.hex_id(c, r)])}
        shared = 0
        for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1 if c % 2 else -1),
                       (-1, 1 if c % 2 else -1)]:
            nb = {tuple(p) for p in shapely.get_coordinates(ll.geometry[H.hex_id(c + dc, r + dr)])}
            common = me & nb
            assert len(common) == 51, (c, r, dc, dr, len(common))  # 50 segments per edge
            shared += 1
        assert shared == 6


def test_densified_edge_stays_on_straight_5880_edge(lattice):
    ring = lattice.densified_ring(30, 30, 1000.0)
    assert len(ring) == 6 * 50
    poly = lattice.polygon(30, 30)
    assert shapely.distance(shapely.points(ring), poly.exterior).max() < 1e-6


def test_assign_points_one_cell_each(lattice):
    grid = H.build_grid(lattice, (4.0e6, 8.0e6, 4.3e6, 8.3e6), CRS)
    cx, cy = lattice.center(20, 22)
    vx, vy = lattice.xy(*lattice.vertex_index(20, 22)[0])  # vertex shared by 3 cells
    pts = gpd.GeoDataFrame({"id_bem": [1, 2]}, geometry=shapely.points([(cx, cy), (vx, vy)]),
                           crs=CRS)
    out = H.assign_points(pts, grid).set_index("id_bem")["hex_id"]
    assert out[1] == H.hex_id(20, 22)
    assert len(out) == 2


def test_clip_by_hex_partitions_dissolve(lattice):
    # buffers centred on a shared vertex, an edge and inside a cell, plus overlapping pair
    tr = Transformer.from_crs(CRS, B.CRS_GEOGRAPHIC, always_xy=True)
    centres = [lattice.xy(*lattice.vertex_index(20, 22)[0]),
               lattice.xy(*lattice.vertex_index(20, 22)[1]),
               lattice.center(20, 22), lattice.center(21, 22)]
    centres.append((centres[-1][0] + 120, centres[-1][1]))
    lon, lat = tr.transform(*np.array(centres).T)
    bufs = B.geodesic_buffers(lon, lat, 100, 256)
    dissolved = shapely.union_all(bufs)
    grid = H.build_grid(lattice, (4.0e6, 8.0e6, 4.3e6, 8.3e6), CRS)
    units = H.clip_by_hex(dissolved, H.grid_to_lonlat(lattice, grid, 1000.0))
    assert units["unit"].is_unique
    assert abs(H.partition_closure(dissolved, units)["rel"]) < 1e-9
    assert len(units) >= 4
    # overlapping pair counted once: dissolve area < Σ individual areas
    assert B.geodesic_areas([dissolved])[0] < B.geodesic_areas(bufs).sum()
