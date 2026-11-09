"""Hexagonal grid for the maps of anthropic change (Fig 2 B–E).

The grid is regular in `grid_crs` (EPSG:5880), flat-top, with a fixed origin
(config/hex.toml), so cells and `hex_id` do not depend on the data. Vertices are
built from an integer lattice — x = x0 + k·R/2, y = y0 + m·h (R circumradius,
h = R·√3/2) — so the vertices shared by neighbouring cells are bitwise
identical; edges are densified in a canonical direction for the same reason.
Reprojected to lon/lat, neighbours still share their edges exactly, and the
national dissolve clipped by the cells adds up to the whole (no gaps, no
double counting).

Sites are assigned to cells by their point (planar, in `grid_crs`); buffer area
is assigned by clipping the national dissolve in lon/lat.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from sitios_arq.buffers import CRS_GEOGRAPHIC, _polygonal, geodesic_areas

# Vertex offsets of a flat-top hexagon in lattice units (x: R/2, y: h), counter-clockwise
# from angle 0°.
_VERTEX_K = (2, 1, -1, -2, -1, 1)
_VERTEX_M = (0, 1, 1, 0, -1, -1)

REQUIRED = ("radius_m", "grid_crs", "hex_width_km", "orientation", "origin_xy_m",
            "edge_densify_m", "min_sites", "periods")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Read config/hex.toml. Each top-level table is one parameter (description + value)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for name, p in cfg.items():
        if not isinstance(p, dict) or "value" not in p:
            raise ValueError(f"Parameter '{name}' needs a 'value'")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Missing parameters: {', '.join(missing)}")
    if cfg["orientation"]["value"] != "flat-top":
        raise ValueError("Only orientation = 'flat-top' is implemented")
    for start, end in cfg["periods"]["value"]:
        if not (isinstance(start, int) and isinstance(end, int) and start < end):
            raise ValueError(f"'periods': invalid period [{start}, {end}]")
    return cfg


def periods(cfg: dict) -> list[tuple[int, int]]:
    """(start, end) of the periods of config/hex.toml (Fig 2)."""
    return [(s, e) for s, e in cfg["periods"]["value"]]


def period_years(cfg: dict) -> list[int]:
    """Years needed by all periods (endpoints), sorted."""
    return sorted({y for p in periods(cfg) for y in p})


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def hex_id(col: int, row: int) -> str:
    return f"c{col}_r{row}"


class HexLattice:
    """Flat-top hexagonal lattice: width = 2R (vertex to vertex), origin = centre of (0, 0)."""

    def __init__(self, width_m: float, origin_xy: tuple[float, float]):
        self.R = width_m / 2.0
        self.h = self.R * math.sqrt(3) / 2.0
        self.x0, self.y0 = float(origin_xy[0]), float(origin_xy[1])

    def vertex_index(self, col: int, row: int) -> list[tuple[int, int]]:
        """Integer lattice indices (k, m) of the 6 vertices of cell (col, row)."""
        kc, mc = 3 * col, 2 * row + (col % 2)
        return [(kc + dk, mc + dm) for dk, dm in zip(_VERTEX_K, _VERTEX_M)]

    def xy(self, k: int, m: int) -> tuple[float, float]:
        return self.x0 + k * (self.R / 2.0), self.y0 + m * self.h

    def center(self, col: int, row: int) -> tuple[float, float]:
        return self.xy(3 * col, 2 * row + (col % 2))

    def cells_covering(self, bounds) -> list[tuple[int, int]]:
        """All (col, row) whose cell may intersect the box (minx, miny, maxx, maxy)."""
        minx, miny, maxx, maxy = bounds
        c0 = math.floor((minx - self.x0) / (1.5 * self.R)) - 1
        c1 = math.ceil((maxx - self.x0) / (1.5 * self.R)) + 1
        r0 = math.floor((miny - self.y0) / (2 * self.h)) - 1
        r1 = math.ceil((maxy - self.y0) / (2 * self.h)) + 1
        return [(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)]

    def polygon(self, col: int, row: int) -> shapely.Polygon:
        """Cell with its 6 vertices (straight edges in the grid CRS)."""
        return shapely.Polygon([self.xy(k, m) for k, m in self.vertex_index(col, row)])

    def densified_ring(self, col: int, row: int, max_seg_m: float) -> np.ndarray:
        """Ring vertices (n, 2) with points every ≤ max_seg_m along each edge.

        Each edge is interpolated from its lower lattice endpoint, so two neighbours
        produce bitwise-identical points on their shared edge.
        """
        idx = self.vertex_index(col, row)
        n = max(1, math.ceil(self.R / max_seg_m))  # edge length = R
        t = np.arange(n) / n
        pts = []
        for a, b in zip(idx, idx[1:] + idx[:1]):
            lo, hi = (a, b) if a < b else (b, a)
            p, q = np.array(self.xy(*lo)), np.array(self.xy(*hi))
            seg = p + t[:, None] * (q - p)              # lo → hi, without hi
            seg = np.vstack([seg, q])                   # lo → hi, closed
            if a != lo:
                seg = seg[::-1]                         # a → b
            pts.append(seg[:-1])                        # drop b (start of next edge)
        return np.vstack(pts)


def build_grid(lattice: HexLattice, bounds, crs) -> gpd.GeoDataFrame:
    """Cells covering `bounds` (grid CRS): hex_id, col, row, straight-edged geometry."""
    cells = lattice.cells_covering(bounds)
    return gpd.GeoDataFrame(
        {"hex_id": [hex_id(c, r) for c, r in cells],
         "col": [c for c, _ in cells], "row": [r for _, r in cells]},
        geometry=[lattice.polygon(c, r) for c, r in cells], crs=crs,
    )


def grid_to_lonlat(lattice: HexLattice, grid: gpd.GeoDataFrame, max_seg_m: float) -> gpd.GeoDataFrame:
    """Densified cells reprojected to EPSG:4674 (same hex_id order as `grid`)."""
    tr = Transformer.from_crs(grid.crs, CRS_GEOGRAPHIC, always_xy=True)
    geoms = []
    for c, r in zip(grid["col"], grid["row"]):
        ring = lattice.densified_ring(c, r, max_seg_m)
        lon, lat = tr.transform(ring[:, 0], ring[:, 1])
        geoms.append(shapely.Polygon(np.column_stack([lon, lat])))
    return gpd.GeoDataFrame(grid[["hex_id", "col", "row"]].copy(), geometry=geoms,
                            crs=CRS_GEOGRAPHIC)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_points(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, key: str = "id_bem") -> pd.DataFrame:
    """One cell per point: [key, hex_id]. A point on a shared edge goes to the smallest (col, row)."""
    j = gpd.sjoin(points[[key, "geometry"]], grid[["hex_id", "col", "row", "geometry"]],
                  how="left", predicate="intersects")
    if j["hex_id"].isna().any():
        raise ValueError(f"{int(j['hex_id'].isna().sum())} points outside the grid")
    j = j.sort_values([key, "col", "row"]).drop_duplicates(key, keep="first")
    return j[[key, "hex_id"]].reset_index(drop=True)


def clip_by_hex(dissolved, cells_ll: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split a lon/lat (multi)polygon among the cells: GeoDataFrame[unit, geometry].

    Parts inside a single cell are kept whole; parts crossing edges are intersected
    with each cell they touch.
    """
    parts = shapely.get_parts(dissolved)
    tree = shapely.STRtree(cells_ll.geometry.values)
    p_idx, c_idx = tree.query(parts, predicate="intersects")
    pieces: dict[int, list] = {}
    for p in np.unique(p_idx):
        cand = c_idx[p_idx == p]
        part = parts[p]
        if len(cand) == 1 and shapely.covered_by(part, cells_ll.geometry.values[cand[0]]):
            pieces.setdefault(int(cand[0]), []).append(part)
            continue
        for c in cand:
            piece = _polygonal(shapely.intersection(part, cells_ll.geometry.values[c]))
            if piece is not None:
                pieces.setdefault(int(c), []).append(piece)
    order = sorted(pieces)
    return gpd.GeoDataFrame(
        {"unit": cells_ll["hex_id"].to_numpy()[order]},
        geometry=[shapely.union_all(pieces[c]) for c in order], crs=cells_ll.crs,
    )


def partition_closure(dissolved, units: gpd.GeoDataFrame) -> dict:
    """Geodesic areas of the dissolve and of its pieces; rel = Σ pieces / whole − 1."""
    whole = float(geodesic_areas([dissolved])[0])
    parts = float(geodesic_areas(units.geometry).sum())
    return {"area_dissolved_m2": whole, "area_units_m2": parts, "rel": parts / whole - 1}
