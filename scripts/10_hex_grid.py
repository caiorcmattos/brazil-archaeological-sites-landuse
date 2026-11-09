"""Hexagonal grid for the maps of anthropic change (Fig 2 B–E).

Usage:
    python scripts/10_hex_grid.py

Builds the fixed-origin flat-top grid of config/hex.toml in EPSG:5880, assigns
every site to one cell by its point, and clips the national dissolve of the
buffers (radius_m) by the cells in lon/lat, so each square metre of buffer
belongs to exactly one cell and the cells add up to NATIONAL (no double
counting of overlapping buffers). The clipped pieces are the input of the
MapBiomas extraction for the hexagons (scripts/11_extract_hex.py).

Inputs: <SITIOS_DATA_ROOT>/processed/sites_filtered.parquet (raw lon/lat),
processed/buffers/buffers_{r}m.parquet and buffers_{r}m_dissolved.parquet,
config/hex.toml.

Outputs (in processed/hex/):
- hex_grid_{w}km.parquet  occupied cells (sites or buffer area): hex_id, col, row,
                          n_sites, sparse (n_sites < min_sites), buffer_area_m2,
                          cell_area_m2 (geodesic); straight-edged geometry, EPSG:5880
- hex_sites_r{r}.parquet  id_bem, hex_id (one cell per site)
- hex_units_r{r}.parquet  national dissolve clipped by cell: unit (= hex_id), EPSG:4674
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sitios_arq import hexgrid as H
from sitios_arq.buffers import CRS_GEOGRAPHIC, geodesic_areas
from sitios_arq.extraction import load_config as load_extraction_config
from sitios_arq.paths import BUFFER_DIR, HEX_DIR
from sitios_arq.paths import SITES_FILTERED as SITES
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/hex.toml"
EXTRACTION_CONFIG = ROOT / "config/extraction.toml"

TOL_PARTITION = 1e-9  # Σ cell pieces vs national dissolve (geodesic area)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("=== 10_hex_grid.py ===\n")

    cfg = H.load_config(CONFIG)
    out_dir = HEX_DIR

    r = cfg["radius_m"]["value"]
    n_expected = load_extraction_config(EXTRACTION_CONFIG)["n_sites"]["value"]
    crs = cfg["grid_crs"]["value"]
    width_km = cfg["hex_width_km"]["value"]
    min_sites = cfg["min_sites"]["value"]
    densify_m = cfg["edge_densify_m"]["value"]
    lattice = H.HexLattice(width_km * 1000.0, tuple(cfg["origin_xy_m"]["value"]))

    # ── Inputs ────────────────────────────────────────────────────────────────
    sites = pd.read_parquet(SITES, columns=["id_bem", "lon", "lat"])
    assert len(sites) == n_expected, f"expected {n_expected} sites, got {len(sites)}"
    assert sites["id_bem"].is_unique
    buf_path = BUFFER_DIR / f"buffers_{r}m.parquet"
    diss_path = BUFFER_DIR / f"buffers_{r}m_dissolved.parquet"
    buf_ids = pd.read_parquet(buf_path, columns=["id_bem"])["id_bem"]
    if set(buf_ids) != set(sites["id_bem"]):
        raise RuntimeError("sites_filtered and the buffers have different id_bem sets")
    dissolved_gdf = gpd.read_parquet(diss_path)
    assert dissolved_gdf.crs.to_epsg() == CRS_GEOGRAPHIC and len(dissolved_gdf) == 1
    dissolved = dissolved_gdf.geometry.iloc[0]

    points = gpd.GeoDataFrame(
        sites[["id_bem"]], geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
        crs=CRS_GEOGRAPHIC,
    ).to_crs(crs)

    # ── Grid ──────────────────────────────────────────────────────────────────
    diss_bounds = dissolved_gdf.to_crs(crs).total_bounds
    pb = points.total_bounds
    bounds = (min(pb[0], diss_bounds[0]), min(pb[1], diss_bounds[1]),
              max(pb[2], diss_bounds[2]), max(pb[3], diss_bounds[3]))
    x0, y0 = lattice.x0, lattice.y0
    if not (x0 < bounds[0] and y0 < bounds[1]):
        print(f"WARNING: origin ({x0:.0f}, {y0:.0f}) not west/south of the data bounds")
    grid = H.build_grid(lattice, bounds, crs)
    cells_ll = H.grid_to_lonlat(lattice, grid, densify_m)
    print(f"Grid: {len(grid)} cells covering the data (width {width_km} km, "
          f"origin {x0:.0f}, {y0:.0f}, {crs})")

    # ── Sites → cells (by point) ──────────────────────────────────────────────
    site_hex = H.assign_points(points, grid)
    assert len(site_hex) == n_expected and site_hex["id_bem"].is_unique
    n_sites = site_hex.groupby("hex_id").size().rename("n_sites")

    # ── National dissolve → cells (clip in lon/lat) ───────────────────────────
    units = H.clip_by_hex(dissolved, cells_ll)
    closure = H.partition_closure(dissolved, units)
    print(f"Σ cell pieces vs national dissolve: {closure['area_units_m2'] / 1e6:.4f} vs "
          f"{closure['area_dissolved_m2'] / 1e6:.4f} km², rel = {closure['rel']:.2e}")
    if abs(closure["rel"]) > TOL_PARTITION:
        raise RuntimeError(f"Partition does not close: rel = {closure['rel']:.2e}")
    buffer_area = pd.Series(geodesic_areas(units.geometry), index=units["unit"],
                            name="buffer_area_m2")

    # ── Cell table ────────────────────────────────────────────────────────────
    occupied = sorted(set(n_sites.index) | set(buffer_area.index))
    cells = grid[grid["hex_id"].isin(occupied)].copy()
    cells["n_sites"] = cells["hex_id"].map(n_sites).fillna(0).astype(int)
    cells["sparse"] = cells["n_sites"] < min_sites
    cells["buffer_area_m2"] = cells["hex_id"].map(buffer_area).fillna(0.0)
    ll = cells_ll.set_index("hex_id").loc[cells["hex_id"]]
    cells["cell_area_m2"] = geodesic_areas(ll.geometry)
    cells = cells.sort_values(["col", "row"], ignore_index=True)

    no_area = cells[(cells["n_sites"] > 0) & (cells["buffer_area_m2"] <= 0)]
    if len(no_area):
        raise RuntimeError(f"{len(no_area)} cells with sites but no buffer area")
    area_only = cells[cells["n_sites"] == 0]
    sparse = cells[cells["sparse"]]

    print(f"\nOccupied cells: {len(cells)}  (with ≥ {min_sites} sites: {(~cells['sparse']).sum()}; "
          f"sparse: {len(sparse)}, of which {len(area_only)} with buffer area but no site point)")
    print(f"Sites in sparse cells: {sparse['n_sites'].sum()} "
          f"({100 * sparse['n_sites'].sum() / n_expected:.2f}%); buffer area in sparse cells: "
          f"{sparse['buffer_area_m2'].sum() / 1e6:.3f} km² "
          f"({100 * sparse['buffer_area_m2'].sum() / closure['area_dissolved_m2']:.2f}%)")
    print(f"Sites per cell: median {cells['n_sites'].median():.0f}, max {cells['n_sites'].max()}")
    rel_cell = cells["cell_area_m2"] / (3 * np.sqrt(3) / 8 * (width_km * 1000) ** 2) - 1
    print(f"Cell geodesic area / nominal − 1: {rel_cell.min():+.3%} to {rel_cell.max():+.3%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save(cells, out_dir / f"hex_grid_{width_km}km.parquet")
    save(site_hex, out_dir / f"hex_sites_r{r}.parquet")
    save(units, out_dir / f"hex_units_r{r}.parquet")
    print(f"\nSaved in {out_dir}")


if __name__ == "__main__":
    main()
