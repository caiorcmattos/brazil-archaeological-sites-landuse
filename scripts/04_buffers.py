"""Geodesic buffers around the filtered sites, national and per-biome dissolves.

Usage:
    python scripts/04_buffers.py

Inputs: <SITIOS_DATA_ROOT>/processed/sites_filtered.parquet (scripts/03_filter_sites.py),
<SITIOS_DATA_ROOT>/processed/biomas_5880.parquet, config/extraction.toml
(paths resolved in sitios_arq.paths).

Outputs (in <SITIOS_DATA_ROOT>/processed/buffers/), all EPSG:4674:
- buffers_{r}m.parquet                 one geodesic buffer per site (id_bem, biome, biome_dist_m)
- buffers_{r}m_dissolved.parquet       national dissolve (unit = NATIONAL)
- buffers_{r}m_dissolved_biome.parquet national dissolve clipped by biome (unit = biome)
- buffer_overlap.csv                   areas, overlap and area merged from outside the biomes
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from sitios_arq import buffers as B
from sitios_arq.extraction import NATIONAL, load_config
from sitios_arq.paths import BUFFER_DIR as OUT_DIR
from sitios_arq.paths import OUT_BIOMAS as BIOMES
from sitios_arq.paths import SITES_FILTERED as SITES
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/extraction.toml"

TOL_NGON = 1e-6       # geodesic area of each buffer vs n-gon πr²
TOL_PARTITION = 1e-9  # Σ biome areas vs national area


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("=== 04_buffers.py ===\n")

    cfg = load_config(CONFIG)
    out_dir = OUT_DIR

    radii = cfg["radii_m"]["value"]
    n_sites = cfg["n_sites"]["value"]
    n_vertices = cfg["n_vertices"]["value"]
    factor = B.polygon_area_factor(n_vertices)

    sites = gpd.read_parquet(SITES)
    assert len(sites) == n_sites, f"expected {n_sites} sites, got {len(sites)}"
    assert sites.crs.to_epsg() == B.CRS_METRIC, sites.crs
    assert sites["id_bem"].is_unique
    biomes = gpd.read_parquet(BIOMES)
    assert biomes.crs.to_epsg() == B.CRS_METRIC, biomes.crs

    n_null = int(sites["biome"].isna().sum())
    sites = B.fill_nearest_biome(sites, biomes)
    d = sites.loc[sites["biome_dist_m"] > 0, "biome_dist_m"]
    print(f"Nearest biome assigned to {n_null} sites "
          f"(distance m: median {d.median():.0f}, max {d.max():.0f})")
    print(sites.loc[sites["biome_dist_m"] > 0, "biome"].value_counts().to_string(), "\n")

    biomes_ll = B.disjoint_biomes(biomes.to_crs(B.CRS_GEOGRAPHIC))

    rows = []

    for r in radii:
        print(f"── Radius {r} m ──")
        geoms = B.geodesic_buffers(sites["lon"], sites["lat"], r, n_vertices)
        buf = gpd.GeoDataFrame(
            sites[["id_bem", "biome", "biome_dist_m"]].reset_index(drop=True),
            geometry=geoms, crs=B.CRS_GEOGRAPHIC,
        )
        area_ind = B.geodesic_areas(buf.geometry)
        rel = area_ind / (np.pi * r ** 2 * factor) - 1
        print(f"  geodesic area / (πr²·{factor:.6f}) − 1: max |rel| = {np.abs(rel).max():.2e}")
        if np.abs(rel).max() > TOL_NGON:
            raise RuntimeError(f"buffer area deviates from the n-gon beyond {TOL_NGON}")

        save(buf, out_dir / f"buffers_{r}m.parquet")

        dissolved = shapely.union_all(buf.geometry.values)
        area_diss = B.geodesic_areas([dissolved])[0]
        nat = gpd.GeoDataFrame({"unit": [NATIONAL]}, geometry=[dissolved], crs=B.CRS_GEOGRAPHIC)
        save(nat, out_dir / f"buffers_{r}m_dissolved.parquet")
        overlap = 100 * (1 - area_diss / area_ind.sum())
        print(f"  national: individual {area_ind.sum() / 1e6:.2f} km² → dissolved "
              f"{area_diss / 1e6:.2f} km² (overlap {overlap:.2f}%)")

        by_biome, outside = B.clip_by_biome(dissolved, biomes_ll, biomes)
        by_biome = by_biome.rename(columns={"biome": "unit"})
        area_b = pd.Series(B.geodesic_areas(by_biome.geometry), index=by_biome["unit"])
        rel_part = (area_b.sum() - area_diss) / area_diss
        print(f"  Σ biomes vs national: rel = {rel_part:.2e}; merged from outside the "
              f"biomes: {outside.sum() / 1e6:.4f} km²")
        if abs(rel_part) > TOL_PARTITION:
            raise RuntimeError(f"Σ biome areas differ from national by {rel_part:.2e}")
        save(by_biome, out_dir / f"buffers_{r}m_dissolved_biome.parquet")

        rows.append({"radius_m": r, "unit": NATIONAL, "n_sites": len(buf),
                     "area_individual_km2": area_ind.sum() / 1e6,
                     "area_dissolved_km2": area_diss / 1e6, "overlap_pct": overlap,
                     "area_outside_biomes_km2": outside.sum() / 1e6})
        ind_by_site_biome = pd.Series(area_ind).groupby(buf["biome"]).sum()
        n_by_site_biome = buf["biome"].value_counts()
        for unit, a in area_b.items():
            rows.append({"radius_m": r, "unit": unit,
                         "n_sites": int(n_by_site_biome.get(unit, 0)),
                         "area_individual_km2": ind_by_site_biome.get(unit, 0.0) / 1e6,
                         "area_dissolved_km2": a / 1e6, "overlap_pct": np.nan,
                         "area_outside_biomes_km2": outside.get(unit, 0.0) / 1e6})

    table = pd.DataFrame(rows)
    save(table, out_dir / "buffer_overlap.csv")
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
