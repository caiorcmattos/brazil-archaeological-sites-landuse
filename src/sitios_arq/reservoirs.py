"""Archaeological sites and artificial water bodies (ANA): association and water status.

A site is reservoir-associated when its buffer intersects an
artificial water body of the ANA `massa_dagua` layer (any use; pre- and post-1985
reservoirs alike). Its water status compares the water share of the buffer in the
start and end years (`config/reservoirs.toml`):

- entirely_water_both  entirely classified as water in both years (flooded by the start year)
- entirely_water_end   entirely classified as water in the end year only (after the start year)
- partial_gain         water share increased, but the buffer is not entirely water at the end
- no_gain              water share did not increase

"Entirely classified as water" means a water share of exactly 1: no area of any other
class in the buffer. Water shares are over the area with data (class 0 rule).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

STATUSES = ("entirely_water_both", "entirely_water_end", "partial_gain", "no_gain")
REQUIRED = ("water_body_type", "association", "radius_m", "years", "entirely_water",
            "sensitivity_min_share", "hydro_use", "top_hydro")


def load_config(path: str | Path) -> dict:
    """Read config/reservoirs.toml. Each top-level table is one parameter (description + value)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for name, p in cfg.items():
        if not isinstance(p, dict) or "value" not in p:
            raise ValueError(f"Parameter '{name}' needs a 'value'")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"Missing parameters: {', '.join(missing)}")
    if cfg["association"]["value"] != "intersects":
        raise ValueError("Only association = 'intersects' is implemented")
    if cfg["entirely_water"]["value"] != "exact":
        raise ValueError("Only entirely_water = 'exact' is implemented")
    if not all(0 < m < 1 for m in cfg["sensitivity_min_share"]["value"]):
        raise ValueError("sensitivity_min_share values must be in (0, 1)")
    start, end = cfg["years"]["value"]
    if not start < end:
        raise ValueError(f"years must be [start, end] with start < end, got {cfg['years']['value']}")
    return cfg


def prepare_water_bodies(bodies: gpd.GeoDataFrame, cfg: dict, crs) -> gpd.GeoDataFrame:
    """Kept ANA water bodies with id, name, use and hydropower flag, valid and in `crs`.

    Name: `nmoriginal`, else `nmespecifi`, else None. Id: ANA `gid` (must be unique;
    names repeat across distinct water bodies and are not identifiers).
    """
    kept = bodies[bodies["detipomass"] == cfg["water_body_type"]["value"]]
    kept = kept[kept.geometry.notna() & ~kept.geometry.is_empty]
    if kept["gid"].duplicated().any():
        raise ValueError("ANA gid is not unique among the kept water bodies")

    def clean(col):
        s = kept[col].astype("string").str.strip()
        return s.where(s != "")

    name = clean("nmoriginal").fillna(clean("nmespecifi"))
    out = gpd.GeoDataFrame({
        "res_id": kept["gid"].astype("int64"),
        "res_name": name.astype(object).where(name.notna(), None),
        "res_use": kept["usoprinc"].astype(object).where(kept["usoprinc"].notna(), None),
        "is_hydro": (kept["usoprinc"] == cfg["hydro_use"]["value"]).to_numpy(),
    }, geometry=kept.geometry.make_valid().to_numpy(), crs=kept.crs)
    return out.to_crs(crs)


def link_sites(buffers: gpd.GeoDataFrame, bodies: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per (site, water body) whose geometries intersect."""
    if buffers.crs != bodies.crs:
        raise ValueError(f"CRS differ: {buffers.crs} vs {bodies.crs}")
    j = gpd.sjoin(buffers[["id_bem", "biome", "geometry"]], bodies, how="inner",
                  predicate="intersects")
    cols = ["id_bem", "biome", "res_id", "res_name", "res_use", "is_hydro"]
    return (pd.DataFrame(j[cols]).drop_duplicates(["id_bem", "res_id"])
            .sort_values(["id_bem", "res_id"], ignore_index=True))


def water_status(start: np.ndarray, end: np.ndarray, min_share: float = 1.0) -> np.ndarray:
    """Status of each buffer from its water share (0–1) in the start and end years.

    Entirely water: share exactly 1 (the rule, `min_share` = 1) or ≥ `min_share` (sensitivity,
    same threshold in both years). Partial gain and no gain are the rest.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    if np.isnan(start).any() or np.isnan(end).any():
        raise ValueError("water share is NaN")
    if not 0 < min_share <= 1:
        raise ValueError(f"min_share must be in (0, 1], got {min_share}")
    ws, we = (start == 1, end == 1) if min_share == 1 else (start >= min_share, end >= min_share)
    return np.select([we & ws, we, end > start], list(STATUSES[:3]), default=STATUSES[3])


def site_table(links: pd.DataFrame, water: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """One row per associated site: water share (%) in both years, Δ (pp), status, hydro flag.

    `water`: id_bem, year, prop_water (from thematic.aggregate_categories).
    """
    start, end = years
    w = water[water["year"].isin(years)].pivot(index="id_bem", columns="year", values="prop_water")
    sites = (links.groupby("id_bem")
             .agg(biome=("biome", "first"), n_reservoirs=("res_id", "nunique"),
                  is_hydro=("is_hydro", "any"))
             .reset_index())
    if links.groupby("id_bem")["biome"].nunique().gt(1).any():
        raise ValueError("site with more than one biome in the links")
    missing = set(sites["id_bem"]) - set(w.dropna().index)
    if missing:
        raise ValueError(f"{len(missing)} associated sites without water share in {years}")
    ps, pe = (w.loc[sites["id_bem"], y].to_numpy() for y in (start, end))
    sites["water_pct_start"] = 100 * ps
    sites["water_pct_end"] = 100 * pe
    sites["delta_pp"] = sites["water_pct_end"] - sites["water_pct_start"]
    sites["status"] = water_status(ps, pe)
    sites.insert(1, "start", start)
    sites.insert(2, "end", end)
    return sites
