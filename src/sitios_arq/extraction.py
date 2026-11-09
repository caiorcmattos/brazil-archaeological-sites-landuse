"""MapBiomas C11 area-by-class extraction inside buffers (exactextract).

The class rasters are EPSG:4326, so pixel area varies with latitude.
`pixel_area_c11_m2.tif` holds the geodesic area of every pixel on the same
grid and is used as the weight raster:

    area_of_class_c = weighted_frac_c × Σ(coverage × pixel_area)

exactextract's `weighted_frac` is the share of the weighted area taken by each
class; its denominator is Σ(coverage × weight) over valid cells. The class
rasters have `nodata = None` (every cell is valid), so that denominator is the
geodesic area of the polygon covered by the grid — obtained by a second call,
`sum` over the area raster. The product is therefore exact. A class raster with
nodata set would silently shrink the denominator, so the grid checks reject it.

Two exactextract calls on purpose: named ops over several rasters
(`sum(raster_name)`) segfault in exactextract 0.2.2 and 0.3.0.
"""

from __future__ import annotations

import logging
import math
import time
import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from sitios_arq.buffers import polygon_area_factor

log = logging.getLogger("extraction")

NATIONAL = "NATIONAL"


# ---------------------------------------------------------------------------
# Configuration and tasks
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Read config/extraction.toml. Each top-level table is one parameter (description + value)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for name, p in cfg.items():
        if not isinstance(p, dict) or "value" not in p:
            raise ValueError(f"Parameter '{name}' needs a 'value'")
    if cfg["main_radius_m"]["value"] not in cfg["radii_m"]["value"]:
        raise ValueError("'main_radius_m' must be one of 'radii_m'")
    return cfg


def series_years(cfg: dict) -> list[int]:
    first, last = cfg["series_years"]["value"]
    return list(range(first, last + 1))


def build_tasks(cfg: dict) -> list[tuple[int, int]]:
    """(radius, year) pairs. The order defines the task id."""
    main = cfg["main_radius_m"]["value"]
    tasks = [(main, y) for y in series_years(cfg)]
    for r in cfg["radii_m"]["value"]:
        if r != main:
            tasks += [(r, y) for y in cfg["sensitivity_years"]["value"]]
    return tasks


def format_array(ids: list[int]) -> str:
    """Compact [7, 8, 9, 19] into '7-9,19' (missing task ids)."""
    if not ids:
        return ""
    chunks, start, prev = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        chunks.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ",".join(chunks)


# ---------------------------------------------------------------------------
# Grid consistency
# ---------------------------------------------------------------------------

def read_header(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "file": Path(path).name,
            "width": src.width,
            "height": src.height,
            "epsg": src.crs.to_epsg() if src.crs else None,
            "transform": tuple(src.transform)[:6],
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "block": src.block_shapes[0],
        }


def compare_grid(h: dict, ref: dict) -> list[str]:
    """Differences of header `h` from `ref`. Empty list = same grid."""
    problems = [f"{k}: {h[k]} != {ref[k]}" for k in ("width", "height", "epsg") if h[k] != ref[k]]
    if not rasterio.Affine(*h["transform"]).almost_equals(
            rasterio.Affine(*ref["transform"]), precision=1e-12):
        problems.append(f"transform: {h['transform']} != {ref['transform']}")
    return problems


def validate_class_raster(h: dict) -> list[str]:
    problems = []
    if h["nodata"] is not None:
        problems.append(f"nodata={h['nodata']} (expected None: nodata shrinks the "
                        "weighted_frac denominator and would bias the areas)")
    if not np.issubdtype(np.dtype(h["dtype"]), np.integer):
        problems.append(f"dtype={h['dtype']} (expected integer)")
    return problems


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def prepare(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert to EPSG:4326 (numerical identity from 4674), sort by Hilbert curve.

    Spatial ordering is the optimisation that matters: exactextract reads raster
    blocks on demand, and neighbouring features reuse decompressed blocks.
    """
    epsg = gdf.crs.to_epsg()
    if epsg == 4674:
        before = shapely.get_coordinates(gdf.geometry.values)
        gdf = gdf.to_crs(4326)
        if not np.array_equal(before, shapely.get_coordinates(gdf.geometry.values)):
            raise RuntimeError("EPSG:4674 → 4326 changed coordinates; expected identity")
    elif epsg != 4326:
        raise ValueError(f"Buffers must be EPSG:4674 or 4326, got {gdf.crs}")
    gdf = gdf.assign(_hilbert=gdf.geometry.hilbert_distance())
    gdf = gdf.sort_values("_hilbert").drop(columns="_hilbert").reset_index(drop=True)
    gdf["_fid"] = np.arange(len(gdf), dtype=np.int64)
    return gdf


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def explode_classes(o1: pd.DataFrame) -> pd.DataFrame:
    """Array columns `unique` / `weighted_frac` to long format."""
    sizes = o1["unique"].map(lambda a: 0 if a is None else len(a)).to_numpy()
    if sizes.sum() == 0:
        return pd.DataFrame({
            "_fid": np.array([], dtype=np.int64),
            "mapbiomas_class": np.array([], dtype=np.int64),
            "_wf": np.array([], dtype=float),
        })
    full = sizes > 0
    return pd.DataFrame({
        "_fid": np.repeat(o1["_fid"].to_numpy(), sizes),
        "mapbiomas_class": np.concatenate(o1.loc[full, "unique"].to_list()),
        "_wf": np.concatenate(o1.loc[full, "weighted_frac"].to_list()),
    })


def extract(class_path: Path, area_path: Path, gdf: gpd.GeoDataFrame,
            batch: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exactextract on `gdf` (from `prepare`). Returns (classes_long, totals).

    Batches only bound peak memory and give progress; the result is identical
    to a single call.
    """
    from exactextract import exact_extract

    cls_s, area_s = str(class_path), str(area_path)
    cls_blocks, tot_blocks = [], []
    t0 = time.time()
    for start in range(0, len(gdf), batch):
        chunk = gdf.iloc[start:start + batch]
        o1 = exact_extract(cls_s, chunk, ["unique", "weighted_frac"],
                           weights=area_s, include_cols=["_fid"], output="pandas")
        o2 = exact_extract(area_s, chunk, ["sum"], include_cols=["_fid"], output="pandas")
        cls_blocks.append(explode_classes(o1))
        tot_blocks.append(o2.rename(columns={"sum": "extracted_area_m2"})[["_fid", "extracted_area_m2"]])
        end = min(start + batch, len(gdf))
        dt = time.time() - t0
        log.info("    %d/%d features (%.0f%%) — %.1fs (%.2f ms/feature)",
                 end, len(gdf), 100 * end / len(gdf), dt, 1000 * dt / end)

    classes = pd.concat(cls_blocks, ignore_index=True)
    totals = pd.concat(tot_blocks, ignore_index=True)
    # A feature entirely off the grid: empty `unique`, NaN `sum`. It produces no
    # class rows but stays in the coverage table with coverage_frac = 0.
    totals["extracted_area_m2"] = totals["extracted_area_m2"].fillna(0.0).astype(float)

    classes = classes.merge(totals, on="_fid", how="left")
    classes["area_m2"] = classes["_wf"] * classes["extracted_area_m2"]
    classes["mapbiomas_class"] = classes["mapbiomas_class"].astype("int16")
    return classes[["_fid", "mapbiomas_class", "area_m2"]], totals


# ---------------------------------------------------------------------------
# Sanity checks (used by 07_merge_extraction.py)
# ---------------------------------------------------------------------------

TOL_GEODESIC = 1e-6


def site_closure(sites: pd.DataFrame, coverage: pd.DataFrame, n_vertices: int) -> pd.DataFrame:
    """Σ class areas vs geodesic area, and geodesic area vs n-gon πr², per site."""
    total = (sites.groupby(["id_bem", "radius_m", "year"], as_index=False)["area_m2"].sum()
             .rename(columns={"area_m2": "sum_classes_m2"}))
    cov = coverage[coverage["level"] == "site"].copy()
    cov["id_bem"] = cov["key"].astype("int64")
    df = total.merge(
        cov[["id_bem", "radius_m", "year", "geodesic_area_m2", "extracted_area_m2", "coverage_frac"]],
        on=["id_bem", "radius_m", "year"], how="outer", indicator=True,
    )
    df["orphan"] = df.pop("_merge") != "both"
    df["sum_classes_m2"] = df["sum_classes_m2"].fillna(0.0)
    df["rel_geodesic"] = (df["sum_classes_m2"] - df["geodesic_area_m2"]) / df["geodesic_area_m2"]
    expected = math.pi * df["radius_m"].astype(float) ** 2 * polygon_area_factor(n_vertices)
    df["rel_ngon"] = df["geodesic_area_m2"] / expected - 1
    return df


def aggregate_closure(aggregate: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    total = (aggregate.groupby(["unit", "radius_m", "year"], as_index=False)["area_m2"].sum()
             .rename(columns={"area_m2": "sum_classes_m2"}))
    cov = coverage[coverage["level"] == "aggregate"].rename(columns={"key": "unit"})
    df = total.merge(
        cov[["unit", "radius_m", "year", "geodesic_area_m2", "extracted_area_m2", "coverage_frac"]],
        on=["unit", "radius_m", "year"], how="outer", indicator=True,
    )
    df["orphan"] = df.pop("_merge") != "both"
    df["sum_classes_m2"] = df["sum_classes_m2"].fillna(0.0)
    df["rel_geodesic"] = (df["sum_classes_m2"] - df["geodesic_area_m2"]) / df["geodesic_area_m2"]
    df["rel_ngon"] = np.nan
    return df


def closure_violations(df: pd.DataFrame, tol: float = TOL_GEODESIC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(explained, unexplained) closure violations.

    A violation is explained when its size does not exceed what the missing
    raster coverage can account for: |rel| ≤ (1 − coverage_frac) + tol.
    """
    viol = df[df["rel_geodesic"].abs() > tol]
    limit = (1 - viol["coverage_frac"].fillna(0)) + tol
    return viol[viol["rel_geodesic"].abs() <= limit], viol[viol["rel_geodesic"].abs() > limit]


def national_vs_biomes(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Σ biome areas vs NATIONAL area per (radius, year). Equal by construction."""
    g = aggregate.assign(group=np.where(aggregate["unit"].eq(NATIONAL), "national", "biomes"))
    p = (g.groupby(["radius_m", "year", "group"])["area_m2"].sum()
         .unstack("group").reset_index())
    p["rel_diff"] = (p["biomes"] - p["national"]) / p["national"]
    return p


def class0_diagnostic(sites: pd.DataFrame) -> pd.DataFrame:
    """Area and share of class 0 (no MapBiomas data) per site, where > 0."""
    total = sites.groupby(["id_bem", "radius_m", "year"])["area_m2"].sum().rename("total_area_m2")
    zero = (sites[sites["mapbiomas_class"] == 0]
            .groupby(["id_bem", "radius_m", "year"])["area_m2"].sum().rename("class0_area_m2"))
    df = pd.concat([zero, total], axis=1, join="inner").reset_index()
    df = df[df["class0_area_m2"] > 0]
    df["class0_frac"] = df["class0_area_m2"] / df["total_area_m2"]
    return df.sort_values(["radius_m", "year", "class0_frac"], ascending=[True, True, False],
                          ignore_index=True)


def class0_review_table(diagnostic: pd.DataFrame, sites: pd.DataFrame, radius_m: int) -> pd.DataFrame:
    """Per-site class0 summary at one radius, for manual GPKG review.

    `sites` is the output of `iphan.load_sitios_iphan` (has `id_bem`, `lon`,
    `lat`). Sorted by `frac_max` descending so the worst cases come first.
    """
    d = diagnostic[diagnostic["radius_m"] == radius_m]
    g = d.groupby("id_bem")["class0_frac"].agg(n_years_present="size", frac_mean="mean",
                                                frac_max="max", frac_min="min").reset_index()
    g = g.merge(sites[["id_bem", "lon", "lat"]], on="id_bem", how="left")
    return g.sort_values("frac_max", ascending=False, ignore_index=True)
