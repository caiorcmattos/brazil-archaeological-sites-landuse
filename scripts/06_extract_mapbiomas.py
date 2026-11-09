"""Area by MapBiomas C11 class inside the buffers (exactextract), per (radius, year).

The unit of work is the (radius, year) pair: each task runs exactextract once over
every buffer of that radius and year, so the tasks are independent and can run in
parallel (e.g. as a job array, one task per `--task-id`). Method in
sitios_arq.extraction.

--raster-dir/--pixel-area/--out-dir/--buffer-dir default to sitios_arq.paths
(SITIOS_DATA_ROOT / SITIOS_RASTER_DIR) but always take priority when passed
explicitly.

Subcommands:
    python scripts/06_extract_mapbiomas.py check-grid
    python scripts/06_extract_mapbiomas.py manifest
    python scripts/06_extract_mapbiomas.py run --task-id 0
    python scripts/06_extract_mapbiomas.py run --radius 100 --year 1985

Each task writes three tables, one file per (radius, year):
    <out-dir>/sites/sites_r100_y1985.parquet          (id_bem, radius_m, year, mapbiomas_class, area_m2)
    <out-dir>/aggregate/aggregate_r100_y1985.parquet  (unit = NATIONAL or biome)
    <out-dir>/coverage/coverage_r100_y1985.parquet    geodesic vs extracted area per feature
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sitios_arq import extraction as E
from sitios_arq.buffers import geodesic_areas
from sitios_arq.paths import BUFFER_DIR
from sitios_arq.paths import EXTRACTION_C11_DIR as OUT_DIR
from sitios_arq.paths import PIXEL_AREA_TIF as PIXEL_AREA
from sitios_arq.paths import RASTER_DIR
from sitios_arq.paths import REF_RASTER_C11_1985 as REF_RASTER
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/extraction.toml"

RASTER_PATTERN = re.compile(r"^brazil_coverage-col11_(\d{4})\.tif$")
RASTER_NAME = "brazil_coverage-col11_{year}.tif"
BATCH = 5000

log = logging.getLogger("extraction")


def raster_path(raster_dir: Path, year: int) -> Path:
    return raster_dir / RASTER_NAME.format(year=year)


def check_pair(raster_dir: Path, pixel_area: Path, year: int) -> None:
    """Per-task guard: the year's raster must match the pixel-area grid."""
    cls = raster_path(raster_dir, year)
    for p in (cls, pixel_area):
        if not p.exists():
            sys.exit(f"ERROR: raster not found: {p}")
    h_cls, h_area = E.read_header(cls), E.read_header(pixel_area)
    problems = E.compare_grid(h_cls, h_area) + E.validate_class_raster(h_cls)
    if problems:
        for p in problems:
            log.error("  %s", p)
        sys.exit(f"ERROR: {cls.name} is incompatible with {pixel_area.name}")
    log.info("Grid OK: %s ≡ %s (%d×%d, EPSG:%s)", cls.name, pixel_area.name,
             h_cls["width"], h_cls["height"], h_cls["epsg"])


# ---------------------------------------------------------------------------
# check-grid / manifest
# ---------------------------------------------------------------------------

def cmd_check_grid(args) -> None:
    cfg = E.load_config(CONFIG)
    raster_dir, pixel_area, out_dir = Path(args.raster_dir), Path(args.pixel_area), Path(args.out_dir)
    if not raster_dir.is_dir():
        sys.exit(f"ERROR: raster directory not found: {raster_dir}")
    years = E.series_years(cfg)

    ignored = sorted(p.name for p in raster_dir.glob("*.tif") if not RASTER_PATTERN.match(p.name))
    for name in ignored:
        log.info("Ignored (does not match %s): %s", RASTER_PATTERN.pattern, name)
    missing = [y for y in years if not raster_path(raster_dir, y).exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} series rasters missing: {missing}")
    if not pixel_area.exists():
        sys.exit(f"ERROR: pixel-area raster not found: {pixel_area}")

    ref = E.read_header(REF_RASTER if REF_RASTER.exists() else raster_path(raster_dir, years[0]))
    rows, failures = [], 0
    targets = [(raster_path(raster_dir, y), y, True) for y in years] + [(pixel_area, None, False)]
    for path, year, is_class in targets:
        h = E.read_header(path)
        problems = E.compare_grid(h, ref) + (E.validate_class_raster(h) if is_class else [])
        rows.append({"file": h["file"], "year": year, "width": h["width"], "height": h["height"],
                     "epsg": h["epsg"], "dtype": h["dtype"], "nodata": h["nodata"],
                     "block": f"{h['block'][0]}x{h['block'][1]}",
                     "compatible": not problems, "problems": "; ".join(problems)})
        if problems:
            failures += 1
            log.error("FAILED %s: %s", h["file"], "; ".join(problems))

    save(pd.DataFrame(rows), out_dir / "grid_check.csv")
    if failures:
        sys.exit(f"ERROR: {failures} incompatible raster(s). Extraction aborted.")
    log.info("Grid consistent (%d rasters + pixel area). OK to run the tasks.", len(years))


def cmd_manifest(args) -> None:
    cfg = E.load_config(CONFIG)
    tasks = E.build_tasks(cfg)
    df = pd.DataFrame([{"task_id": i, "radius_m": r, "year": y} for i, (r, y) in enumerate(tasks)])
    out = Path(args.out_dir) / "manifest_tasks.csv"
    save(df, out)
    log.info("Manifest with %d tasks → %s", len(df), out)
    log.info("Run each task with: run --task-id N, N = 0..%d", len(df) - 1)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_sites(buffer_dir: Path, radius: int, limit: int | None) -> gpd.GeoDataFrame:
    path = buffer_dir / f"buffers_{radius}m.parquet"
    gdf = gpd.read_parquet(path)[["id_bem", "geometry"]]
    if limit:
        gdf = gdf.head(limit)
        log.warning("TEST MODE: limited to %d sites (not for production)", limit)
    log.info("Individual buffers: %s (%d features)", path.name, len(gdf))
    return E.prepare(gdf)


def load_aggregates(buffer_dir: Path, radius: int) -> gpd.GeoDataFrame:
    """National and per-biome dissolves, exploded into disjoint parts.

    The national dissolve covers all of Brazil; asking exactextract for it at
    once would need a raster-sized window. Exploded, the summed result is
    identical and memory is trivial.
    """
    parts = []
    for name in (f"buffers_{radius}m_dissolved.parquet", f"buffers_{radius}m_dissolved_biome.parquet"):
        gdf = gpd.read_parquet(buffer_dir / name)[["unit", "geometry"]].explode(index_parts=False)
        log.info("Dissolved %s: %d parts", name, len(gdf))
        parts.append(gdf)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs, geometry="geometry")
    return E.prepare(gdf)


def coverage_table(gdf: gpd.GeoDataFrame, totals: pd.DataFrame, key: str) -> pd.DataFrame:
    ref = pd.DataFrame({"_fid": gdf["_fid"].to_numpy(), "key": gdf[key].astype(str).to_numpy(),
                        "geodesic_area_m2": geodesic_areas(gdf.geometry)})
    return ref.merge(totals, on="_fid", how="left")


def cmd_run(args) -> None:
    cfg = E.load_config(CONFIG)
    tasks = E.build_tasks(cfg)
    if args.task_id is not None:
        if not 0 <= args.task_id < len(tasks):
            sys.exit(f"ERROR: task-id {args.task_id} outside 0..{len(tasks) - 1}")
        radius, year = tasks[args.task_id]
    elif args.radius is not None and args.year is not None:
        radius, year = args.radius, args.year
    else:
        sys.exit("ERROR: give --task-id OR --radius and --year")

    raster_dir, pixel_area = Path(args.raster_dir), Path(args.pixel_area)
    buffer_dir, out_dir = Path(args.buffer_dir), Path(args.out_dir)
    log.info("=" * 68)
    log.info("Task %s | radius %d m | year %d", args.task_id, radius, year)
    log.info("=" * 68)

    check_pair(raster_dir, pixel_area, year)
    cls_path = raster_path(raster_dir, year)
    t0 = time.time()
    tag = f"r{radius}_y{year}"
    log.info("── Site level ──")
    sites = load_sites(buffer_dir, radius, args.limit_sites)
    cls_sites, tot_sites = E.extract(cls_path, pixel_area, sites, args.batch)
    tab_sites = (
        cls_sites.merge(sites[["_fid", "id_bem"]], on="_fid")
        .assign(radius_m=np.int16(radius), year=np.int16(year))
        [["id_bem", "radius_m", "year", "mapbiomas_class", "area_m2"]]
        .sort_values(["id_bem", "mapbiomas_class"], ignore_index=True)
    )
    cov_sites = coverage_table(sites, tot_sites, "id_bem").assign(level="site")
    save(tab_sites, out_dir / "sites" / f"sites_{tag}.parquet", index=False)
    log.info("Saved site level: %d rows, %d sites, %d classes", len(tab_sites),
             tab_sites["id_bem"].nunique(), tab_sites["mapbiomas_class"].nunique())

    log.info("── Aggregate level (national + biomes) ──")
    agg = load_aggregates(buffer_dir, radius)
    cls_agg, tot_agg = E.extract(cls_path, pixel_area, agg, args.batch)
    tab_agg = (
        cls_agg.merge(agg[["_fid", "unit"]], on="_fid")
        .groupby(["unit", "mapbiomas_class"], as_index=False)["area_m2"].sum()
        .assign(radius_m=np.int16(radius), year=np.int16(year))
        [["unit", "radius_m", "year", "mapbiomas_class", "area_m2"]]
        .sort_values(["unit", "mapbiomas_class"], ignore_index=True)
    )
    cov_agg = (coverage_table(agg, tot_agg, "unit")
               .groupby("key", as_index=False)[["geodesic_area_m2", "extracted_area_m2"]].sum()
               .assign(level="aggregate"))
    save(tab_agg, out_dir / "aggregate" / f"aggregate_{tag}.parquet", index=False)
    log.info("Saved aggregate level: %d rows, %d units", len(tab_agg), tab_agg["unit"].nunique())

    cols = ["level", "key", "geodesic_area_m2", "extracted_area_m2"]
    cov = pd.concat([cov_sites[cols], cov_agg[cols]], ignore_index=True)
    cov["coverage_frac"] = np.where(cov["geodesic_area_m2"] > 0,
                                    cov["extracted_area_m2"] / cov["geodesic_area_m2"], np.nan)
    cov = cov.assign(radius_m=np.int16(radius), year=np.int16(year))[
        ["level", "key", "radius_m", "year", "geodesic_area_m2", "extracted_area_m2", "coverage_frac"]]
    save(cov, out_dir / "coverage" / f"coverage_{tag}.parquet", index=False)
    partial = cov[cov["coverage_frac"] < 0.999]
    log.info("Coverage: %d features | %d partial (<0.999) | %d none", len(cov), len(partial),
             int((cov["coverage_frac"].fillna(0) == 0).sum()))
    log.info("Task done in %.1f s", time.time() - t0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--raster-dir", default=str(RASTER_DIR))
    common.add_argument("--pixel-area", default=str(PIXEL_AREA))
    common.add_argument("--out-dir", default=str(OUT_DIR))

    sub.add_parser("check-grid", parents=[common], help="Check the grid of the series rasters") \
        .set_defaults(func=cmd_check_grid)
    sub.add_parser("manifest", parents=[common], help="Write the task_id → (radius, year) map") \
        .set_defaults(func=cmd_manifest)
    pr = sub.add_parser("run", parents=[common], help="Run one (radius, year) pair")
    pr.add_argument("--task-id", type=int)
    pr.add_argument("--radius", type=int)
    pr.add_argument("--year", type=int)
    pr.add_argument("--buffer-dir", default=str(BUFFER_DIR))
    pr.add_argument("--batch", type=int, default=BATCH)
    pr.add_argument("--limit-sites", type=int, help="local smoke test only — limits the number of sites")
    pr.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
