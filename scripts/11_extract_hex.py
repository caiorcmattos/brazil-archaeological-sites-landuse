"""Area by MapBiomas C11 class inside the hexagon pieces of the national dissolve.

One independent task per year (they can run in parallel, e.g. as a job array).
The units are the pieces written by 10_hex_grid.py (`hex_units_r{r}.parquet`:
national dissolve of the buffers clipped by hexagon), so every square metre of
buffer is extracted once and the cells add up to NATIONAL. Same method as
06_extract_mapbiomas.py (sitios_arq.extraction: weighted_frac × Σ pixel area).

Years = the whole main series of config/extraction.toml (series_years, 1985–2025),
so any period can be computed later without re-running; the endpoints of the
periods in config/hex.toml must be inside it.

--raster-dir/--pixel-area/--out-dir/--hex-dir default to sitios_arq.paths
(SITIOS_DATA_ROOT / SITIOS_RASTER_DIR) but always take priority when passed.

Subcommands:
    python scripts/11_extract_hex.py manifest
    python scripts/11_extract_hex.py run --task-id 0
    python scripts/11_extract_hex.py run --year 1985
    python scripts/11_extract_hex.py merge

Outputs (in --out-dir, default processed/extraction_hex_c11/):
    hex/hex_y1985.parquet            (unit, radius_m, year, mapbiomas_class, area_m2); unit = hex_id
    coverage/coverage_y1985.parquet  geodesic vs extracted area per unit
    manifest_tasks.csv               task_id → year
    extraction_hex_c11.parquet       merged (merge)
    extraction_hex_sanity.csv        check results (merge)

Checks (merge; exits with code 1 if any fails):
    - completeness: both files of every task in the manifest;
    - same units in every year, equal to hex_units_r{r}.parquet;
    - closure: Σ class areas vs geodesic area of each unit (1e-6);
    - raster coverage: every unit fully covered (coverage_frac ≥ 0.999); partial
      coverage fails unless --allow-partial-coverage, which reports the closure
      violations it explains instead of failing (P-026);
    - Σ cells vs NATIONAL of the main extraction (extraction_aggregate_c11.parquet),
      per year and class: |difference| / national area ≤ 1e-6.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sitios_arq import extraction as E
from sitios_arq import hexgrid as H
from sitios_arq.buffers import geodesic_areas
from sitios_arq.paths import EXTRACTION_C11_DIR, EXTRACTION_HEX_C11_DIR, HEX_DIR
from sitios_arq.paths import PIXEL_AREA_TIF as PIXEL_AREA
from sitios_arq.paths import RASTER_DIR
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/hex.toml"
EXTRACTION_CONFIG = ROOT / "config/extraction.toml"
RASTER_NAME = "brazil_coverage-col11_{year}.tif"
BATCH = 5000
TOL = 1e-6

log = logging.getLogger("extraction_hex")


def raster_path(raster_dir: Path, year: int) -> Path:
    return raster_dir / RASTER_NAME.format(year=year)


def tasks_from_config() -> tuple[int, list[int]]:
    """(radius, years): every year of the main series; task_id = index in `years`."""
    cfg = H.load_config(CONFIG)
    years = E.series_years(E.load_config(EXTRACTION_CONFIG))
    outside = sorted(set(H.period_years(cfg)) - set(years))
    if outside:
        raise ValueError(f"Period years outside series_years: {outside}")
    return cfg["radius_m"]["value"], years


def units_path(hex_dir: Path, radius: int) -> Path:
    return hex_dir / f"hex_units_r{radius}.parquet"


def check_pair(raster_dir: Path, pixel_area: Path, year: int) -> None:
    """The year's raster must exist and match the pixel-area grid."""
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
    log.info("Grid OK: %s ≡ %s", cls.name, pixel_area.name)


# ---------------------------------------------------------------------------
# manifest / run
# ---------------------------------------------------------------------------

def cmd_manifest(args) -> None:
    radius, years = tasks_from_config()
    df = pd.DataFrame({"task_id": range(len(years)), "radius_m": radius, "year": years})
    out = Path(args.out_dir) / "manifest_tasks.csv"
    save(df, out)
    log.info("Manifest with %d tasks (years %d–%d) → %s", len(df), years[0], years[-1], out)
    log.info("Run each task with: run --task-id N, N = 0..%d", len(df) - 1)


def cmd_run(args) -> None:
    radius, years = tasks_from_config()
    if args.task_id is not None:
        if not 0 <= args.task_id < len(years):
            sys.exit(f"ERROR: task-id {args.task_id} outside 0..{len(years) - 1}")
        year = years[args.task_id]
    elif args.year is not None:
        year = args.year
    else:
        sys.exit("ERROR: give --task-id OR --year")

    raster_dir, pixel_area = Path(args.raster_dir), Path(args.pixel_area)
    out_dir, upath = Path(args.out_dir), units_path(Path(args.hex_dir), radius)
    log.info("=" * 68)
    log.info("Task %s | radius %d m | year %d", args.task_id, radius, year)
    log.info("=" * 68)
    check_pair(raster_dir, pixel_area, year)
    cls_path = raster_path(raster_dir, year)
    t0 = time.time()

    # Pieces exploded into disjoint parts: a whole cell would need a ~100 km window
    # per feature; the summed result is identical (as in 06 for the dissolves).
    units = gpd.read_parquet(upath)[["unit", "geometry"]]
    parts = E.prepare(units.explode(index_parts=False).reset_index(drop=True))
    log.info("Units: %s (%d cells, %d parts)", upath.name, len(units), len(parts))

    cls, tot = E.extract(cls_path, pixel_area, parts, args.batch)
    tab = (
        cls.merge(parts[["_fid", "unit"]], on="_fid")
        .groupby(["unit", "mapbiomas_class"], as_index=False)["area_m2"].sum()
        .assign(radius_m=np.int16(radius), year=np.int16(year))
        [["unit", "radius_m", "year", "mapbiomas_class", "area_m2"]]
        .sort_values(["unit", "mapbiomas_class"], ignore_index=True)
    )
    cov = (pd.DataFrame({"_fid": parts["_fid"].to_numpy(), "unit": parts["unit"].to_numpy(),
                         "geodesic_area_m2": geodesic_areas(parts.geometry)})
           .merge(tot, on="_fid", how="left")
           .groupby("unit", as_index=False)[["geodesic_area_m2", "extracted_area_m2"]].sum())
    cov["coverage_frac"] = cov["extracted_area_m2"] / cov["geodesic_area_m2"]
    cov = cov.assign(radius_m=np.int16(radius), year=np.int16(year))

    tag = f"y{year}"
    save(tab, out_dir / "hex" / f"hex_{tag}.parquet", index=False)
    save(cov, out_dir / "coverage" / f"coverage_{tag}.parquet", index=False)
    partial = cov[cov["coverage_frac"] < 0.999]
    log.info("Saved: %d rows, %d cells, %d classes | partial coverage: %d cells",
             len(tab), tab["unit"].nunique(), tab["mapbiomas_class"].nunique(), len(partial))
    log.info("Task done in %.1f s", time.time() - t0)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def read_dir(folder: Path) -> pd.DataFrame:
    files = sorted(glob(str(folder / "*.parquet")))
    if not files:
        sys.exit(f"ERROR: no .parquet in {folder}")
    return pd.read_parquet(files)


def cmd_merge(args) -> None:
    radius, years = tasks_from_config()
    in_dir = out_dir = Path(args.out_dir)
    upath = units_path(Path(args.hex_dir), radius)
    nat_path = Path(args.national) if args.national else EXTRACTION_C11_DIR / "extraction_aggregate_c11.parquet"
    rows, problems = [], 0

    missing = [i for i, y in enumerate(years)
               if not ((in_dir / "hex" / f"hex_y{y}.parquet").exists()
                       and (in_dir / "coverage" / f"coverage_y{y}.parquet").exists())]
    if missing and not args.skip_completeness:
        log.error("Missing tasks %s. Resubmit with:", missing)
        log.error("    task ids: %s (run --task-id)", E.format_array(missing))
        problems += len(missing)
    elif missing:
        log.warning("Completeness not enforced (--skip-completeness); missing tasks %s", missing)
    rows.append({"check": "completeness", "value": len(missing), "tolerance": 0,
                 "ok": not missing})

    hexes, cov = read_dir(in_dir / "hex"), read_dir(in_dir / "coverage")
    present = sorted(hexes["year"].unique().tolist())
    log.info("Read: %d rows, years %s", len(hexes), present)

    expected = set(pd.read_parquet(upath, columns=["unit"])["unit"])
    bad_units = [y for y in present if set(cov.loc[cov["year"] == y, "unit"]) != expected]
    if bad_units:
        log.error("Units differ from %s in years %s", upath.name, bad_units)
        problems += 1
    rows.append({"check": "units_match_hex_units", "value": len(bad_units), "tolerance": 0,
                 "ok": not bad_units})

    # Closure per unit and year
    total = (hexes.groupby(["unit", "radius_m", "year"], as_index=False)["area_m2"].sum()
             .rename(columns={"area_m2": "sum_classes_m2"}))
    clo = total.merge(cov, on=["unit", "radius_m", "year"], how="outer")
    clo["sum_classes_m2"] = clo["sum_classes_m2"].fillna(0.0)
    clo["rel_geodesic"] = (clo["sum_classes_m2"] - clo["geodesic_area_m2"]) / clo["geodesic_area_m2"]
    explained, unexplained = E.closure_violations(clo, TOL)
    max_rel = float(clo["rel_geodesic"].abs().max())
    log.info("Closure: max |rel| = %.2e | explained by coverage: %d | unexplained: %d",
             max_rel, len(explained), len(unexplained))
    problems += len(unexplained)
    n_partial = int((clo["coverage_frac"] < 0.999).sum())
    if n_partial and not args.allow_partial_coverage:
        log.error("%d unit-years with partial raster coverage (--allow-partial-coverage to accept)", n_partial)
        problems += n_partial
    rows.append({"check": "partial_coverage", "value": n_partial, "tolerance": 0,
                 "ok": n_partial == 0 or bool(args.allow_partial_coverage)})
    rows.append({"check": "closure_max_rel", "value": max_rel, "tolerance": TOL,
                 "ok": unexplained.empty})

    # Σ cells vs NATIONAL of the main extraction, per year and class
    if not nat_path.exists():
        sys.exit(f"ERROR: national aggregate not found: {nat_path}")
    nat = pd.read_parquet(nat_path, filters=[("unit", "==", E.NATIONAL), ("radius_m", "==", radius)])
    nat = nat[nat["year"].isin(present)].groupby(["year", "mapbiomas_class"])["area_m2"].sum()
    cells = hexes.groupby(["year", "mapbiomas_class"])["area_m2"].sum()
    both = pd.concat({"national": nat, "cells": cells}, axis=1).fillna(0.0)
    nat_total = both["national"].groupby(level="year").transform("sum")
    both["rel_to_total"] = (both["cells"] - both["national"]).abs() / nat_total
    max_nat = float(both["rel_to_total"].max())
    ok_nat = max_nat <= TOL and set(nat.index.get_level_values("year")) == set(present)
    log.info("Σ cells vs NATIONAL (per year × class, / national area): max %.2e", max_nat)
    if not ok_nat:
        log.error("Σ cells differs from NATIONAL beyond %.0e (or years missing in %s)", TOL, nat_path.name)
        problems += 1
    rows.append({"check": "cells_vs_national_max_rel", "value": max_nat, "tolerance": TOL, "ok": ok_nat})

    save(hexes.sort_values(["unit", "year", "mapbiomas_class"], ignore_index=True), out_dir / "extraction_hex_c11.parquet", index=False)
    save(pd.DataFrame(rows), out_dir / "extraction_hex_sanity.csv")
    if problems:
        sys.exit(f"ERROR: {problems} problem(s) in the merge")
    log.info("Merge OK")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", default=str(EXTRACTION_HEX_C11_DIR))
    common.add_argument("--hex-dir", default=str(HEX_DIR))

    sub.add_parser("manifest", parents=[common], help="Write the task_id → year map") \
        .set_defaults(func=cmd_manifest)

    pr = sub.add_parser("run", parents=[common], help="Extract one year")
    pr.add_argument("--task-id", type=int)
    pr.add_argument("--year", type=int)
    pr.add_argument("--raster-dir", default=str(RASTER_DIR))
    pr.add_argument("--pixel-area", default=str(PIXEL_AREA))
    pr.add_argument("--batch", type=int, default=BATCH)
    pr.set_defaults(func=cmd_run)

    pm = sub.add_parser("merge", parents=[common], help="Merge the years and run the checks")
    pm.add_argument("--national", help="extraction_aggregate_c11.parquet (default: paths)")
    pm.add_argument("--skip-completeness", action="store_true", help="smoke test with missing years")
    pm.add_argument("--allow-partial-coverage", action="store_true",
                    help="accept units partly outside the raster (closure checked against coverage)")
    pm.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
