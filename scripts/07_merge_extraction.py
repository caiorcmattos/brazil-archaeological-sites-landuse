"""Merge and sanity-check the partial outputs of the C11 extraction array.

    python scripts/07_merge_extraction.py
    python scripts/07_merge_extraction.py --in-dir /tmp/smoke --out-dir /tmp/smoke --skip-completeness

Exits with code 1 if a task is missing or a tolerance is violated.

Checks
------
- completeness: the three files of every task in the manifest;
- number of sites per (radius, year) equals the buffer base;
- closure: Σ class areas vs geodesic area of each feature (tolerance 1e-6);
- raster coverage: every feature fully covered by the raster (coverage_frac
  ≥ 0.999). Partial coverage fails unless --allow-partial-coverage, because a
  closure violation it explains can also hide a broken geometry (P-026);
  with the flag, violations it explains are reported, not failed;
- n-gon: geodesic area of each site buffer vs πr² × area factor of the
  regular n-gon (tolerance 1e-6). Buffers are geodesic, so this must hold
  everywhere;
- NATIONAL vs Σ biomes: equal by construction (biome clip), tolerance 1e-6.

Class 0 (no MapBiomas data) is reported in class0_diagnostic.csv only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from glob import glob
from pathlib import Path

import pandas as pd

from sitios_arq import extraction as E
from sitios_arq.paths import BUFFER_DIR
from sitios_arq.paths import EXTRACTION_C11_DIR as IN_DIR
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/extraction.toml"

log = logging.getLogger("merge")


def read_dir(folder: Path) -> pd.DataFrame:
    files = sorted(glob(str(folder / "*.parquet")))
    if not files:
        sys.exit(f"ERROR: no .parquet in {folder}")
    return pd.read_parquet(files)


def missing_tasks(in_dir: Path, tasks: list[tuple[int, int]]) -> list[int]:
    missing = []
    for tid, (r, y) in enumerate(tasks):
        tag = f"r{r}_y{y}"
        absent = [p.name for p in (in_dir / "sites" / f"sites_{tag}.parquet",
                                   in_dir / "aggregate" / f"aggregate_{tag}.parquet",
                                   in_dir / "coverage" / f"coverage_{tag}.parquet") if not p.exists()]
        if absent:
            missing.append(tid)
            log.error("task %d (radius %d, year %d): missing %s", tid, r, y, ", ".join(absent))
    return missing


def report(df: pd.DataFrame, level: str, rows: list[dict], check_ngon: bool,
           allow_partial: bool = False) -> int:
    """Log and collect the closure report. Returns the number of failures."""
    explained, unexplained = E.closure_violations(df)
    partial = df[df["coverage_frac"] < 0.999]
    none = df[df["coverage_frac"].fillna(0) == 0]
    orphans = int(df["orphan"].sum())
    ngon_bad = df[df["rel_ngon"].abs() > E.TOL_GEODESIC] if check_ngon else df.iloc[:0]

    log.info("── Closure %s ──", level)
    log.info("  features: %d | orphans: %d", len(df), orphans)
    log.info("  max |rel| vs geodesic area: %.3e (tolerance %.0e)",
             df["rel_geodesic"].abs().max(), E.TOL_GEODESIC)
    log.info("  violations: %d explained by partial coverage, %d unexplained",
             len(explained), len(unexplained))
    log.info("  partial coverage (<0.999): %d | no coverage: %d", len(partial), len(none))
    if check_ngon:
        log.info("  n-gon: max |geodesic/(πr²·factor) − 1| = %.3e; violations: %d",
                 df["rel_ngon"].abs().max(), len(ngon_bad))
    for _, r in unexplained.reindex(
            unexplained["rel_geodesic"].abs().sort_values(ascending=False).index).head(10).iterrows():
        log.error("    radius %s year %s → rel=%.3e coverage_frac=%.6f",
                  r["radius_m"], r["year"], r["rel_geodesic"], r["coverage_frac"])

    rows.append({
        "level": level, "n_features": len(df), "n_orphans": orphans,
        "max_abs_rel_geodesic": df["rel_geodesic"].abs().max(),
        "n_violations_explained": len(explained), "n_violations_unexplained": len(unexplained),
        "n_partial_coverage": len(partial), "n_no_coverage": len(none),
        "max_abs_rel_ngon": df["rel_ngon"].abs().max() if check_ngon else float("nan"),
        "n_violations_ngon": len(ngon_bad),
    })
    if len(partial) and not allow_partial:
        log.error("  %d %s features with partial raster coverage (use --allow-partial-coverage "
                  "only if they really extend beyond the raster)", len(partial), level)
    return len(unexplained) + orphans + len(ngon_bad) + (0 if allow_partial else len(partial))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default=str(IN_DIR))
    p.add_argument("--out-dir", default=str(IN_DIR))
    p.add_argument("--buffer-dir", default=str(BUFFER_DIR))
    p.add_argument("--skip-completeness", action="store_true",
                   help="do not require every task nor the full site count (partial smoke test)")
    p.add_argument("--allow-partial-coverage", action="store_true",
                   help="accept features partly outside the raster (closure checked against coverage)")
    args = p.parse_args()

    cfg = E.load_config(CONFIG)
    tasks = E.build_tasks(cfg)
    n_vertices = cfg["n_vertices"]["value"]
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    problems = 0

    n_expected = len(pd.read_parquet(Path(args.buffer_dir) / "buffers_100m.parquet", columns=["id_bem"]))
    log.info("Site base: %d (buffers_100m.parquet)", n_expected)

    if args.skip_completeness:
        log.warning("Completeness not checked (--skip-completeness)")
    else:
        missing = missing_tasks(in_dir, tasks)
        if missing:
            log.error("Missing %d tasks. Resubmit with:", len(missing))
            log.error("    task ids: %s (scripts/06_extract_mapbiomas.py run --task-id)", E.format_array(missing))
            problems += len(missing)
        else:
            log.info("Completeness: all %d tasks present at the 3 levels", len(tasks))

    sites = read_dir(in_dir / "sites")
    aggregate = read_dir(in_dir / "aggregate")
    coverage = read_dir(in_dir / "coverage")
    log.info("Read: sites=%d rows | aggregate=%d | coverage=%d", len(sites), len(aggregate), len(coverage))

    if not args.skip_completeness:
        count = sites.groupby(["radius_m", "year"])["id_bem"].nunique()
        bad = count[count != n_expected]
        for (r, y), n in bad.items():
            log.error("    radius %d year %d → %d sites (expected %d)", r, y, n, n_expected)
        problems += len(bad)
        if bad.empty:
            log.info("Site count: %d in all %d combinations", n_expected, len(count))

    rows: list[dict] = []
    allow = args.allow_partial_coverage
    problems += report(E.site_closure(sites, coverage, n_vertices), "site", rows, check_ngon=True,
                       allow_partial=allow)
    problems += report(E.aggregate_closure(aggregate, coverage), "aggregate", rows, check_ngon=False,
                       allow_partial=allow)

    nb = E.national_vs_biomes(aggregate)
    n_bad = int((nb["rel_diff"].abs() > E.TOL_GEODESIC).sum())
    log.info("── NATIONAL vs Σ biomes ──")
    log.info("  max |rel diff| = %.3e over %d (radius, year); violations: %d",
             nb["rel_diff"].abs().max(), len(nb), n_bad)
    problems += n_bad
    rows.append({"level": "national_vs_biomes", "n_features": len(nb),
                 "max_abs_rel_geodesic": nb["rel_diff"].abs().max(),
                 "n_violations_unexplained": n_bad})

    c0 = E.class0_diagnostic(sites)
    log.info("── Class 0 (diagnostic only) ──")
    for (r, y), g in c0.groupby(["radius_m", "year"]):
        log.info("  radius %d year %d: %d sites with class 0 (max frac %.3f, %d with frac > 0.5)",
                 r, y, len(g), g["class0_frac"].max(), int((g["class0_frac"] > 0.5).sum()))

    common = dict(index=False)
    for df, name, desc in (
        (sites, "extraction_sites_c11.parquet", "Area (m²) by MapBiomas C11 class per site, all radii and years"),
        (aggregate, "extraction_aggregate_c11.parquet", "Area (m²) by MapBiomas C11 class, national and per biome"),
        (coverage, "extraction_coverage_c11.parquet", "Raster coverage per feature, all radii and years"),
    ):
        save(df, out_dir / name, **common)
        log.info("Merged: %s (%d rows)", name, len(df))

    save(c0, out_dir / "class0_diagnostic.csv")
    save(pd.DataFrame(rows), out_dir / "extraction_sanity.csv")

    if problems:
        sys.exit(f"FAILED: {problems} problem(s). See above.")
    log.info("Sanity OK — no problems.")


if __name__ == "__main__":
    main()
