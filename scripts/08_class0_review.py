"""Per-site summary of class 0 (no MapBiomas data) for manual GPKG review.

    python scripts/08_class0_review.py
    python scripts/08_class0_review.py --radius 500 --in-dir /path/to/class0_diagnostic_dir

`class0_diagnostic.csv` (from `07_merge_extraction.py`) is diagnostic only —
no site is excluded because of it. This script builds
a candidate list and a GeoPackage for manual inspection in QGIS; it never
writes to `config/filtering_decisions.csv` (hand-edited, manual decisions
only).

Outputs (in --out-dir, default = --in-dir):
- class0_review_r{radius}.csv   id_bem, biome, n_years_present, frac_mean/max/min, lon, lat
- class0_review_r{radius}.gpkg  same, as points at raw coordinates (EPSG:4674)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sitios_arq import extraction as E
from sitios_arq.iphan import CRS_IPHAN, SHP_SITIOS_2026_09_08 as SHP_SITES, load_sitios_iphan
from sitios_arq.paths import BUFFER_DIR
from sitios_arq.paths import EXTRACTION_C11_DIR as IN_DIR
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/extraction.toml"

log = logging.getLogger("class0_review")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = E.load_config(CONFIG)
    radii = cfg["radii_m"]["value"]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radius", type=int, default=cfg["main_radius_m"]["value"], choices=radii,
                   help="buffer radius to review (default: main time series radius)")
    p.add_argument("--in-dir", default=str(IN_DIR), help="directory holding class0_diagnostic.csv")
    p.add_argument("--out-dir", default=None, help="default: same as --in-dir")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    diagnostic_path = in_dir / "class0_diagnostic.csv"
    diagnostic = pd.read_csv(diagnostic_path)
    log.info("Read: %s (%d rows)", diagnostic_path, len(diagnostic))

    sites = load_sitios_iphan(SHP_SITES)
    buffers = pd.read_parquet(BUFFER_DIR / f"buffers_{args.radius}m.parquet", columns=["id_bem", "biome"])

    review = E.class0_review_table(diagnostic, sites, radius_m=args.radius)
    review = review.merge(buffers, on="id_bem", how="left")

    log.info("── Class 0 review (radius %d m) ──", args.radius)
    log.info("  sites with class0 presence: %d", len(review))
    for lo, hi in [(0.999, 1.0 + 1e-9), (0.5, 0.999), (0.1, 0.5), (0.01, 0.1), (0.0, 0.01)]:
        n = int(review["frac_max"].between(lo, hi, inclusive="left" if lo > 0 else "both").sum())
        log.info("  frac_max in [%.3f, %.3f): %d sites", lo, hi, n)

    review_points = gpd.GeoDataFrame(
        review, geometry=gpd.points_from_xy(review["lon"], review["lat"]), crs=CRS_IPHAN
    )

    save(review, out_dir / f"class0_review_r{args.radius}.csv")
    save(review_points, out_dir / f"class0_review_r{args.radius}.gpkg", layer="class0_review")
    log.info("Saved: class0_review_r%d.csv, class0_review_r%d.gpkg (in %s)", args.radius, args.radius, out_dir)


if __name__ == "__main__":
    main()
