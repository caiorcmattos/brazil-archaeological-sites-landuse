"""Thematic aggregation of the C11 extraction (Natural / Anthropic / Water / No data).

    python scripts/09_thematic_aggregation.py
    python scripts/09_thematic_aggregation.py --radius 100 --year 2020 --out-dir /tmp/smoke

Reads the merged products of `07_merge_extraction.py` (never modifies them)
and the legend `config/mapbiomas_legend_c11.csv`. Exits with code 1 if a
check fails; raises if a class code has no category in the legend.

Class 0 rule: class 0 belongs to no category and is left
out of the denominator. Proportion = category area / (total − class 0 area).

Outputs (in --out-dir)
----------------------
- thematic_sites_c11.parquet       per id_bem × radius × year: category areas,
  total, area with data, proportions; `nodata_any_year` flags sites with
  class 0 in at least one year at that radius (with/without analyses)
- thematic_aggregate_c11.parquet   same per unit (NATIONAL and biomes)
- thematic_class0_summary.csv      per radius × year: sites with class 0 and
  the distribution of their class 0 share
- thematic_sanity.csv              check results

Checks
------
- closure: Σ four category areas vs total area (tolerance 1e-6, as in 07);
- class 0 area is 0 for every site at 100 m;
- NATIONAL vs Σ biomes per category (tolerance 1e-6).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from sitios_arq import extraction as E
from sitios_arq import thematic as T
from sitios_arq.paths import EXTRACTION_C11_DIR as IN_DIR
from sitios_arq.paths import THEMATIC_C11_DIR as OUT_DIR
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
LEGEND = ROOT / "config/mapbiomas_legend_c11.csv"
MAIN_RADIUS = 100

log = logging.getLogger("thematic")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default=str(IN_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--radius", type=int, help="only this radius (smoke test)")
    p.add_argument("--year", type=int, help="only this year (smoke test)")
    args = p.parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)

    legend = T.load_legend(LEGEND)
    log.info("Legend: %d codes (%s)", len(legend), LEGEND.name)

    filters = [(c, "==", v) for c, v in (("radius_m", args.radius), ("year", args.year)) if v is not None]
    inputs = [in_dir / "extraction_sites_c11.parquet", in_dir / "extraction_aggregate_c11.parquet"]
    sites = pd.read_parquet(inputs[0], filters=filters or None)
    aggregate = pd.read_parquet(inputs[1], filters=filters or None)
    log.info("Read: sites=%d rows | aggregate=%d rows%s", len(sites), len(aggregate),
             f" (filters {filters})" if filters else "")

    ts = T.aggregate_categories(sites, legend, ["id_bem", "radius_m", "year"])
    ta = T.aggregate_categories(aggregate, legend, ["unit", "radius_m", "year"])
    ts["nodata_any_year"] = ts.groupby(["radius_m", "id_bem"])["area_nodata_m2"].transform("max") > 0

    problems = 0
    rows: list[dict] = []
    for name, df in (("site", ts), ("aggregate", ta)):
        rel = T.closure_rel(df).abs()
        n_bad = int((rel > E.TOL_GEODESIC).sum())
        log.info("── Closure (%s) ── max |rel| = %.3e over %d rows; violations: %d",
                 name, rel.max(), len(df), n_bad)
        rows.append({"check": f"closure_{name}", "n": len(df), "max_abs_rel": rel.max(), "n_violations": n_bad})
        problems += n_bad

    main_r = ts[ts["radius_m"] == MAIN_RADIUS]
    n_bad = int((main_r["area_nodata_m2"] != 0).sum())
    log.info("── Class 0 at %d m ── %d site-years, %d with class 0 > 0", MAIN_RADIUS, len(main_r), n_bad)
    rows.append({"check": f"nodata_zero_r{MAIN_RADIUS}", "n": len(main_r), "max_abs_rel": float("nan"),
                 "n_violations": n_bad})
    problems += n_bad

    nb = T.national_vs_biomes(ta)
    n_bad = int((nb["rel_diff"] > E.TOL_GEODESIC).sum())
    log.info("── NATIONAL vs Σ biomes ── max rel diff = %.3e over %d (radius, year, category); violations: %d",
             nb["rel_diff"].max(), len(nb), n_bad)
    rows.append({"check": "national_vs_biomes", "n": len(nb), "max_abs_rel": nb["rel_diff"].max(),
                 "n_violations": n_bad})
    problems += n_bad

    c0 = T.class0_site_summary(ts)
    log.info("── Sites with class 0 (left out of the denominator) ──")
    for _, r in c0.iterrows():
        log.info("  radius %4d year %d: %4d sites (%.2f%%), max share %.3f, %d with share > 0.1; "
                 "any year at this radius: %d",
                 r["radius_m"], r["year"], r["n_sites_nodata"], 100 * r["share_sites_nodata"],
                 r["nodata_frac_max"] if r["n_sites_nodata"] else 0.0, r["n_sites_nodata_gt_0.1"],
                 r["n_sites_any_year"])

    save(ts, out_dir / "thematic_sites_c11.parquet", index=False)
    save(ta, out_dir / "thematic_aggregate_c11.parquet", index=False)
    save(c0, out_dir / "thematic_class0_summary.csv")
    save(pd.DataFrame(rows), out_dir / "thematic_sanity.csv")
    log.info("Saved in %s", out_dir)

    if problems:
        sys.exit(f"FAILED: {problems} problem(s). See above.")
    log.info("Checks OK — no problems.")


if __name__ == "__main__":
    main()
