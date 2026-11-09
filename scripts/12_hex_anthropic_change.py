"""Change in anthropic cover per hexagon and period (Fig 2 B–E).

    python scripts/12_hex_anthropic_change.py
    python scripts/12_hex_anthropic_change.py --extraction /tmp/smoke/extraction_hex_c11.parquet --out-dir /tmp/smoke

Reads the merged hexagon extraction (11_extract_hex.py merge), aggregates the
MapBiomas classes into Natural / Anthropic / Water / No data (config legend) and
computes, per cell and period of config/hex.toml (main + test), delta_pp and the
mean annual rate (sitios_arq.change). Site counts and the sparse flag come from
10_hex_grid.py. Exits with code 1 if a check fails.

Outputs (in --out-dir, default processed/hex/)
----------------------------------------------
- thematic_hex_c11.parquet          per cell × year: category areas, total, area with
  data, proportions (as thematic_aggregate_c11)
- hex_anthropic_change_r{r}.parquet per cell × period: hex_id, n_sites, sparse, period,
  start, end, n_years, anthropic shares, delta (m², pp), rate (pp/yr, m²/yr)

Checks
------
- closure: Σ four category areas vs total area per cell-year (1e-6);
- no class 0 area at 100 m (as for the sites);
- consecutive periods add up to the period they span, per cell;
- per period, Σ cells delta_anthropic_m2 equals the NATIONAL delta of
  thematic_aggregate_c11 (|diff| / national area ≤ 1e-6);
- every cell of hex_grid with buffer area has a row, and vice versa.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from sitios_arq import change as C
from sitios_arq import hexgrid as H
from sitios_arq import thematic as T
from sitios_arq.extraction import NATIONAL
from sitios_arq.paths import EXTRACTION_HEX_C11_DIR, HEX_DIR, THEMATIC_C11_DIR
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/hex.toml"
LEGEND = ROOT / "config/mapbiomas_legend_c11.csv"
TOL = 1e-6

log = logging.getLogger("hex_change")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extraction", default=str(EXTRACTION_HEX_C11_DIR / "extraction_hex_c11.parquet"))
    p.add_argument("--hex-dir", default=str(HEX_DIR))
    p.add_argument("--out-dir", default=str(HEX_DIR))
    args = p.parse_args()
    out_dir, hex_dir = Path(args.out_dir), Path(args.hex_dir)

    cfg = H.load_config(CONFIG)
    r, periods = cfg["radius_m"]["value"], H.periods(cfg)
    grid_path = hex_dir / f"hex_grid_{cfg['hex_width_km']['value']}km.parquet"
    nat_path = THEMATIC_C11_DIR / "thematic_aggregate_c11.parquet"
    problems, rows = 0, []

    # ── Thematic aggregation per cell-year ────────────────────────────────────
    ext = pd.read_parquet(args.extraction)
    if set(ext["radius_m"].unique()) != {r}:
        sys.exit(f"ERROR: extraction radius {sorted(ext['radius_m'].unique())} ≠ config {r}")
    th = T.aggregate_categories(ext, T.load_legend(LEGEND), ["unit", "radius_m", "year"])
    clo = float(T.closure_rel(th).abs().max())
    n_c0 = int((th["area_nodata_m2"] > 0).sum())
    log.info("Thematic: %d cell-years | closure max |rel| %.2e | cell-years with class 0: %d",
             len(th), clo, n_c0)
    problems += int(clo > TOL) + int(n_c0 > 0)
    rows += [{"check": "closure_max_rel", "value": clo, "tolerance": TOL, "ok": bool(clo <= TOL)},
             {"check": "cell_years_with_class0", "value": n_c0, "tolerance": 0, "ok": bool(n_c0 == 0)}]

    # ── Change per cell and period ────────────────────────────────────────────
    ch = C.period_change(th, periods)
    C.check_periods_additivity(ch, periods)
    log.info("Periods add up (main and test), per cell")

    nat = pd.read_parquet(nat_path)
    nat = nat[(nat["unit"] == NATIONAL) & (nat["radius_m"] == r)]
    nch = C.period_change(nat, periods).set_index("period")
    tot = ch.groupby("period")["delta_anthropic_m2"].sum()
    rel = ((tot - nch["delta_anthropic_m2"]).abs() / nch["area_with_data_mean_m2"]).max()
    log.info("Σ cells vs NATIONAL delta (/ national area): max %.2e", rel)
    problems += bool(rel > TOL)
    rows.append({"check": "cells_vs_national_delta_max_rel", "value": float(rel), "tolerance": TOL,
                 "ok": bool(rel <= TOL)})

    grid = pd.read_parquet(grid_path, columns=["hex_id", "n_sites", "sparse", "buffer_area_m2"])
    expected = set(grid.loc[grid["buffer_area_m2"] > 0, "hex_id"])
    got = set(ch["unit"])
    if expected != got:
        log.error("Cells differ from %s: %d missing, %d extra", grid_path.name,
                  len(expected - got), len(got - expected))
        problems += 1
    rows.append({"check": "cells_match_grid", "value": len(expected ^ got), "tolerance": 0,
                 "ok": expected == got})

    ch = (ch.rename(columns={"unit": "hex_id"})
          .merge(grid[["hex_id", "n_sites", "sparse"]], on="hex_id", how="left"))
    ch.insert(1, "n_sites", ch.pop("n_sites"))
    ch.insert(2, "sparse", ch.pop("sparse"))

    for period, g in ch[~ch["sparse"]].groupby("period", sort=False):
        n = nch.loc[period]
        log.info("  %-10s %-4s cells %4d | rate pp/yr median %+.3f (P10 %+.3f, P90 %+.3f) | "
                 "gain %4d, loss %4d | Brazil %+.3f",
                 period, len(g), g["rate_pp_per_yr"].median(),
                 g["rate_pp_per_yr"].quantile(0.1), g["rate_pp_per_yr"].quantile(0.9),
                 int((g["delta_pp"] > 0).sum()), int((g["delta_pp"] < 0).sum()), n["rate_pp_per_yr"])

    # ── Save ──────────────────────────────────────────────────────────────────
    save(th, out_dir / "thematic_hex_c11.parquet", index=False)
    save(ch, out_dir / f"hex_anthropic_change_r{r}.parquet", index=False)
    if problems:
        sys.exit(f"ERROR: {problems} check(s) failed")
    log.info("Done → %s", out_dir)


if __name__ == "__main__":
    main()
