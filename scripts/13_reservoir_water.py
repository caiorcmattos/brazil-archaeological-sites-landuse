"""Sites associated with artificial water bodies (ANA) and their water status.

    python scripts/13_reservoir_water.py
    python scripts/13_reservoir_water.py --out-dir /tmp/smoke
    python scripts/13_reservoir_water.py --water-radius 500 --out-dir <dir>   # radius sensitivity

Parameters in config/reservoirs.toml.

--water-radius (sensitivity): sites stay associated by the radius of the config (100 m);
only the water share and the status use the other radius.

Inputs
------
- data/processed/buffers/buffers_{r}m.parquet          buffers (EPSG:4674)
- data/shapefiles/massa_dagua/massa_dagua.shp          ANA water bodies
- data/processed/extraction_c11/extraction_sites_c11.parquet  area by class per site
- config/mapbiomas_legend_c11.csv                       thematic categories

Outputs (in --out-dir, default data/processed/reservoirs/)
---------------------------------------------------------
- site_reservoir_links.parquet   one row per (site, water body) that intersect: id_bem, biome,
  res_id (ANA gid), res_name, res_use (ANA usoprinc), is_hydro
- reservoir_sites_water.parquet  one row per associated site: water share (%) in the start
  and end years, delta_pp, status (sitios_arq.reservoirs.STATUSES), is_hydro (any link),
  n_reservoirs
- reservoir_sites_water_radius.parquet  the same for every radius of config/extraction.toml
  (radius_m column), with the sites associated at radius_m (100 m): sensitivity to the radius;
  the rows of the main radius equal reservoir_sites_water.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sitios_arq import extraction as E
from sitios_arq import reservoirs as R
from sitios_arq import thematic as T
from sitios_arq.paths import BUFFER_DIR, EXTRACTION_C11_DIR, RESERVOIRS_DIR, SHP_MASSA_DAGUA
from sitios_arq.io import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/reservoirs.toml"
LEGEND = ROOT / "config/mapbiomas_legend_c11.csv"
EXTRACTION_CONFIG = ROOT / "config/extraction.toml"

log = logging.getLogger("reservoirs")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(RESERVOIRS_DIR))
    p.add_argument("--water-radius", type=int,
                   help="radius of the water share and status; association stays at radius_m (sensitivity)")
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    cfg = R.load_config(CONFIG)
    r = cfg["radius_m"]["value"]
    rw = args.water_radius or r
    years = cfg["years"]["value"]
    if rw != r:
        log.warning("Water share at %d m, association at %d m: sensitivity run", rw, r)

    buf_src = BUFFER_DIR / f"buffers_{r}m.parquet"
    buffers = gpd.read_parquet(buf_src, columns=["id_bem", "biome", "geometry"])
    bodies = R.prepare_water_bodies(gpd.read_file(SHP_MASSA_DAGUA), cfg, buffers.crs)
    log.info("Buffers: %d | water bodies kept (%s): %d, hydropower: %d", len(buffers),
             cfg["water_body_type"]["value"], len(bodies), bodies["is_hydro"].sum())

    links = R.link_sites(buffers, bodies)
    log.info("Links: %d | sites: %d | sites with > 1 water body: %d", len(links),
             links["id_bem"].nunique(), links.groupby("id_bem").size().gt(1).sum())
    save(links, out_dir / "site_reservoir_links.parquet")

    ex_src = EXTRACTION_C11_DIR / "extraction_sites_c11.parquet"
    ex = pd.read_parquet(ex_src, filters=[("radius_m", "=", rw), ("year", "in", years)])
    ex = ex[ex["id_bem"].isin(links["id_bem"])]
    water = T.aggregate_categories(ex, T.load_legend(LEGEND), ["id_bem", "year"])
    sites = R.site_table(links, water, years)
    for status, n in sites["status"].value_counts().reindex(R.STATUSES, fill_value=0).items():
        log.info("  %-20s %5d  (hydropower: %d)", status, n,
                 sites.loc[sites["status"] == status, "is_hydro"].sum())
    save(sites, out_dir / "reservoir_sites_water.parquet")


    radii = E.load_config(EXTRACTION_CONFIG)["radii_m"]["value"]
    exr = pd.read_parquet(ex_src, filters=[("radius_m", "in", radii), ("year", "in", years)])
    exr = exr[exr["id_bem"].isin(links["id_bem"])]
    wr = T.aggregate_categories(exr, T.load_legend(LEGEND), ["radius_m", "id_bem", "year"])
    by_r = pd.concat([R.site_table(links, g.drop(columns="radius_m"), years).assign(radius_m=rr)
                      for rr, g in wr.groupby("radius_m")], ignore_index=True)
    same = by_r[by_r["radius_m"] == rw].drop(columns="radius_m").reset_index(drop=True)
    if not same.equals(sites.reset_index(drop=True)):
        raise RuntimeError(f"Radius table at {rw} m differs from reservoir_sites_water")
    for rr, g in by_r.groupby("radius_m"):
        log.info("  %4d m: entirely water %4d (both %3d)", rr, g["status"].str.startswith("entirely").sum(),
                 (g["status"] == "entirely_water_both").sum())
    save(by_r, out_dir / "reservoir_sites_water_radius.parquet")


if __name__ == "__main__":
    main()
