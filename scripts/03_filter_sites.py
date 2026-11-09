"""Filter the IPHAN database (snapshot 2026-09-08) to continental Brazil.

Every rule runs on every site; each site gets every flag it raises, a
`status` and a `primary_reason`. Parameters in config/filtering.toml;
manual decisions in config/filtering_decisions.csv (hand-edited; override
the rules); data-availability exclusions in config/class0_exclusions.csv
(override rules and decisions).

Usage:
    python scripts/03_filter_sites.py

Outputs (in <SITIOS_DATA_ROOT>/processed/):
- sites_filtered.parquet    `kept` sites only (EPSG:5880)
- filtering_status.parquet  every site: status, reason, flags, metrics
- review_register.csv       removed and pending sites (read-only; `current_decision`)
- review.gpkg               same content as the register, points at raw coordinates (EPSG:4674)
- filtering_summary.csv     counts and distributions (long format)
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from sitios_arq import filtering as F
from sitios_arq.iphan import SHP_SITIOS_2026_09_08 as SHP_SITES, load_sitios_iphan
from sitios_arq.paths import PROCESSED_DIR as PROC
from sitios_arq.paths import REF_RASTER_C11_1985 as TIF_MAPBIOMAS
from sitios_arq.io import save

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "config/filtering.toml"
DECISIONS = ROOT / "config/filtering_decisions.csv"
EXCLUSIONS = ROOT / "config/class0_exclusions.csv"

REFERENCES = {
    "brazil": "brazil_5880",
    "land": "land_5880",
    "neighbors": "neighbor_countries_5880",
    "coastline": "coastline_5880",
    "border": "border_line_5880",
    "islands": "islands_5880",
    "municipalities": "municipalities_5880",
    "biomes": "biomas_5880",
}

QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_reference(name: str) -> gpd.GeoDataFrame:
    path = PROC / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{name}.parquet not found in {PROC}; "
            "run scripts/01_prepare_ibge_layers.py and scripts/02_prepare_filtering_references.py"
        )
    gdf = gpd.read_parquet(path)
    if gdf.crs is None or gdf.crs.to_epsg() != F.CRS_METRIC:
        raise ValueError(f"{name}: expected CRS EPSG:{F.CRS_METRIC}")
    return gdf


def load_references() -> F.References:
    g = {k: read_reference(v) for k, v in REFERENCES.items()}
    refs = F.References(
        brazil=g["brazil"].geometry.iloc[0],
        land=g["land"].geometry.iloc[0],
        neighbors=g["neighbors"],
        coastline=g["coastline"],
        border=g["border"],
        islands=g["islands"],
        municipalities=g["municipalities"],
        biomes=g["biomes"],
    )
    return refs


def mapbiomas_bounds() -> tuple[float, float, float, float]:
    """Collection bounds read from the raster metadata (not hard-coded)."""
    with rasterio.open(TIF_MAPBIOMAS) as r:
        if r.crs.to_epsg() != 4326:
            raise ValueError(f"MapBiomas raster: expected CRS EPSG:4326, found {r.crs}")
        b = r.bounds
    return (b.left, b.bottom, b.right, b.top)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(table: pd.DataFrame, flags: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []

    def add(dimension, category, value):
        rows.append({"dimension": dimension, "category": str(category), "value": value})

    add("total", "sites", len(table))
    for k, n in table["status"].value_counts().items():
        add("status", k, n)
    for k, n in table["status_source"].value_counts().items():
        add("status_source", k, n)
    for c in flags.columns:
        add("flag", c.removeprefix("flag_"), int(flags[c].sum()))
    for k, n in table["primary_reason"].fillna("(none)").value_counts().items():
        add("primary_reason", k, n)
    for k, n in table["coast_side"].fillna("(invalid)").value_counts().items():
        add("coast_side", k, n)
    bs = table.groupby([table["biome"].fillna("(outside)"), "status"]).size()
    for (b, s), n in bs.items():
        add("biome_status", f"{b}|{s}", n)
    for term in F.values(cfg)["shipwreck_terms"]:
        n = table["shipwreck_terms"].str.split("|").map(lambda ts: term in ts).sum()
        add("shipwreck_term", term, int(n))
    for col, side in [("dist_coast_m", "sea"), ("dist_coast_m", "land"),
                      ("dist_border_m", "other_country"), ("dist_border_m", "land")]:
        q = table.loc[table["coast_side"].eq(side), col].quantile(QUANTILES)
        for p, x in q.items():
            add(f"{col}|{side}", f"q{p:.2f}", round(float(x), 1))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("=== 03_filter_sites.py ===\n")

    cfg = F.load_config(CONFIG)
    out_dir = PROC

    print(f"Reading {SHP_SITES} ...")
    sites = load_sitios_iphan(SHP_SITES)
    print(f"  {len(sites)} sites")

    print("Reading references ...")
    refs = load_references()
    bounds = mapbiomas_bounds()
    print(f"  MapBiomas bounds (lon/lat): {tuple(round(v, 4) for v in bounds)}")

    print("Computing metrics, flags and status ...")
    m = F.compute_metrics(sites, refs, cfg, bounds)
    flags = F.build_flags(m, cfg)
    decisions = F.load_decisions(DECISIONS)
    s = F.apply_decisions(m, flags, decisions, cfg)
    exclusions = F.load_exclusions(EXCLUSIONS)
    s = F.apply_exclusions(s, exclusions)
    table = F.status_table(sites, m, s)
    register = F.review_register(table)

    kept = sites.loc[s["status"].eq("kept").to_numpy()].copy()
    kept = kept.join(table.set_index(sites.index)[["biome", "spatial_mun"]])
    if kept["lon"].abs().gt(180).any() or kept["lat"].abs().gt(90).any():
        print("  WARNING: some kept sites (manual decision) have invalid coordinates")
    kept = kept.to_crs(F.CRS_METRIC).reset_index(drop=True)  # no __index_level_0__ (P-027 c)

    review = gpd.GeoDataFrame(
        register, geometry=gpd.points_from_xy(register["lon"], register["lat"]), crs=F.CRS_GEOGRAPHIC
    )

    print(f"\nSaving to {out_dir}/ ...")
    save(kept, out_dir / "sites_filtered.parquet")
    save(table, out_dir / "filtering_status.parquet")
    save(register, out_dir / "review_register.csv")
    save(review, out_dir / "review.gpkg", layer="review")
    res = summary(table, flags, cfg)
    save(res, out_dir / "filtering_summary.csv")

    print("\n=== SUMMARY ===")
    for dim in ["total", "status", "status_source", "flag", "primary_reason", "coast_side"]:
        sub = res[res["dimension"] == dim]
        print(f"\n{dim}:")
        for _, r in sub.iterrows():
            print(f"  {r['category']:<22} {r['value']:g}")


if __name__ == "__main__":
    main()
