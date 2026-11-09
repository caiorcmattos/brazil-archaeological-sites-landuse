"""Standardize the IBGE layers (states and biomes) for the pipeline.

Reads the shapefiles in <SITIOS_DATA_ROOT>/shapefiles/ (read-only), reprojects them
to EPSG:5880 and saves them to <SITIOS_DATA_ROOT>/processed/. Paths are resolved in
sitios_arq.paths.
"""

import argparse
from pathlib import Path

import geopandas as gpd

from sitios_arq.io import save
from sitios_arq.paths import OUT_BIOMAS, OUT_ESTADOS, SHP_BIOMAS, SHP_ESTADOS

CRS_IN = 4674
CRS_OUT = 5880

STATE_COLUMNS = ["NM_UF", "SIGLA_UF", "CD_UF", "geometry"]
BIOME_COLUMNS = ["Bioma", "CD_Bioma", "geometry"]

N_STATES = 27
N_BIOMES = 6


def standardize(shp: Path, columns: list[str], n_expected: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp)
    epsg = gdf.crs.to_epsg() if gdf.crs else None
    if epsg != CRS_IN:
        raise ValueError(f"{shp.name}: expected CRS EPSG:{CRS_IN}, found EPSG:{epsg}")
    gdf = gdf.to_crs(epsg=CRS_OUT)[columns]
    if len(gdf) != n_expected:
        raise ValueError(f"{shp.name}: expected {n_expected} features, found {len(gdf)}")
    return gdf


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("=== 01_prepare_ibge_layers.py ===\n")

    for shp, columns, n, out in [(SHP_ESTADOS, STATE_COLUMNS, N_STATES, OUT_ESTADOS),
                                 (SHP_BIOMAS, BIOME_COLUMNS, N_BIOMES, OUT_BIOMAS)]:
        print(f"Reading {shp} ...")
        gdf = standardize(shp, columns, n)
        save(gdf, out)
        print(f"  {len(gdf)} features, EPSG:{gdf.crs.to_epsg()} → {out}")


if __name__ == "__main__":
    main()
