"""Prepare the filtering reference layers from BC250 (IBGE).

Reads `lml_pais_a`, `hid_ilha_a` and `lml_municipio_a` from
<SITIOS_DATA_ROOT>/shapefiles/ilhas_ibge_bc250/bc250_2026-03-03.gpkg
(read-only; see sitios_arq.paths), reprojects to EPSG:5880 and saves to
<SITIOS_DATA_ROOT>/processed/:

- brazil_5880              Brazil polygon
- land_5880                Brazil ∪ every island of hid_ilha_a
- neighbor_countries_5880  other countries of lml_pais_a
- coastline_5880           outline of land, minus land-border stretches
- border_line_5880         Brazil outline shared with neighbors
- islands_5880             hid_ilha_a (name, island_type, geodesic area_km2)
- municipalities_5880      lml_municipio_a (geocode, name)

Avoids re-reading the GPKG (~1.5 GB) on every run of scripts/03_filter_sites.py.
"""

import argparse

import geopandas as gpd
import shapely

from sitios_arq.filtering import derive_lines
from sitios_arq.geodesia import area_geodesica_m2
from sitios_arq.paths import GPKG_BC250, PROCESSED_DIR as OUT
from sitios_arq.io import save

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRS_IN = 4674
CRS_OUT = 5880

# An outline segment within BORDER_EPS_M of a neighbor country = border.
# In BC250 the outlines coincide: 17,326 km are classified as border for
# any eps between 0.01 m and 10 m (checked on 2026-09-25).
BORDER_EPS_M = 1.0

N_COUNTRIES = 15

# BC250 `tipoilha` → island_type (empty → "untyped").
ISLAND_TYPES = {"Marítima": "maritime", "Fluvial": "fluvial", "Lacustre": "lacustrine", "": "untyped"}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_layer(layer: str, columns: list[str]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(GPKG_BC250, layer=layer)
    epsg = gdf.crs.to_epsg() if gdf.crs else None
    if epsg != CRS_IN:
        raise ValueError(f"{layer}: expected CRS EPSG:{CRS_IN}, found EPSG:{epsg}")
    return gdf[columns + ["geometry"]]


def write_layer(gdf: gpd.GeoDataFrame, name: str) -> None:
    save(gdf, OUT / f"{name}.parquet")
    print(f"  {name}: {len(gdf)} features")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("=== 02_prepare_filtering_references.py ===\n")

    print("Reading lml_pais_a ...")
    countries = read_layer("lml_pais_a", ["nome", "sigla"]).rename(columns={"nome": "name", "sigla": "iso2"})
    if len(countries) != N_COUNTRIES or (countries["iso2"] == "BR").sum() != 1:
        raise ValueError(f"lml_pais_a: expected {N_COUNTRIES} countries with a single BR")
    countries = countries.to_crs(CRS_OUT)
    brazil = countries[countries["iso2"] == "BR"].reset_index(drop=True)
    neighbors = countries[countries["iso2"] != "BR"].reset_index(drop=True)

    print("Reading hid_ilha_a ...")
    islands = read_layer("hid_ilha_a", ["nome", "tipoilha"]).rename(columns={"nome": "name"})
    tipo = islands.pop("tipoilha").fillna("").str.strip()
    unknown = sorted(set(tipo) - set(ISLAND_TYPES))
    if unknown:
        raise ValueError(f"hid_ilha_a: unmapped tipoilha values {unknown}")
    islands.insert(1, "island_type", tipo.map(ISLAND_TYPES))
    islands["area_km2"] = islands.geometry.map(area_geodesica_m2) / 1e6
    islands = islands.to_crs(CRS_OUT)
    islands["geometry"] = shapely.make_valid(islands.geometry.values)

    print("Reading lml_municipio_a ...")
    municipalities = (
        read_layer("lml_municipio_a", ["geocodigo", "nome"])
        .rename(columns={"geocodigo": "geocode", "nome": "name"})
        .to_crs(CRS_OUT)
    )

    print("Union Brazil ∪ islands ...")
    geom_br = brazil.geometry.iloc[0]
    land_geom = shapely.union_all([geom_br, *islands.geometry.values])
    land = gpd.GeoDataFrame(geometry=[land_geom], crs=CRS_OUT)

    print("Deriving coastline and border lines ...")
    coastline, border = derive_lines(geom_br, land_geom, neighbors.geometry.values, BORDER_EPS_M)
    coastline = gpd.GeoDataFrame(geometry=coastline, crs=CRS_OUT)
    border = gpd.GeoDataFrame(geometry=border, crs=CRS_OUT)
    print(f"  coastline: {coastline.length.sum() / 1e3:,.0f} km; border: {border.length.sum() / 1e3:,.0f} km")

    print("\nSaving ...")
    write_layer(brazil, "brazil_5880")
    write_layer(neighbors, "neighbor_countries_5880")
    write_layer(islands, "islands_5880")
    write_layer(municipalities, "municipalities_5880")
    write_layer(land, "land_5880")
    write_layer(coastline, "coastline_5880")
    write_layer(border, "border_line_5880")


if __name__ == "__main__":
    main()
