"""Single place where data paths are resolved, configurable through the environment.

Only where data are read and written is defined here. Two environment variables
control everything:

- SITIOS_DATA_ROOT: data root (raw/, processed/, shapefiles/). Default: <repo>/data.
- SITIOS_RASTER_DIR: directory of the MapBiomas C11 rasters (1985-2025), used by
  scripts/06_extract_mapbiomas.py and scripts/11_extract_hex.py. Default:
  <SITIOS_DATA_ROOT>/raw.

Command-line arguments (--raster-dir, --pixel-area, --out-dir, --buffer-dir) take
priority over these variables: scripts only use these values as argparse defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_ROOT = _env_path("SITIOS_DATA_ROOT", REPO_ROOT / "data")

RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SHAPEFILES_DIR = DATA_ROOT / "shapefiles"

RASTER_DIR = _env_path("SITIOS_RASTER_DIR", RAW_DIR)

# raw/
REF_RASTER_C11_1985 = RAW_DIR / "brazil_coverage-col11_1985.tif"
IPHAN_SITIOS_SHP = RAW_DIR / "iphan_geoserver_2026-09-08/sitios/sitios.shp"

# shapefiles/
SHP_ESTADOS = SHAPEFILES_DIR / "BR_UF_2023/BR_UF_2023.shp"
SHP_BIOMAS = SHAPEFILES_DIR / "Biomas_250mil/lm_bioma_250.shp"
GPKG_BC250 = SHAPEFILES_DIR / "ilhas_ibge_bc250/bc250_2026-03-03.gpkg"
SHP_MASSA_DAGUA = SHAPEFILES_DIR / "massa_dagua/massa_dagua.shp"

# processed/
OUT_ESTADOS = PROCESSED_DIR / "estados_5880.parquet"
OUT_BIOMAS = PROCESSED_DIR / "biomas_5880.parquet"
SITES_FILTERED = PROCESSED_DIR / "sites_filtered.parquet"
BUFFER_DIR = PROCESSED_DIR / "buffers"
PIXEL_AREA_TIF = PROCESSED_DIR / "pixel_area_c11_m2.tif"
EXTRACTION_C11_DIR = PROCESSED_DIR / "extraction_c11"
THEMATIC_C11_DIR = PROCESSED_DIR / "thematic_c11"
SUMMARY_DIR = PROCESSED_DIR / "summary"
HEX_DIR = PROCESSED_DIR / "hex"
EXTRACTION_HEX_C11_DIR = PROCESSED_DIR / "extraction_hex_c11"
RESERVOIRS_DIR = PROCESSED_DIR / "reservoirs"

# figures (outside data/)
FIGURES_DIR = REPO_ROOT / "outputs/figures"
