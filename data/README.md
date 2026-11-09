# Input data

Data are not stored in this repository. The code reads them from `$SITIOS_DATA_ROOT` (default: this folder), with the layout below. The rasters can live elsewhere (`$SITIOS_RASTER_DIR`).

The Zenodo data package ([10.5281/zenodo.24086255](https://doi.org/10.5281/zenodo.24086255)) follows the same layout once `raw.zip` and `processed.zip` are unzipped side by side, and already contains the IPHAN snapshot and every processed product needed for steps 09 onwards (see the main README).

```
$SITIOS_DATA_ROOT/
├── raw/
│   ├── iphan_geoserver_2026-09-08/sitios/sitios.shp
│   └── brazil_coverage-col11_{1985..2025}.tif     (or in $SITIOS_RASTER_DIR)
├── shapefiles/
│   ├── BR_UF_2023/BR_UF_2023.shp
│   ├── Biomas_250mil/lm_bioma_250.shp
│   ├── ilhas_ibge_bc250/bc250_2026-03-03.gpkg
│   └── massa_dagua/massa_dagua.shp
└── processed/                                      written by the pipeline
```

| Path | Source | Accessed | Used by |
|---|---|---|---|
| `raw/iphan_geoserver_2026-09-08/sitios/` | IPHAN GeoServer, layer `SICG:sitios` (http://geoserver.iphan.gov.br/geoserver/SICG/ows). The layer changes over time; the snapshot used is in the Zenodo package | 2026-09-08 | 03, 08 |
| `raw/brazil_coverage-col11_{year}.tif` | MapBiomas Brazil Collection 11, annual land-cover maps of Brazil, 1985–2025 (EPSG:4326, ~30 m, one file per year; https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection11/lulc/coverage/brazil_coverage/brazil_coverage-col11_{year}.tif) | | 05, 06, 11 |
| `shapefiles/BR_UF_2023/` | IBGE, federative units 2023 (territorial mesh; https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2023/Brasil/BR_UF_2023.zip) | | 01 |
| `shapefiles/Biomas_250mil/` | IBGE, biomes 1:250,000 (https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/) | 2025-03-12 | 01 |
| `shapefiles/ilhas_ibge_bc250/bc250_2026-03-03.gpkg` | IBGE, Continuous Cartographic Base 1:250,000 (BC250), version 2026-03-03; layers `lml_pais_a`, `hid_ilha_a`, `lml_municipio_a` | | 02 |
| `shapefiles/massa_dagua/` | ANA/SNIRH, national reference database of water bodies, version 2019 (`geoft_bho_massa_dagua_v2019`, updated in May 2020; water bodies filled later are not in the layer) (https://metadados.snirh.gov.br/geonetwork/srv/api/records/7d054e5a-8cc9-403c-9f1a-085fd933610c) | 2026-03-11 | 13 |
