"""Reader for the IPHAN archaeological-site snapshot (GeoServer, 2026-09-08).

The data are read as delivered: no coordinate is reprojected, corrected or
dropped here. Some points have coordinates outside the lon/lat range (e.g. UTM
values entered as degrees); screening for that is done in
`sitios_arq.filtering`, which only sets flags.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from sitios_arq.paths import IPHAN_SITIOS_SHP

SHP_SITIOS_2026_09_08 = IPHAN_SITIOS_SHP
CRS_IPHAN = 4674  # SIRGAS 2000 geographic (see .prj)


def load_sitios_iphan(path: str | Path) -> gpd.GeoDataFrame:
    """Read the site shapefile, with raw `lon`/`lat` copied from the geometry.

    Checks the CRS and uniqueness of `id_bem` (primary identifier).
    """
    gdf = gpd.read_file(path)

    epsg = gdf.crs.to_epsg() if gdf.crs else None
    if epsg != CRS_IPHAN:
        # The GeoServer .prj is a BOUNDCRS SIRGAS 2000 → WGS 84; to_epsg()
        # may not recognize it. Accept only if the source CRS is EPSG:4674.
        fonte = getattr(gdf.crs, "source_crs", None)
        if fonte is None or fonte.to_epsg() != CRS_IPHAN:
            raise ValueError(f"Expected CRS EPSG:{CRS_IPHAN}, found {gdf.crs}")
        gdf = gdf.set_crs(CRS_IPHAN, allow_override=True)

    if not gdf["id_bem"].is_unique:
        raise ValueError("id_bem is not unique in the IPHAN snapshot")

    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf
