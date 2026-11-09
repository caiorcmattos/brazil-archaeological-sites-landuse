"""Geodesic buffers around sites, nearest-biome assignment and biome clipping.

Buffers are true geodesic circles on the WGS84 ellipsoid: each vertex lies
exactly `radius_m` geodesic metres from the site (pyproj.Geod.fwd), with
`n_vertices` equally spaced azimuths. They are built from the raw lon/lat
(EPSG:4674) and every later geometric operation (dissolve, biome clip) stays in
lon/lat, the space in which exactextract sees the edges. No reprojection
deforms a buffer after it is built.

WGS84 is the same ellipsoid used by `geodesia.area_geodesica_m2` and by the
pixel-area raster; GRS80 (SIRGAS 2000) differs from it negligibly.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod

from sitios_arq.geodesia import area_geodesica_m2

GEOD = Geod(ellps="WGS84")
CRS_GEOGRAPHIC = 4674
CRS_METRIC = 5880


def polygon_area_factor(n_vertices: int) -> float:
    """Area of a regular n-gon inscribed in a circle, as a fraction of πr²."""
    return n_vertices * math.sin(2 * math.pi / n_vertices) / (2 * math.pi)


def geodesic_buffers(lon, lat, radius_m: float, n_vertices: int) -> np.ndarray:
    """Geodesic circle polygons (lon/lat) around each (lon, lat) point."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)
    az = np.arange(n_vertices) * (360.0 / n_vertices)
    x, y, _ = GEOD.fwd(
        np.repeat(lon, n_vertices), np.repeat(lat, n_vertices),
        np.tile(az, n), np.full(n * n_vertices, float(radius_m)),
    )
    coords = np.stack([x, y], axis=1).reshape(n, n_vertices, 2)
    return shapely.polygons(shapely.linearrings(coords))


def geodesic_areas(geoms) -> np.ndarray:
    return np.array([area_geodesica_m2(g) for g in geoms], dtype=float)


def fill_nearest_biome(sites: gpd.GeoDataFrame, biomes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill null `biome` with the nearest biome polygon (planar distance, EPSG:5880).

    Adds `biome_dist_m`: 0 for sites that already had a biome, the distance to
    the assigned polygon otherwise.
    """
    assert sites.crs.to_epsg() == CRS_METRIC and biomes.crs.to_epsg() == CRS_METRIC
    out = sites.copy()
    out["biome_dist_m"] = 0.0
    missing = out["biome"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(
            out.loc[missing, ["geometry"]], biomes[["Bioma", "geometry"]],
            how="left", distance_col="biome_dist_m",
        )
        near = near[~near.index.duplicated(keep="first")]
        out.loc[missing, "biome"] = near["Bioma"]
        out.loc[missing, "biome_dist_m"] = near["biome_dist_m"]
    if out["biome"].isna().any():
        raise RuntimeError("Some sites still have no biome after nearest assignment")
    return out


def _polygonal(g):
    """Polygonal part of a geometry (drops points/lines from make_valid/intersection)."""
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    polys = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(polys) if polys else None


def disjoint_biomes(biomes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Valid, pairwise-disjoint biome polygons (in `CD_Bioma` order).

    The IBGE layer has one invalid polygon and slivers of overlap (~20 m²);
    each biome loses whatever an earlier biome already covers, so no point is
    in two biomes.
    """
    b = biomes.sort_values("CD_Bioma")
    geoms, seen = [], None
    for g in b.geometry:
        g = _polygonal(shapely.make_valid(g))
        if seen is not None:
            g = _polygonal(shapely.difference(g, seen))
        geoms.append(g)
        seen = g if seen is None else shapely.union(seen, g)
    return gpd.GeoDataFrame({"biome": b["Bioma"].to_numpy()}, geometry=geoms, crs=biomes.crs)


def clip_by_biome(dissolved, biomes_disjoint: gpd.GeoDataFrame,
                  biomes_metric: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Split a dissolved lon/lat geometry among biomes; each point in one biome only.

    The geometry is intersected with each (disjoint) biome polygon. What
    lies outside every biome polygon (coastal or offshore pieces) is merged into
    the nearest biome (planar distance in EPSG:5880) and treated as part of it.

    Returns (GeoDataFrame[biome, geometry] in the CRS of `biomes_disjoint`,
    geodesic area in m² merged from outside the polygons, per biome).
    """
    # One bulk overlay per biome against the whole dissolved geometry. (Testing
    # each part with covered_by against 10⁵-vertex biome polygons first is
    # ~1000× slower than the overlay itself.)
    # The rest is the dissolved geometry minus each biome polygon, not minus the
    # union of the cut pieces: that union can leave zero-width spikes whose
    # pieces lose area (P-026, revisão cega B-15).
    pieces: dict[str, list] = {b: [] for b in biomes_disjoint["biome"]}
    r = dissolved
    for name, g in zip(biomes_disjoint["biome"], biomes_disjoint.geometry):
        piece = _polygonal(shapely.intersection(dissolved, g))
        if piece is not None:
            pieces[name].append(piece)
            r = shapely.difference(r, g)
    r = _polygonal(r)
    rest = [p for p in shapely.get_parts(r) if p.area > 0] if r is not None else []

    outside_m2 = pd.Series(0.0, index=list(pieces))
    if rest:
        rest_gdf = gpd.GeoDataFrame(geometry=rest, crs=biomes_disjoint.crs)
        near = gpd.sjoin_nearest(
            rest_gdf.to_crs(CRS_METRIC), biomes_metric[["Bioma", "geometry"]], how="left",
        )
        near = near[~near.index.duplicated(keep="first")]
        areas = geodesic_areas(rest_gdf.geometry)
        for piece, name, a in zip(rest_gdf.geometry, near.loc[rest_gdf.index, "Bioma"], areas):
            pieces[name].append(piece)
            outside_m2[name] += a

    names = [b for b in pieces if pieces[b]]
    out = gpd.GeoDataFrame(
        {"biome": names},
        geometry=[shapely.union_all(pieces[b]) for b in names],
        crs=biomes_disjoint.crs,
    )
    return out, outside_m2
