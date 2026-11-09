"""Geodesic area of lon/lat geometries (EPSG:4326), in m².

WGS84 ellipsoid — the same used in `scripts/05_pixel_area.py` to build
`pixel_area_c11_m2.tif` and in `sitios_arq.buffers` for the geodesic buffers. Keeping both the same is what makes the geodesic area
comparable with the area exactextract sums from that raster.

Why not call `pyproj.Geod.geometry_area_perimeter()` directly
-------------------------------------------------------------
It works, *provided* interior rings have the opposite orientation to the
exterior — it sums signed areas, and the sign is what makes the hole subtract.
In an earlier version of the pipeline (planar buffers in EPSG:5880), `unary_union`
produced CW exteriors and CCW interiors, and the comparison against exactextract
confirmed it: on the holed parts of the dissolved buffers, both approaches agreed
to a relative error ≤ 3.4e-9, even for parts with holes of up to 2.6 million m².

What this function adds is orientation independence: it takes the absolute
area of each ring and subtracts the interiors explicitly, so a hole cannot be
added by mistake. For a polygon built with the hole in the same orientation as
the exterior, `geometry_area_perimeter()` returns 16.5e9 m² where the correct
value is 7.8e9 — this function returns 7.8e9 in both cases.

Cheap insurance: the geodesic area is the reference against which the whole
extraction closure is checked, so it is worth not depending on a winding
convention that nothing in the pipeline explicitly guarantees.
"""

from __future__ import annotations

from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def _area_anel(coords) -> float:
    """Absolute area of a closed lon/lat ring, in m²."""
    lons, lats = zip(*[(c[0], c[1]) for c in coords])
    return abs(GEOD.polygon_area_perimeter(lons, lats)[0])


def area_geodesica_m2(geom) -> float:
    """Geodesic area of a lon/lat Polygon/MultiPolygon, minus holes.

    Independent of ring orientation.
    """
    tipo = geom.geom_type
    if tipo == "Polygon":
        if geom.is_empty:
            return 0.0
        area = _area_anel(geom.exterior.coords)
        for anel in geom.interiors:
            area -= _area_anel(anel.coords)
        return area
    if tipo == "MultiPolygon":
        return sum(area_geodesica_m2(p) for p in geom.geoms)
    if tipo == "GeometryCollection":
        return sum(
            area_geodesica_m2(g)
            for g in geom.geoms
            if g.geom_type in ("Polygon", "MultiPolygon")
        )
    raise TypeError(f"Unsupported geometry type for area: {tipo}")
