"""Fig 1 — archaeological sites per biome: (A) map, (B) number of sites per biome.

Draws only. Counts come from `summary/sites_per_biome.csv` (scripts/14_summary_tables.py);
the biome of each point comes from `buffers_100m.parquet` (lookup by `id_bem`, used only
for colour). Fails if the point count per biome differs from the table.

Map in EPSG:5880 (SIRGAS 2000 / Brazil Polyconic), without a scale bar (the Polyconic
scale varies by up to ~6% across the map; the projection is given in the caption). Biomes
as light tints of their own colour, with thin boundaries (exception to the 2 pt rule, see
`style.BOUNDARY_WIDTH`); points with a white halo (`style.POINT_*`).

Islands more than `OFFSHORE_KM` from the mainland and without sites (Abrolhos, oceanic
islands) are not drawn: they only showed up as specks of the 2 pt outline outside the frame.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from shapely.geometry import LineString

from sitios_arq import style as S
from sitios_arq.paths import BUFFER_DIR, OUT_BIOMAS, SITES_FILTERED, SUMMARY_DIR

NAME = "sites_biomes"
DESCRIPTION = "Fig 1: archaeological sites per biome (map, EPSG:5880; bar plot)"
CRS = 5880

BIOME_TINT = 0.16      # fraction of the biome colour in the map background
OFFSHORE_KM = 20       # parts farther from the mainland, without sites, are not drawn
# Generalization for drawing only (1:250,000 is too much detail for a ~100 mm map):
# 250 m ≈ 0.006 mm on paper. Does not change counts or the product in data/processed.
SIMPLIFY_M = 250

# Layout in figure fractions (180 mm wide)
MAP_BOX = [0.01, 0.01, 0.55, 0.93]       # left, bottom, width, height
BAR_BOX = [0.72, 0.17, 0.15, 0.76]
PANEL_B_X = 0.58
PANEL_LABEL_Y = 0.985


def inputs() -> list[Path]:
    return [SITES_FILTERED, OUT_BIOMAS, BUFFER_DIR / "buffers_100m.parquet",
            SUMMARY_DIR / "sites_per_biome.csv"]


def load() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    sites_path, biomes_path, buffers_path, table_path = inputs()
    sites = gpd.read_parquet(sites_path, columns=["id_bem", "geometry"])
    biomes = gpd.read_parquet(biomes_path)
    biomes["geometry"] = biomes.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    lookup = pd.read_parquet(buffers_path, columns=["id_bem", "biome"])
    table = pd.read_csv(table_path)
    for g, label in ((sites, "sites"), (biomes, "biomes")):
        if g.crs is None or g.crs.to_epsg() != CRS:
            raise ValueError(f"{label}: CRS {g.crs} != EPSG:{CRS}")

    sites = sites.merge(lookup, on="id_bem", how="left", validate="one_to_one")
    if sites["biome"].isna().any():
        raise ValueError(f"{sites['biome'].isna().sum()} sites without a biome in the lookup")
    counts = sites["biome"].value_counts()
    expected = table.set_index("biome")["n_sites"]
    if not counts.sort_index().equals(expected.sort_index()):
        raise ValueError(f"Points per biome != sites_per_biome.csv:\n{counts}\n{expected}")
    return sites, drop_offshore_islands(biomes, sites), table


def land_parts(biomes: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Disjoint parts of the union of the biomes (mainland and islands)."""
    return gpd.GeoSeries([biomes.unary_union], crs=biomes.crs).explode(index_parts=False,
                                                                       ignore_index=True)


def drop_offshore_islands(biomes: gpd.GeoDataFrame, sites: gpd.GeoDataFrame,
                          offshore_km: float = OFFSHORE_KM) -> gpd.GeoDataFrame:
    """Removes from the biomes the parts more than `offshore_km` from the mainland with no site."""
    parts = land_parts(biomes)
    mainland = parts.iloc[parts.area.values.argmax()]
    far = parts[parts.distance(mainland) > offshore_km * 1000]
    with_sites = gpd.sjoin(gpd.GeoDataFrame(geometry=far), sites[["geometry"]],
                           predicate="intersects").index.unique()
    drop = far.drop(with_sites)
    if drop.empty:
        return biomes
    pieces = biomes.explode(index_parts=False, ignore_index=True)
    keep = ~pieces.representative_point().intersects(drop.unary_union)
    return pieces[keep].dissolve("Bioma", as_index=False)


def mainland_bounds(biomes: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Extent of the largest polygon (mainland): oceanic islands fall outside the frame."""
    parts = land_parts(biomes)
    return tuple(parts.iloc[[parts.area.values.argmax()]].total_bounds)


def national_outline(biomes: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Exterior rings of the union of the biomes (mainland and remaining islands).

    The interior rings of the union are numerical slivers between neighbouring biomes
    (~3,000 rings, total area ≈ 0 km²; no biome has a hole of its own) and, drawn
    with a 2 pt line, turned into black specks.
    """
    return gpd.GeoSeries([LineString(p.exterior.coords) for p in land_parts(biomes)], crs=biomes.crs)


def draw_map(ax, sites: gpd.GeoDataFrame, biomes: gpd.GeoDataFrame, order: list[str]) -> None:
    for biome, color in S.BIOME_COLORS.items():
        biomes[biomes["Bioma"] == biome].plot(ax=ax, facecolor=S.tint(color, BIOME_TINT),
                                              edgecolor="none", zorder=1)
    biomes.boundary.plot(ax=ax, color=S.BOUNDARY_COLOR, linewidth=S.BOUNDARY_WIDTH, zorder=1.5)
    national_outline(biomes).plot(ax=ax, color=S.INK, linewidth=S.LINE_WIDTH, zorder=2)

    # biomes with more sites underneath; the Pantanal (125) on top
    pts = sites.assign(rank=sites["biome"].map({b: i for i, b in enumerate(order)})).sort_values("rank")
    ax.scatter(pts.geometry.x, pts.geometry.y, s=S.POINT_SIZE, c=pts["biome"].map(S.BIOME_COLORS),
               edgecolors=S.POINT_EDGE_COLOR, linewidths=S.POINT_EDGE_WIDTH,
               rasterized=True, zorder=3)

    xmin, ymin, xmax, ymax = mainland_bounds(biomes)
    pad = 0.01 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_bars(ax, table: pd.DataFrame) -> None:
    y = range(len(table))
    ax.barh(y, table["n_sites"], height=0.62, color=table["biome"].map(S.BIOME_COLORS))
    vmax = table["n_sites"].max()
    for yi, r in zip(y, table.itertuples()):
        ax.text(r.n_sites + 0.03 * vmax, yi, f"{r.n_sites:,} ({r.pct:.1f}%)",
                va="center", ha="left", fontsize=S.FONT_SIZE_SMALL, clip_on=False)

    ax.set_yticks(list(y), table["biome_en"])
    ax.invert_yaxis()
    ax.set_ylabel("Biome")
    ax.set_xlabel("Number of archaeological sites")
    step = 6000
    last_tick = step * -(-vmax // step)
    ax.set_xticks(range(0, int(last_tick) + 1, step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(0, last_tick)  # value labels extend past the axis edge (clip_on=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def make() -> plt.Figure:
    S.apply_style()
    sites, biomes, table = load()

    xmin, ymin, xmax, ymax = mainland_bounds(biomes)
    map_w_mm = S.WIDTH_2COL_MM * MAP_BOX[2]
    height_mm = map_w_mm * (ymax - ymin) / (xmax - xmin) / MAP_BOX[3]
    fig = plt.figure(figsize=(S.mm_to_in(S.WIDTH_2COL_MM), S.mm_to_in(height_mm)))

    ax_map = fig.add_axes(MAP_BOX)
    ax_bar = fig.add_axes(BAR_BOX)
    draw_map(ax_map, sites, biomes, order=list(table["biome"]))
    draw_bars(ax_bar, table)

    # panel letters outside each panel area (replaced by the journal)
    label_kw = dict(fontsize=S.FONT_SIZE_PANEL, fontweight="medium", va="top", ha="left")
    fig.text(0.01, PANEL_LABEL_Y, "(A)", **label_kw)
    fig.text(PANEL_B_X, PANEL_LABEL_Y, "(B)", **label_kw)
    return fig
