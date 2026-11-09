"""Fig 2 — land-cover change around the sites: (A) composition 1985 → 2025, (B–E) hexagon maps.

Draws only. Composition from `summary/landcover_composition_national.csv` (NATIONAL
dissolve, 100 m; scripts/14_summary_tables.py); change per cell from
`hex/hex_anthropic_change_r100.parquet` (scripts/12_hex_anthropic_change.py),
cell geometry from `hex/hex_grid_100km.parquet`. Periods = `periods` of config/hex.toml
(the one spanning the others goes in the large panel); test periods are left out.

(A) Composition-only alluvial (`figures/alluvial.py`): the ribbons link each class to itself and do
not represent pixel transitions (state this in the caption). (B–E) Classes of
`style.ANTHROPIC_CHANGE_*`, with arrows for the extremes; cells with fewer than `min_sites`
sites are omitted (background). Maps in EPSG:5880, without a scale bar; inner biome boundaries
generalized by `INNER_SIMPLIFY_M` for drawing only, thin (exception to the 2 pt rule,
`style.BOUNDARY_WIDTH`).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec

from sitios_arq import hexgrid as H
from sitios_arq import style as S
from sitios_arq.figures.alluvial import draw_alluvial
from sitios_arq.figures.sites_biomes import mainland_bounds, national_outline
from sitios_arq.paths import HEX_DIR, OUT_BIOMAS, REPO_ROOT, SUMMARY_DIR

NAME = "landcover_change"
DESCRIPTION = ("Fig 2: land-cover composition around sites 1985 vs 2025 (alluvial) and change in "
               "anthropic cover per 100 km hexagon and period (EPSG:5880)")
HEX_CONFIG = REPO_ROOT / "config/hex.toml"
CRS = 5880

# Generalization for drawing only: 250 m on the outline; 5 km on the inner boundaries, which
# become noise over 100 km cells (the 1:250,000 line follows rivers).
SIMPLIFY_M = 250
INNER_SIMPLIFY_M = 5000

FIG_HEIGHT_MM = 170
CBAR_LABEL = "Change in anthropic cover within 100 m of sites (%)"


def inputs() -> list[Path]:
    return [SUMMARY_DIR / "landcover_composition_national.csv",
            HEX_DIR / "hex_anthropic_change_r100.parquet",
            HEX_DIR / "hex_grid_100km.parquet", OUT_BIOMAS]


def load():
    comp_path, change_path, grid_path, biomes_path = inputs()
    comp = pd.read_csv(comp_path)
    change = pd.read_parquet(change_path)
    grid = gpd.read_parquet(grid_path, columns=["hex_id", "n_sites", "sparse", "geometry"])
    biomes = gpd.read_parquet(biomes_path)
    for g, label in ((grid, "grid"), (biomes, "biomes")):
        if g.crs is None or g.crs.to_epsg() != CRS:
            raise ValueError(f"{label}: CRS {g.crs} != EPSG:{CRS}")
    biomes["geometry"] = biomes.geometry.simplify(SIMPLIFY_M, preserve_topology=True)

    cfg = H.load_config(HEX_CONFIG)
    main = H.periods(cfg)
    full = (min(s for s, _ in main), max(e for _, e in main))
    periods = [full] + [p for p in main if p != full]
    labels = [f"{s}–{e}" for s, e in periods]

    # consistency checks against the tables (nothing is recomputed here)
    missing = set(labels) - set(change["period"])
    if missing:
        raise ValueError(f"Periods missing in {change_path.name}: {sorted(missing)}")
    years = sorted(comp["year"].unique())
    if years != [full[0], full[1]]:
        raise ValueError(f"Composition years {years} != full period {full}")
    if not (comp.groupby("year")["pct"].sum().sub(100).abs() < 1e-6).all():
        raise ValueError("Composition shares do not add up to 100")
    shown = change[(change["period"].isin(labels)) & ~change["sparse"]]
    if shown["delta_pp"].isna().any():
        raise ValueError("Cells with ≥ min_sites sites and no value")
    if not shown.groupby("period")["hex_id"].nunique().eq((~grid["sparse"]).sum()).all():
        raise ValueError("Shown cells differ from the non-sparse cells of the grid")
    return comp, change, grid, biomes, labels


def scale():
    colors = S.ANTHROPIC_CHANGE_COLORS
    cmap = ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    return cmap, BoundaryNorm(S.ANTHROPIC_CHANGE_BOUNDS, cmap.N)


def draw_composition(ax, comp: pd.DataFrame) -> None:
    pct = comp.pivot(index="year", columns="category", values="pct").sort_index() / 100
    draw_alluvial(ax, pct)


def draw_map(ax, cells: gpd.GeoDataFrame, biomes: gpd.GeoDataFrame, inner: gpd.GeoSeries,
             title: str, cmap, norm) -> None:
    cells.plot(ax=ax, color=cmap(norm(cells["delta_pp"].to_numpy())), edgecolor="none", zorder=1)
    inner.plot(ax=ax, color=S.GREY_DARK, linewidth=S.BOUNDARY_WIDTH, zorder=2)
    national_outline(biomes).plot(ax=ax, color=S.INK, linewidth=S.LINE_WIDTH, zorder=3)
    xmin, ymin, xmax, ymax = mainland_bounds(biomes)
    pad = 0.01 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=S.FONT_SIZE, pad=2)


def draw_colorbar(fig, cax, cmap, norm) -> None:
    bounds = S.ANTHROPIC_CHANGE_BOUNDS
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="both", spacing="uniform", ticks=bounds, extendfrac=0.08)
    cb.ax.set_xticklabels([f"{b:g}".replace("-", "−") for b in bounds])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, labelsize=S.FONT_SIZE_SMALL)
    cb.set_label(CBAR_LABEL, fontsize=S.FONT_SIZE_SMALL)


def make() -> plt.Figure:
    S.apply_style()
    comp, change, grid, biomes, labels = load()
    cmap, norm = scale()
    inner = gpd.GeoSeries(biomes.geometry.simplify(INNER_SIMPLIFY_M, preserve_topology=True).boundary,
                          crs=biomes.crs)
    shown = grid[~grid["sparse"]]

    fig = plt.figure(figsize=(S.mm_to_in(S.WIDTH_2COL_MM), S.mm_to_in(FIG_HEIGHT_MM)))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.15, 1.0, 0.07], hspace=0.12, wspace=0.05,
                  left=0.04, right=0.99, top=0.93, bottom=0.08)
    draw_composition(fig.add_subplot(gs[0, 0:2]), comp)
    axes = [fig.add_subplot(gs[0, 2:6])] + [fig.add_subplot(gs[1, 2 * i:2 * i + 2]) for i in range(3)]
    for ax, lb in zip(axes, labels):
        cells = shown.merge(change.loc[change["period"] == lb, ["hex_id", "delta_pp"]], on="hex_id")
        draw_map(ax, cells, biomes, inner, lb, cmap, norm)
    draw_colorbar(fig, fig.add_subplot(gs[2, 1:5]), cmap, norm)

    # panel letters outside each panel area (replaced by the journal)
    kw = dict(fontsize=S.FONT_SIZE_PANEL, fontweight="medium", va="top", ha="left")
    fig.text(0.01, 0.985, "(A)", **kw)
    fig.text(axes[0].get_position().x0 - 0.01, 0.985, "(B)", **kw)
    for ax, letter in zip(axes[1:], "CDE"):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.01, bb.y1 + 0.035, f"({letter})", **kw)
    return fig
