"""Fig 5 — change in anthropic and water cover around the sites per biome, 1985 → 2025.

Draws only. Δpp from `summary/landcover_change_biomes.csv` (100 m dissolve clipped by
biome; scripts/14_summary_tables.py); number of sites from `summary/sites_per_biome.csv`.
Each biome is a bubble at (Δ anthropic, Δ water); bubble area is proportional to the number
of sites. Dashed zero lines: exception recorded in `style.REFERENCE_LINE_DASH`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Circle
from matplotlib.text import Text

from sitios_arq import style as S
from sitios_arq.paths import SUMMARY_DIR

NAME = "anthropic_water_change"
DESCRIPTION = ("Fig 5: change in anthropic (x) and water (y) cover around sites per biome, "
               "1985–2025, dissolved 100 m buffers clipped by biome; bubble area ∝ number of sites")
FIG_HEIGHT_MM = 100            # 180 × 100 mm (2 columns)

# Bubble area (pt²) per site: 11,507 sites (Mata Atlântica) ≈ 12 mm diameter.
PT2_PER_SITE = 1 / 10
LEGEND_SITES = [1000, 5000, 10000]
LEGEND_ANCHOR = (0.25, 0.33)     # centre-left corner of the size legend (axes fraction)
LEGEND_GAP_PT = 3.0              # vertical gap between the example bubbles

PAD_TOP = 3.5                    # margin above the largest Δ water (pp), for the quadrant label
PAD_X, PAD_Y = 2.5, 2.5          # margin from the data to the axis limits (pp)
LEFT_QUADRANT_FRAC = 0.18        # minimum width of the x < 0 quadrant, to fit the label
LABEL_GAP_PT = 2.0
# Label side relative to the bubble (default: right)
LABEL_SIDE = {"Caatinga": "above", "Pantanal": "left"}

# Quadrant labels: what the sign of the changes implies for the landscape, without assigning
# a cause (increasing water is not necessarily a reservoir; decreasing water is not necessarily
# drought). Lines left-aligned in every quadrant.
QUADRANTS = [  # (x, y in axes fraction, ha, va, text)
    (0.99, 0.99, "right", "top", "Land conversion and\nwater expansion"),
    (0.01, 0.99, "left", "top", "Water expansion,\nanthropic retreat"),
    (0.99, 0.01, "right", "bottom", "Land conversion,\nwater retreat"),
    (0.01, 0.01, "left", "bottom", "Anthropic and\nwater retreat"),
]


def bubble_size(n_sites):
    """Marker area in pt² (the `s` argument of scatter)."""
    return np.asarray(n_sites, dtype=float) * PT2_PER_SITE


def inputs() -> list[Path]:
    return [SUMMARY_DIR / "landcover_change_biomes.csv", SUMMARY_DIR / "sites_per_biome.csv"]


def load() -> pd.DataFrame:
    change_path, sites_path = inputs()
    change = pd.read_csv(change_path)
    sites = pd.read_csv(sites_path)

    # consistency checks against the tables (nothing is recomputed here)
    if set(change["biome"]) != set(sites["biome"]) or set(sites["biome"]) != set(S.BIOME_COLORS):
        raise ValueError(f"Biomes differ: {sorted(set(change['biome']) ^ set(sites['biome']))}")
    if change[["start", "end"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Change table must have a single period")
    wide = change.pivot(index="biome", columns="category", values="delta_pp")
    df = (wide.rename(columns={"Anthropic": "anthropic", "Water": "water"})[["anthropic", "water"]]
          .join(sites.set_index("biome")[["biome_en", "n_sites"]])
          .sort_values("n_sites", ascending=False, kind="stable"))
    df.attrs["period"] = tuple(change[["start", "end"]].iloc[0])
    return df


def size_legend(ax) -> None:
    """Stacked example bubbles, without overlap (spacing in pt from the radii)."""
    radii = [float(np.sqrt(bubble_size(n)) / 2) for n in LEGEND_SITES]
    rmax = max(radii)
    title_h = S.FONT_SIZE_SMALL * 1.6
    height = title_h + sum(2 * r for r in radii) + LEGEND_GAP_PT * (len(radii) - 1)
    da = DrawingArea(2 * rmax + 40, height, 0, 0)
    da.add_artist(Text(0, height, "Number of sites", va="top", ha="left", fontsize=S.FONT_SIZE_SMALL))
    y = height - title_h
    for n, r in zip(LEGEND_SITES, radii):
        y -= r
        da.add_artist(Circle((rmax, y), r, facecolor=S.BUBBLE_LEGEND_COLOR, edgecolor="none"))
        da.add_artist(Text(2 * rmax + 4, y, f"{n:,}", va="center", ha="left", fontsize=S.FONT_SIZE_SMALL))
        y -= r + LEGEND_GAP_PT
    ax.add_artist(AnchoredOffsetbox("center left", child=da, bbox_to_anchor=LEGEND_ANCHOR,
                                    bbox_transform=ax.transAxes, frameon=False, pad=0, borderpad=0))


def label_offset(side: str, r: float) -> tuple[float, float, str, str]:
    d = r + LABEL_GAP_PT
    return {"right": (d, 0, "left", "center"), "left": (-d, 0, "right", "center"),
            "above": (0, d, "center", "bottom"), "below": (0, -d, "center", "top")}[side]


def make() -> plt.Figure:
    S.apply_style()
    df = load()
    start, end = df.attrs["period"]
    fig, ax = plt.subplots(figsize=(S.mm_to_in(S.WIDTH_2COL_MM), S.mm_to_in(FIG_HEIGHT_MM)))

    xmax = df["anthropic"].max() + PAD_X
    xmin = min(df["anthropic"].min() - PAD_X, -LEFT_QUADRANT_FRAC / (1 - LEFT_QUADRANT_FRAC) * xmax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(df["water"].min(), 0) - PAD_Y, max(df["water"].max(), 0) + PAD_TOP)
    for f in (ax.axvline, ax.axhline):
        f(0, color=S.REFERENCE_LINE_COLOR, linestyle=S.REFERENCE_LINE_DASH, zorder=1)

    ax.scatter(df["anthropic"], df["water"], s=bubble_size(df["n_sites"]),
               color=[S.BIOME_COLORS[b] for b in df.index], edgecolors=S.BUBBLE_EDGE_COLOR,
               linewidths=S.BUBBLE_EDGE_WIDTH, zorder=3)
    halo = [patheffects.withStroke(linewidth=S.TEXT_HALO_WIDTH, foreground=S.TEXT_HALO_COLOR)]
    for biome, row in df.iterrows():
        r = np.sqrt(bubble_size(row["n_sites"])) / 2
        dx, dy, ha, va = label_offset(LABEL_SIDE.get(biome, "right"), r)
        ax.annotate(row["biome_en"], (row["anthropic"], row["water"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, zorder=4, path_effects=halo)

    for x, y, ha, va, text in QUADRANTS:
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=S.GREY_DARK,
                fontsize=S.FONT_SIZE_SMALL, linespacing=1.1, multialignment="left")
    size_legend(ax)

    ax.set_xlabel(f"Change in anthropic cover, {start}–{end} (%)")
    ax.set_ylabel(f"Change in water cover, {start}–{end} (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig
