"""Fig 3 — land-cover composition around the sites per biome, 1985 → 2025 (alluvials).

Draws only. Composition from `summary/landcover_composition_biomes.csv` (100 m dissolve
clipped by biome; scripts/14_summary_tables.py); number of sites and panel order
(descending) from `summary/sites_per_biome.csv`. n is the number of sites assigned to the
biome; the composition comes from the clipped dissolve.

Composition-only alluvial (`figures/alluvial.py`), as in Fig 2A: the ribbons do not represent
pixel transitions (state this in the caption). Class labels on thin ribbons go above the panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sitios_arq import style as S
from sitios_arq.figures.alluvial import CATEGORIES, draw_alluvial
from sitios_arq.paths import SUMMARY_DIR

NAME = "landcover_change_biomes"
DESCRIPTION = ("Fig 3: land-cover composition around sites 1985 vs 2025 per biome (alluvials, "
               "dissolved 100 m buffers clipped by biome), biomes ordered by number of sites")
FIG_HEIGHT_MM = 135
NROWS, NCOLS = 2, 3


def inputs() -> list[Path]:
    return [SUMMARY_DIR / "landcover_composition_biomes.csv", SUMMARY_DIR / "sites_per_biome.csv"]


def load():
    comp_path, sites_path = inputs()
    comp = pd.read_csv(comp_path)
    sites = pd.read_csv(sites_path).sort_values("n_sites", ascending=False, kind="stable")

    # consistency checks against the tables (nothing is recomputed here)
    if set(comp["biome"]) != set(sites["biome"]) or len(sites) != NROWS * NCOLS:
        raise ValueError(f"Biomes differ: {sorted(set(comp['biome']) ^ set(sites['biome']))}")
    if set(comp["category"]) != set(CATEGORIES) or comp["year"].nunique() != 2:
        raise ValueError("Composition table must have the three categories and two years")
    if not (comp.groupby(["biome", "year"])["pct"].sum().sub(100).abs() < 1e-6).all():
        raise ValueError("Composition shares do not add up to 100")
    return comp, sites


def header(ax, biome_en: str, n_sites: int, letter: str) -> None:
    kw = dict(xycoords="axes fraction", textcoords="offset points", va="bottom")
    ax.annotate(f"n = {n_sites:,}", (0.5, 1.0), xytext=(0, 13), ha="center", fontsize=S.FONT_SIZE_SMALL,
                color=S.GREY_DARK, **kw)
    ax.annotate(biome_en, (0.5, 1.0), xytext=(0, 24), ha="center", fontsize=S.FONT_SIZE,
                fontweight="medium", **kw)
    # letter outside the panel area (replaced by the journal)
    ax.annotate(f"({letter})", (0.0, 1.0), xytext=(-2, 24), ha="left", fontsize=S.FONT_SIZE_PANEL,
                fontweight="medium", **kw)


def make() -> plt.Figure:
    S.apply_style()
    comp, sites = load()
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(S.mm_to_in(S.WIDTH_2COL_MM), S.mm_to_in(FIG_HEIGHT_MM)))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.14, hspace=0.36)
    for ax, row, letter in zip(axes.ravel(), sites.itertuples(), "ABCDEF"):
        pct = (comp[comp["biome"] == row.biome].pivot(index="year", columns="category", values="pct")
               .sort_index() / 100)
        draw_alluvial(ax, pct, pct_fmt="{:.0f}%", year_size=S.FONT_SIZE_SMALL, nudge_thin=True)
        header(ax, row.biome_en, row.n_sites, letter)
    return fig
