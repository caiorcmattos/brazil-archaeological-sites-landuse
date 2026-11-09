"""Fig 4 — buffers entirely classified as water around reservoirs: (A) per biome, (B) top hydropower.

Draws only. Counts from `summary/reservoir_water_biomes.csv` and `summary/reservoir_water_hydro.csv`
(scripts/14_summary_tables.py). Stacked bars: buffers entirely water already at
the start of the series (reservoirs older than 1985) and only at the end. Buffers with partial gain
are reported in the text. (A) shows the biomes with at least one buffer entirely water (not Pampa
or Pantanal), in descending order; (B) the hydropower reservoirs marked `in_figure` in the table,
in `rank` order.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from sitios_arq import style as S
from sitios_arq.extraction import NATIONAL
from sitios_arq.paths import SUMMARY_DIR

NAME = "reservoir_water"
DESCRIPTION = ("Fig 4: 100 m buffers entirely classified as water around artificial water bodies, "
               "start vs end of the series, per biome (A) and for the hydropower reservoirs with most (B)")
FIG_HEIGHT_MM = 115

PARTS = ["entirely_water_both", "entirely_water_end"]   # stacking order (from the left)
BAR_HEIGHT = 0.7
LABEL_WRAP = 18           # characters per line in the y-axis labels
TOTAL_GAP_PT = 3.0        # from the end of the bar to the total
X_PAD = 1.12              # x-axis limit / largest total (room for the total)
AXES_LEFT, AXES_RIGHT = 0.20, 0.97
# Correct spelling where the ANA name differs (label only; the tables keep the ANA name).
NAME_LABELS = {"UHE Luis Eduardo Magalhães": "UHE Luís Eduardo Magalhães"}
PANEL_TOPS = (0.94, 0.50)  # top of axes (A) and (B), in figure fraction
PANEL_HEIGHT = 0.32


def inputs() -> list[Path]:
    return [SUMMARY_DIR / "reservoir_water_biomes.csv", SUMMARY_DIR / "reservoir_water_hydro.csv"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    biomes_path, hydro_path = inputs()
    b = pd.read_csv(biomes_path)
    h = pd.read_csv(hydro_path)

    # consistency checks against the tables (nothing is recomputed here)
    for df, name in ((b, "biomes"), (h, "hydro")):
        if not (df["n_entirely_water"] == df[[f"n_{p}" for p in PARTS]].sum(axis=1)).all():
            raise ValueError(f"{name}: n_entirely_water != both + end")
        if df[["start", "end"]].drop_duplicates().shape[0] != 1:
            raise ValueError(f"{name}: more than one period")
    if not set(b["unit"]) - {NATIONAL} <= set(S.BIOME_LABELS_EN):
        raise ValueError(f"Unknown units: {sorted(set(b['unit']) - set(S.BIOME_LABELS_EN))}")

    b = (b[(b["unit"] != NATIONAL) & (b["n_entirely_water"] > 0)]
         .sort_values(["n_entirely_water", "n_associated"], ascending=False, kind="stable"))
    b = b.assign(label=b["unit_en"])
    h = h[h["in_figure"]].sort_values("rank")
    h = h.assign(label=h["res_name"].replace(NAME_LABELS))
    period = tuple(b[["start", "end"]].iloc[0])
    if tuple(h[["start", "end"]].iloc[0]) != period:
        raise ValueError("Tables cover different periods")
    return b, h


def bars(ax, df: pd.DataFrame) -> None:
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for part in PARTS:
        ax.barh(y, df[f"n_{part}"], left=left, height=BAR_HEIGHT, color=S.ENTIRELY_WATER_COLORS[part],
                linewidth=0)
        left += df[f"n_{part}"].to_numpy()
    for yi, total in zip(y, df["n_entirely_water"]):
        ax.annotate(f"{total:,}", (total, yi), xytext=(TOTAL_GAP_PT, 0), textcoords="offset points",
                    ha="left", va="center")
    ax.set_yticks(y, [textwrap.fill(s, LABEL_WRAP) for s in df["label"]])
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xlim(0, df["n_entirely_water"].max() * X_PAD)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)


def make() -> plt.Figure:
    S.apply_style()
    b, h = load()
    start, end = b[["start", "end"]].iloc[0]
    fig = plt.figure(figsize=(S.mm_to_in(S.WIDTH_2COL_MM), S.mm_to_in(FIG_HEIGHT_MM)))
    letter_kw = dict(fontsize=S.FONT_SIZE_PANEL, fontweight="medium", va="top", ha="left")
    for letter, top, df in zip("AB", PANEL_TOPS, (b, h)):
        ax = fig.add_axes([AXES_LEFT, top - PANEL_HEIGHT, AXES_RIGHT - AXES_LEFT, PANEL_HEIGHT])
        bars(ax, df)
        ax.set_xlabel(f"Buffers entirely classified as water in {end}")
        fig.text(0.01, top + 0.03, f"({letter})", **letter_kw)

    handles = [Patch(color=S.ENTIRELY_WATER_COLORS["entirely_water_both"],
                     label=f"Already entirely water in {start}"),
               Patch(color=S.ENTIRELY_WATER_COLORS["entirely_water_end"],
                     label=f"Became entirely water after {start}")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=((AXES_LEFT + AXES_RIGHT) / 2, 0.0), handlelength=1.4,
               columnspacing=2.0, fontsize=S.FONT_SIZE)
    return fig
