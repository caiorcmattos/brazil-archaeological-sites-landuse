"""Fig. S2 — sensitivity of the buffers entirely classified as water.

Draws only. (A) Sites associated with artificial water bodies per class of water share of the
100 m buffer in 2025, from `summary/reservoir_water_share_hist.csv`: the "exactly 100%" class
(the entirely-water rule) set apart on the left and the 10% classes in descending order, all
in the same grey (the skewed distribution reads better from 100%). The peak at 100% and the
low classes just below it show that the count hardly depends on the threshold (the tolerance
in numbers is in `reservoir_water_tolerance.csv` and in the text). (B) National count per
radius from `summary/reservoir_water_radius.csv` (sites associated at 100 m), with the share
of the 100 m count; the reading as tolerated position error (radius − 100 m) goes in the
figure caption, not on the axis. Tables from scripts/14_summary_tables.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from sitios_arq import style as S
from sitios_arq.extraction import NATIONAL
from sitios_arq.paths import SUMMARY_DIR

NAME = "reservoir_sensitivity"
DESCRIPTION = ("Fig S2: reservoir-associated sites per water share of the 100 m buffer in 2025 (A) "
               "and buffers entirely classified as water per buffer radius (B)")
FIG_HEIGHT_MM = 85
PARTS = ["entirely_water_both", "entirely_water_end"]
AX_A = [0.085, 0.20, 0.42, 0.62]
AX_B = [0.66, 0.20, 0.27, 0.62]
GAP = 1.0               # gap (in class widths) between [90, 100) and the 100% class
BAR_FRAC = 0.85         # bar width / class width
COUNT_GAP_PT = 2.0
BAR_HEIGHT = 0.65
Y_PAD = 1.12            # axis limit / largest count


def inputs() -> list[Path]:
    return [SUMMARY_DIR / "reservoir_water_share_hist.csv", SUMMARY_DIR / "reservoir_water_radius.csv"]


def load():
    hist_path, rad_path = inputs()
    hist = pd.read_csv(hist_path)
    rad = pd.read_csv(rad_path)

    # consistency checks against the tables (nothing is recomputed here)
    if not (rad["n_entirely_water"] == rad[[f"n_{p}" for p in PARTS]].sum(axis=1)).all():
        raise ValueError("radius: n_entirely_water != both + end")
    if rad[["start", "end"]].drop_duplicates().shape[0] != 1 or hist["year"].nunique() != 1:
        raise ValueError("More than one period")
    if int(hist["year"].iloc[0]) != int(rad["end"].iloc[0]):
        raise ValueError("Histogram year differs from the end year of the radius table")
    hist = hist[hist["unit"] == NATIONAL].sort_values("bin_low")
    rad = rad[rad["unit"] == NATIONAL].sort_values("radius_m")
    ref = rad[rad["tolerated_displacement_m"] == 0].iloc[0]
    if int(hist.loc[hist["entirely_water"], "n_sites"].iloc[0]) != int(ref["n_entirely_water"]):
        raise ValueError("The 100% class differs from the entirely-water count at the reference radius")
    if int(hist["n_sites"].sum()) != int(ref["n_associated"]):
        raise ValueError("Histogram total differs from the associated sites")
    return hist, rad, ref


def count_label(ax, x: float, n: int) -> None:
    ax.annotate(f"{n:,}", (x, n), xytext=(0, COUNT_GAP_PT), textcoords="offset points", ha="center",
                va="bottom", fontsize=S.FONT_SIZE_SMALL)


def draw_share(ax, hist: pd.DataFrame) -> None:
    """10% classes and, separated by GAP, the 100% class; x axis inverted (100% on the left)."""
    part = hist[~hist["entirely_water"]]
    width = float((part["bin_high"] - part["bin_low"]).iloc[0])
    x_full = 100 + (GAP + 0.5) * width
    x = np.r_[((part["bin_low"] + part["bin_high"]) / 2).to_numpy(), x_full]
    n = np.r_[part["n_sites"].to_numpy(), hist.loc[hist["entirely_water"], "n_sites"].to_numpy()]
    ax.bar(x, n, width=BAR_FRAC * width, color=S.GREY_MID, linewidth=0)
    for xi, ni in zip(x, n):
        count_label(ax, xi, int(ni))

    edges = list(range(0, 101, 20))
    ax.set_xticks([*edges, x_full], [*map(str, edges), "Entirely\nwater"])
    ax.set_xlim(x_full + width, -0.5 * width)
    ax.set_ylim(0, hist["n_sites"].max() * Y_PAD)
    ax.set_xlabel(f"Water share of the 100 m buffer in {int(hist['year'].iloc[0])} (%)")
    ax.set_ylabel("Reservoir-associated sites")
    ax.spines[["top", "right"]].set_visible(False)


def draw_radius(ax, rad: pd.DataFrame) -> None:
    y = np.arange(len(rad))
    left = np.zeros(len(rad))
    for part in PARTS:
        ax.barh(y, rad[f"n_{part}"], left=left, height=BAR_HEIGHT, color=S.ENTIRELY_WATER_COLORS[part],
                linewidth=0)
        left += rad[f"n_{part}"].to_numpy()
    for yi, row in zip(y, rad.itertuples()):
        ax.annotate(f"{row.n_entirely_water:,} ({row.pct_of_reference:.0f}%)", (row.n_entirely_water, yi),
                    xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_yticks(y, [f"{r:,} m" for r in rad["radius_m"]])
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.set_xlim(0, rad["n_entirely_water"].max() * Y_PAD * 1.15)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Buffer radius")
    ax.set_xlabel(f"Buffers entirely classified as water in {int(rad['end'].iloc[0])}")
    ax.spines[["top", "right"]].set_visible(False)


def make() -> plt.Figure:
    S.apply_style()
    hist, rad, ref = load()
    start = int(rad["start"].iloc[0])
    fig = plt.figure(figsize=(S.mm_to_in(S.WIDTH_2COL_MM), S.mm_to_in(FIG_HEIGHT_MM)))
    ax_a, ax_b = fig.add_axes(AX_A), fig.add_axes(AX_B)
    draw_share(ax_a, hist)
    draw_radius(ax_b, rad)

    # letters outside the panel area (replaced by the journal)
    kw = dict(fontsize=S.FONT_SIZE_PANEL, fontweight="medium", va="top", ha="left")
    fig.text(0.005, 0.99, "(A)", **kw)
    fig.text(0.545, 0.99, "(B)", **kw)
    handles = [Patch(color=S.ENTIRELY_WATER_COLORS["entirely_water_both"],
                     label=f"Already entirely water in {start}"),
               Patch(color=S.ENTIRELY_WATER_COLORS["entirely_water_end"],
                     label=f"Became entirely water after {start}")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.945), ncol=2, frameon=False,
               fontsize=S.FONT_SIZE_SMALL, handlelength=1.4, columnspacing=1.6)
    return fig
