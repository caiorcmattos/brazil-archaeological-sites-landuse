"""Composition alluvial (two years, three classes), used by Figs 2A and 3. Drawing only.

Ribbons link each class to itself and do not represent pixel transitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from sitios_arq import style as S

CATEGORIES = ["Natural", "Anthropic", "Water"]   # bottom to top in the bars
RIBBON_TINT = 0.35
CURVE = 0.4             # Bézier handle length, fraction of the distance between bars
MIN_INSIDE = 0.06       # minimum mean thickness for the class label to go inside the ribbon
THIN = 0.15             # below this, an asymmetric ribbon has its label shifted to the thick side
NUDGE = 0.035


def label_color(c: str) -> str:
    return S.INK if c == "Anthropic" else S.CLASS_COLORS[c]


def label_position(a0, a1, b0, b1, x0, x1, k, nudge_thin):
    """Position (x, y, va) of the class label; with `nudge_thin`, thin ribbons go above the panel."""
    ta, tb = a1 - a0, b1 - b0
    if not nudge_thin:
        return 0.5, ((a0 + a1) / 2 + (b0 + b1) / 2) / 2, "center"
    if (ta + tb) / 2 < MIN_INSIDE:
        return 0.5, 1.02, "bottom"
    x = 0.5
    if (ta + tb) / 2 < THIN and abs(ta - tb) > 0.03:
        x += NUDGE if tb > ta else -NUDGE
    # y on the ribbon midline at x (the Bézier x(t) is the same on both edges)
    t = np.linspace(0, 1, 501)
    bx = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * (x0 + k) + 3 * (1 - t) * t ** 2 * (x1 - k) + t ** 3 * x1
    s = np.interp(x, bx, 3 * (1 - t) * t ** 2 + t ** 3)
    return x, ((a0 + a1) / 2) * (1 - s) + ((b0 + b1) / 2) * s, "center"


def draw_alluvial(ax, pct: pd.DataFrame, *, xl: float = 0.27, xr: float = 0.73, w: float = 0.12,
                  pct_fmt: str = "{:.1f}%", year_size: float = S.FONT_SIZE, label_pad: float = 0.03,
                  nudge_thin: bool = False) -> None:
    """`pct`: fractions (0–1), index = two years, columns = CATEGORIES."""
    years = list(pct.index)
    pos = {}
    for yr in years:
        y = 0.0
        for c in CATEGORIES:
            pos[(yr, c)] = (y, y + pct.loc[yr, c])
            y += pct.loc[yr, c]

    x0, x1 = xl + w / 2, xr - w / 2
    k = CURVE * (x1 - x0)
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO] + [MplPath.CURVE4] * 3 \
        + [MplPath.CLOSEPOLY]
    for c in CATEGORIES:
        (a0, a1), (b0, b1) = pos[(years[0], c)], pos[(years[1], c)]
        verts = [(x0, a0), (x0 + k, a0), (x1 - k, b0), (x1, b0), (x1, b1), (x1 - k, b1),
                 (x0 + k, a1), (x0, a1), (x0, a0)]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=S.tint(S.CLASS_COLORS[c], RIBBON_TINT),
                               edgecolor="none", zorder=1))
        x, y, va = label_position(a0, a1, b0, b1, x0, x1, k, nudge_thin)
        ax.text(x, y, c, ha="center", va=va, fontsize=S.FONT_SIZE_SMALL,
                color=label_color(c), zorder=3)

    for yr, x, side in ((years[0], xl, -1), (years[1], xr, 1)):
        for c in CATEGORIES:
            y0, y1 = pos[(yr, c)]
            ax.add_patch(Rectangle((x - w / 2, y0), w, y1 - y0, facecolor=S.CLASS_COLORS[c],
                                   edgecolor="white", linewidth=0.6, zorder=2))
            ax.text(x + side * (w / 2 + label_pad), (y0 + y1) / 2, pct_fmt.format(100 * (y1 - y0)),
                    ha="left" if side > 0 else "right", va="center", fontsize=S.FONT_SIZE_SMALL,
                    color=label_color(c))
        ax.text(x, 1.02, str(yr), ha="center", va="bottom", fontsize=year_size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.axis("off")
