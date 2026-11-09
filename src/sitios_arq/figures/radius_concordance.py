"""Fig. S1 — sign concordance of land-cover change with the 100 m radius.

Draws only. Concordance from `summary/radius_sensitivity_concordance.csv`
(scripts/14_summary_tables.py); biome order (descending number of sites, as in Fig 3) from
`summary/sites_per_biome.csv`; period order = `periods` of config/hex.toml.

(A) Matrix. Rows: Brazil (NATIONAL dissolve) and the biomes (dissolve clipped by biome).
Columns: Natural | Anthropic | Water, each with the 100 m reference and the sensitivity
radii. Each cell has one glyph per period: at the reference, an up/down triangle for the
sign of `delta_pp_ref`; at the other radii, a filled square when `sign_agrees` and a square
with an X when not. (B) The discordances (`sign_agrees` false): change at 100 m and at the
discordant radius, over a ±1 % background band. No tolerance in the classification (raw Δ in the table).

Text without "Δ" or "pp": "change" and "%" (no n/4 labels either). Outline and X of the
discordance glyph and the rule under Brazil in `style.THIN_RULE_WIDTH` (exception to the
2 pt rule); the groups of panel B are separated by space, without a rule.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from sitios_arq import hexgrid as H
from sitios_arq import style as S
from sitios_arq.extraction import NATIONAL
from sitios_arq.paths import REPO_ROOT, SUMMARY_DIR

NAME = "radius_concordance"
DESCRIPTION = ("Fig S1: sign of the change in Natural/Anthropic/Water cover at 200, 500 and "
               "1000 m vs the 100 m reference, per unit (Brazil and biomes) and period (A), and the "
               "discordant cases with both changes (B)")
HEX_CONFIG = REPO_ROOT / "config/hex.toml"
CATEGORIES = ["Natural", "Anthropic", "Water"]

# Layout in mm (y downwards); panel A on an axis with 1 unit = 1 mm
MARGIN = 1.0
LABEL_W = 27.0          # column of unit names
GROUP_GAP = 4.0         # between class groups
REF_FRAC = 0.30         # fraction of the group taken by the reference column
GLYPH = 2.0             # glyph side
GLYPH_GAP = 0.55
X_INSET = 0.38          # inset of the X inside the square
BAR_TOP, BAR_H = 1.0, 1.0
CLASS_Y = 2.8           # top of the class name
HEAD_Y = 12.8           # baseline of the radius headers
FIRST_ROW = 16.5        # centre of the Brazil row
SEP_BELOW = 3.6         # rule below the centre of the Brazil row
BIOME_FIRST = 7.4       # from the Brazil centre to the first biome
ROW_PITCH = 5.4
LEGEND_DY = 6.0         # from the last centre to the first legend line
LEGEND_PITCH = 4.6
LEGEND_SEP = "  ·  "

# Panel B
PANEL_GAP = 7.0         # from the end of panel A to the top of panel B (letter)
B_TOP_PAD = 8.0         # from the letter to the top of the axis (legend in between)
B_ROW = 4.6             # mm per row
B_GROUP_GAP = 0.6       # space between groups, in rows
B_BOTTOM = 13.0         # x-axis label
B_LEFT, B_RIGHT = 76.0, 176.0
BAND = 1.0              # half-width of the background band (%)
B_XPAD = 1.15           # x-axis limit / largest |change|
MARKER = 5.5            # pt


def inputs() -> list[Path]:
    return [SUMMARY_DIR / "radius_sensitivity_concordance.csv", SUMMARY_DIR / "sites_per_biome.csv"]


def load():
    conc_path, sites_path = inputs()
    conc = pd.read_csv(conc_path)
    sites = pd.read_csv(sites_path).sort_values("n_sites", ascending=False, kind="stable")
    cfg = H.load_config(HEX_CONFIG)
    periods = [f"{s}–{e}" for s, e in H.periods(cfg)]

    # consistency checks against the tables (nothing is recomputed here)
    units = [NATIONAL, *sites["biome"]]
    if set(conc["unit"]) != set(units):
        raise ValueError(f"Units differ: {sorted(set(conc['unit']) ^ set(units))}")
    if set(conc["category"]) != set(CATEGORIES) or set(conc["period"]) != set(periods):
        raise ValueError("Concordance table must have the three categories and the main periods")
    if conc["reference_radius_m"].nunique() != 1:
        raise ValueError("More than one reference radius")
    if (conc["delta_pp_ref"] == 0).any():
        raise ValueError("Reference change exactly 0: no sign glyph for it")
    if conc.groupby(["unit", "category", "period"])["delta_pp_ref"].nunique().gt(1).any():
        raise ValueError("Reference change differs between radii")
    labels = conc.drop_duplicates("unit").set_index("unit")["unit_en"].loc[units]
    return conc, units, labels, periods


def discordances(conc: pd.DataFrame, units: list[str], periods: list[str]) -> pd.DataFrame:
    """Rows with sign opposite to 100 m, in matrix order (unit, class, radius, period)."""
    d = conc[~conc["sign_agrees"]].copy()
    keys = {"unit": units, "category": CATEGORIES, "period": periods}
    for col, order in keys.items():
        d[f"_{col}"] = d[col].map({v: i for i, v in enumerate(order)})
    return (d.sort_values(["_unit", "_category", "radius_m", "_period"])
            .drop(columns=[f"_{c}" for c in keys]).reset_index(drop=True))


def geometry(n_radii: int, width: float):
    """x0 of each group, width of the reference column and of the others."""
    group_w = (width - 2 * MARGIN - LABEL_W - 2 * GROUP_GAP) / 3
    ref_w = REF_FRAC * group_w
    sub_w = (group_w - ref_w) / n_radii
    x0 = [MARGIN + LABEL_W + i * (group_w + GROUP_GAP) for i in range(3)]
    return x0, group_w, ref_w, sub_w


def glyph_offsets(n: int) -> list[float]:
    span = n * GLYPH + (n - 1) * GLYPH_GAP
    return [-span / 2 + GLYPH / 2 + i * (GLYPH + GLYPH_GAP) for i in range(n)]


def triangle(ax, x: float, y: float, up: bool) -> None:
    h = GLYPH / 2
    tip, base = (y - h, y + h) if up else (y + h, y - h)
    ax.add_patch(Polygon([(x - h, base), (x + h, base), (x, tip)], closed=True,
                         facecolor=S.GREY_MID, edgecolor="none"))


def square(ax, x: float, y: float, agrees: bool) -> None:
    h = GLYPH / 2
    if agrees:
        ax.add_patch(Rectangle((x - h, y - h), GLYPH, GLYPH, facecolor=S.INK, edgecolor="none"))
        return
    ax.add_patch(Rectangle((x - h, y - h), GLYPH, GLYPH, facecolor="none", edgecolor=S.INK,
                           linewidth=S.THIN_RULE_WIDTH))
    d = h - X_INSET
    for sx in (1, -1):
        ax.add_line(Line2D([x - d, x + d], [y - sx * d, y + sx * d], color=S.INK,
                           linewidth=S.THIN_RULE_WIDTH, solid_capstyle="butt"))


def inline(ax, x: float, y: float, items) -> float:
    """Text and glyphs on one line, left to right; returns the final x (mm)."""
    renderer = ax.figure.canvas.get_renderer()
    mm_per_px = 25.4 / ax.figure.dpi
    for it in items:
        if isinstance(it, str):
            t = ax.text(x, y, it, ha="left", va="center", fontsize=S.FONT_SIZE_SMALL, color=S.INK)
            x += t.get_window_extent(renderer).width * mm_per_px
        else:
            kind, arg = it
            (triangle if kind == "tri" else square)(ax, x + GLYPH / 2, y, arg)
            x += GLYPH + 1.0
    return x


def height_a(n_units: int) -> float:
    last = FIRST_ROW + BIOME_FIRST + (n_units - 2) * ROW_PITCH
    return last + LEGEND_DY + LEGEND_PITCH + 3.0


def draw_matrix(ax, conc: pd.DataFrame, units: list[str], labels: pd.Series,
                periods: list[str], width: float) -> float:
    """Draws the matrix and legend in `ax` (coordinates in mm); returns the height used."""
    radii = sorted(conc["radius_m"].unique())
    ref_r = int(conc["reference_radius_m"].iloc[0])
    x0, group_w, ref_w, sub_w = geometry(len(radii), width)
    offs = glyph_offsets(len(periods))
    rows = [FIRST_ROW] + [FIRST_ROW + BIOME_FIRST + i * ROW_PITCH for i in range(len(units) - 1)]

    ref = conc.drop_duplicates(["unit", "category", "period"]).set_index(
        ["unit", "category", "period"])["delta_pp_ref"]
    agree = conc.set_index(["unit", "category", "radius_m", "period"])["sign_agrees"]

    for g, cat in enumerate(CATEGORIES):
        ax.add_patch(Rectangle((x0[g], BAR_TOP), group_w, BAR_H, facecolor=S.CLASS_COLORS[cat],
                               edgecolor="none"))
        ax.text(x0[g] + group_w / 2, CLASS_Y, cat, ha="center", va="top",
                fontsize=S.FONT_SIZE, fontweight="medium")
        cx_ref = x0[g] + ref_w / 2
        ax.text(cx_ref, HEAD_Y, f"{ref_r} m\n(reference)", ha="center", va="bottom",
                fontsize=S.FONT_SIZE_SMALL, color=S.GREY_DARK, linespacing=1.0)
        cxs = [x0[g] + ref_w + (i + 0.5) * sub_w for i in range(len(radii))]
        for cx, r in zip(cxs, radii):
            ax.text(cx, HEAD_Y, f"{r} m", ha="center", va="bottom", fontsize=S.FONT_SIZE_SMALL)

        for u, y in zip(units, rows):
            for dx, p in zip(offs, periods):
                triangle(ax, cx_ref + dx, y, ref[(u, cat, p)] > 0)
            for cx, r in zip(cxs, radii):
                for dx, p in zip(offs, periods):
                    square(ax, cx + dx, y, bool(agree[(u, cat, r, p)]))

    for u, y in zip(units, rows):
        ax.text(MARGIN, y, labels[u], ha="left", va="center", fontsize=S.FONT_SIZE,
                fontweight="medium" if u == NATIONAL else "normal")
    sep = FIRST_ROW + SEP_BELOW
    ax.add_line(Line2D([MARGIN, width - MARGIN], [sep, sep], color=S.GREY_MID,
                       linewidth=S.THIN_RULE_WIDTH))

    y1 = rows[-1] + LEGEND_DY
    y2 = y1 + LEGEND_PITCH
    items = ["Each cell shows four periods, in order:"]
    for p in periods:
        items += ["  ", ("sq", True), p]
    ends = [inline(ax, MARGIN, y1, items),
            inline(ax, MARGIN, y2, [("tri", True), ("tri", False),
                                    f"sign of change at {ref_r} m (increase / decrease)", LEGEND_SEP,
                                    ("sq", True), f"same sign as at {ref_r} m", LEGEND_SEP,
                                    ("sq", False), "opposite sign"])]
    if max(ends) > width - MARGIN:
        raise RuntimeError(f"Legend wider than the figure: {max(ends):.1f} mm")
    return y2 + 3.0


def b_positions(d: pd.DataFrame) -> np.ndarray:
    """y position of each row of panel B, with space between groups (unit, class)."""
    new_group = (~d.duplicated(["unit", "category"])).to_numpy()
    new_group[0] = False
    return np.arange(len(d)) + B_GROUP_GAP * np.cumsum(new_group)


def height_b(d: pd.DataFrame) -> float:
    return B_TOP_PAD + (b_positions(d)[-1] + 1) * B_ROW + B_BOTTOM


def draw_discordances(fig, d: pd.DataFrame, top: float, width: float, height: float) -> None:
    """Panel B: change at 100 m (triangle) and at the discordant radius (square), per row."""
    y = b_positions(d)
    span = y[-1] + 1
    ax_top = top + B_TOP_PAD
    ax = fig.add_axes([B_LEFT / width, 1 - (ax_top + span * B_ROW) / height,
                       (B_RIGHT - B_LEFT) / width, span * B_ROW / height])
    ref_r = int(d["reference_radius_m"].iloc[0])
    lim = B_XPAD * max(d["delta_pp"].abs().max(), d["delta_pp_ref"].abs().max(), BAND)
    ax.axvspan(-BAND, BAND, color=S.PAPER, zorder=0, linewidth=0)
    ax.axvline(0, color=S.REFERENCE_LINE_COLOR, linestyle=S.REFERENCE_LINE_DASH, zorder=1)
    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.delta_pp_ref, r.delta_pp], [yi, yi], color=S.GREY_LIGHT, linewidth=S.LINE_WIDTH,
                solid_capstyle="butt", zorder=2)
        ax.plot(r.delta_pp_ref, yi, marker="^" if r.delta_pp_ref > 0 else "v", color=S.GREY_MID,
                markersize=MARKER, linestyle="none", zorder=3)
        ax.plot(r.delta_pp, yi, marker="s", color=S.INK, markersize=MARKER * 0.85, linestyle="none",
                zorder=3)
    ax.set_yticks(y, [f"{r.radius_m} m, {r.period}" for r in d.itertuples()])
    ax.set_ylim(y[-1] + 0.5, -0.5)
    ax.set_xlim(-lim, lim)
    ax.tick_params(axis="y", length=0, labelsize=S.FONT_SIZE_SMALL)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Change in cover (%)")

    first = ~d.duplicated(["unit", "category"]).to_numpy()
    for yi, r in zip(y[first], d[first].itertuples()):
        fig.text(MARGIN / width, 1 - (ax_top + (yi + 0.5) * B_ROW) / height,
                 f"{r.unit_en}, {r.category}", ha="left", va="center",
                 fontsize=S.FONT_SIZE, fontweight="medium")

    handles = [Line2D([], [], marker="^", color=S.GREY_MID, linestyle="none", markersize=MARKER,
                      label=f"Change at {ref_r} m"),
               Line2D([], [], marker="s", color=S.INK, linestyle="none", markersize=MARKER * 0.85,
                      label="Change at the discordant radius"),
               Patch(facecolor=S.PAPER, edgecolor="none", label=f"Change within ±{BAND:g}%")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
              fontsize=S.FONT_SIZE_SMALL, handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)


def mm_axes(fig, width: float, height: float, top: float = 0.0):
    """Frameless axis with 1 unit = 1 mm, spanning `height` mm from `top`."""
    fh = fig.get_figheight() * 25.4
    ax = fig.add_axes([0, 1 - (top + height) / fh, 1, height / fh])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return ax


def make() -> plt.Figure:
    S.apply_style()
    conc, units, labels, periods = load()
    d = discordances(conc, units, periods)
    w = S.WIDTH_2COL_MM
    h_a = height_a(len(units))
    top_b = h_a + PANEL_GAP
    h = top_b + height_b(d)
    fig = plt.figure(figsize=(S.mm_to_in(w), S.mm_to_in(h)))

    ax_a = mm_axes(fig, w, h_a)
    if abs(draw_matrix(ax_a, conc, units, labels, periods, w) - h_a) > 1e-6:
        raise RuntimeError("Panel A layout height mismatch")
    draw_discordances(fig, d, top_b, w, h)

    # letters outside the panel area (replaced by the journal)
    kw = dict(fontsize=S.FONT_SIZE_PANEL, fontweight="medium", va="top", ha="left")
    fig.text(MARGIN / w, 1 - 0.5 / h, "(A)", **kw)
    fig.text(MARGIN / w, 1 - top_b / h, "(B)", **kw)
    return fig
