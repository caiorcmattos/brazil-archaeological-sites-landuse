"""Single style for the manuscript figures: palettes, typography, sizes and lines.

No figure module defines its own colors or fonts; everything comes from here, so
that a biome or a class always has the same color in every figure.

Journal rules (Frontiers figure guidelines):
- width of 85 mm (1 column) or 180 mm (2 columns), height of at most one page;
- text of at least 8 pt at final size;
- lines of at least 2 pt and solid (no dashes);
- 300 dpi at final size, RGB mode; TIFF/JPEG/EPS;
- panel letters (A), (B)… outside the image (the journal replaces them in typesetting).

"Pigmentos" biome palette: validated under simulated deuteranopia,
protanopia and tritanopia (Machado et al. 2009), with a minimum ΔE (CIE76) of 13
between biome pairs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

from sitios_arq.paths import REPO_ROOT

FONTS_DIR = REPO_ROOT / "assets/fonts"
FONT_FAMILY = "IBM Plex Sans"

# ---------------------------------------------------------------------------
# Biomes (names as they appear in the data: column `biome` / `Bioma`)
# ---------------------------------------------------------------------------

BIOME_COLORS = {
    "Amazônia": "#1F5C45",        # jade
    "Mata Atlântica": "#3E9C94",  # verdigris
    "Caatinga": "#C4633D",        # terracotta
    "Cerrado": "#D3A12B",         # ochre
    "Pampa": "#8C3B6E",           # plum
    "Pantanal": "#78A6D2",        # slate blue
}

BIOME_LABELS_EN = {
    "Amazônia": "Amazon",
    "Mata Atlântica": "Atlantic Forest",
    "Caatinga": "Caatinga",
    "Cerrado": "Cerrado",
    "Pampa": "Pampa",
    "Pantanal": "Pantanal",
}

# Fixed order for legends and panels not sorted by value.
BIOME_ORDER = ["Amazônia", "Mata Atlântica", "Caatinga", "Cerrado", "Pampa", "Pantanal"]

# ---------------------------------------------------------------------------
# Thematic classes and scales
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "Natural": "#2F7D6B",
    "Anthropic": "#C9A27E",
    "Water": "#1F4E8C",
}

# Change in anthropic cover: recovery (teal) → neutral → gain (clay).
DIVERGING_ANTHROPIC = ["#1E5F55", "#4E9A8C", "#A9CFC5", "#E6E3DD", "#E2B48F", "#C2703F", "#7E3A1C"]

# Classed anthropic change (pp; hexagon maps of Fig 2): 8 classes centred on 0, with
# no neutral class and asymmetric (gain goes beyond 50 pp). The extremes (< −10 and > 50)
# are the colorbar arrows (extend="both"). Derived from DIVERGING_ANTHROPIC with
# re-spaced lightness and two extra clay tones; validated under deuteranopia, protanopia
# and tritanopia (Machado et al. 2009): minimum ΔE (CIE76) of 13.5 between adjacent
# classes (protanopia, −3–0 × 0–3).
ANTHROPIC_CHANGE_BOUNDS = [-10, -3, 0, 3, 10, 25, 50]
ANTHROPIC_CHANGE_COLORS = [
    "#1E5F55",  # < −10
    "#4E9A8C",  # −10 to −3
    "#98C4B8",  # −3 to 0
    "#F2DCC8",  # 0 to 3
    "#E2B48F",  # 3 to 10
    "#D08A5C",  # 10 to 25
    "#A85A30",  # 25 to 50
    "#7E3A1C",  # > 50
]

# Sequential water scale (light → dark).
SEQUENTIAL_FLOODING = ["#C9DCEB", "#7FA8CF", "#3A6EA8", "#13305E"]

# Buffers entirely classified as water (Fig 4): already at the start of the series
# (reservoirs older than 1985) and only at the end. Two SEQUENTIAL_FLOODING steps far
# apart in lightness, distinguishable without relying on hue (no yellow; does not
# clash with the biome colors).
ENTIRELY_WATER_COLORS = {
    "entirely_water_both": SEQUENTIAL_FLOODING[1],
    "entirely_water_end": SEQUENTIAL_FLOODING[3],
}

INK = "#1A1A1A"
GREY_DARK = "#6B6B6B"
GREY_MID = "#9A968F"
GREY_LIGHT = "#D9D6D0"
PAPER = "#F4F2EE"

# ---------------------------------------------------------------------------
# Sizes, typography and lines
# ---------------------------------------------------------------------------

WIDTH_1COL_MM = 85
WIDTH_2COL_MM = 180

FONT_SIZE = 8.5          # base
FONT_SIZE_SMALL = 8.0    # journal minimum
FONT_SIZE_PANEL = 10.0   # letters (A), (B)

LINE_WIDTH = 2.0         # journal minimum, in pt

# Exception to the 2 pt rule: internal biome boundaries (background cartographic
# feature). At 2 pt, the intricate IBGE 1:250,000 lines become wide bands (0.7 mm ≈ 30 km
# on the map) that hide the points; generalizing at 4 km did not solve it. The national
# outline, axes and ticks stay at 2 pt.
BOUNDARY_WIDTH = 0.9
BOUNDARY_COLOR = GREY_MID

# Exception to the solid-line rule (Fig 5): zero reference lines in the change plots
# are dashed (still 2 pt). Pattern in multiples of the line width (matplotlib scales the
# dashes by the width).
REFERENCE_LINE_DASH = (0, (3, 2))
REFERENCE_LINE_COLOR = GREY_MID

# Exception to the 2 pt rule (radius concordance matrix): outline and X of the
# disagreement glyph (~2 mm square) and the rule separating Brazil from the biomes. At
# 2 pt the X fills the glyph and it can no longer be told apart from the solid square.
THIN_RULE_WIDTH = 0.8

# Bubbles (Fig 5): white edge separates overlapping bubbles; size legend in grey.
BUBBLE_EDGE_COLOR = "#FFFFFF"
BUBBLE_EDGE_WIDTH = 0.5  # pt
BUBBLE_LEGEND_COLOR = GREY_LIGHT

# White halo on labels that cross reference lines (pt).
TEXT_HALO_WIDTH = 2.0
TEXT_HALO_COLOR = "#FFFFFF"

# Site points on maps: a white halo makes them stand out on dark biomes
# (Amazônia, Pampa) and separates overlapping points without darkening dense areas.
POINT_SIZE = 3.2         # pt²
POINT_EDGE_WIDTH = 0.25  # pt
POINT_EDGE_COLOR = "#FFFFFF"

DPI_RASTER = 600         # TIFF/PNG and rasterized parts of the PDF (≥ 300 required)


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def tint(hex_color: str, amount: float) -> str:
    """Mix `hex_color` with white; `amount` = fraction of the original color (0–1)."""
    rgb = mpl.colors.to_rgb(hex_color)
    return mpl.colors.to_hex([1 - amount * (1 - c) for c in rgb])


def register_fonts(fonts_dir: Path = FONTS_DIR) -> None:
    for ttf in sorted(fonts_dir.glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    if FONT_FAMILY not in {f.name for f in font_manager.fontManager.ttflist}:
        raise FileNotFoundError(f"Font {FONT_FAMILY!r} not found in {fonts_dir}")


def apply_style() -> None:
    """Set matplotlib rcParams for the manuscript figures."""
    register_fonts()
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE_SMALL,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE_SMALL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": LINE_WIDTH,
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,  # embedded TrueType: editable text in the PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
