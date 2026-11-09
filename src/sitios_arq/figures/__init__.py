"""Manuscript figures. Each module exposes `NAME`, `DESCRIPTION`, `inputs()` and `make()`.

Modules are named by content; the figure number in the manuscript is only in
`FIGURE_NUMBERS`.
"""

from sitios_arq.figures import (anthropic_water_change, landcover_change, landcover_change_biomes,
                                radius_concordance, reservoir_sensitivity, reservoir_water, sites_biomes)

FIGURES = {m.NAME: m for m in (sites_biomes, landcover_change, landcover_change_biomes,
                               reservoir_water, anthropic_water_change, radius_concordance,
                               reservoir_sensitivity)}

FIGURE_NUMBERS = {
    "1": "sites_biomes",
    "2": "landcover_change",
    "3": "landcover_change_biomes",
    "4": "reservoir_water",
    "5": "anthropic_water_change",
    "S1": "radius_concordance",
    "S2": "reservoir_sensitivity",
}
