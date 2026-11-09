"""Build manuscript figures from processed tables (no analysis here).

    python scripts/make_figures.py --fig sites_biomes
    python scripts/make_figures.py --fig 1
    python scripts/make_figures.py --all

Figure modules live in `src/sitios_arq/figures/`; manuscript numbers map to
module names in `FIGURE_NUMBERS`. Each figure is written as PDF (vector master),
TIFF (RGB, submission) and PNG (preview); see `sitios_arq.figures.export`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sitios_arq.figures import FIGURE_NUMBERS, FIGURES  # noqa: E402
from sitios_arq.figures.export import save_figure  # noqa: E402
from sitios_arq.paths import FIGURES_DIR  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # font subsetting in the PDF
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fig", action="append", help="module name or manuscript number (repeatable)")
    g.add_argument("--all", action="store_true")
    p.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = p.parse_args()

    names = list(FIGURES) if args.all else [FIGURE_NUMBERS.get(f, f) for f in args.fig]
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        p.error(f"unknown figure(s): {unknown}; available: {sorted(FIGURES)} / numbers {sorted(FIGURE_NUMBERS)}")

    for name in names:
        module = FIGURES[name]
        missing = [f for f in module.inputs() if not Path(f).exists()]
        if missing:
            p.error(f"{name}: missing input(s): {', '.join(map(str, missing))}")
        fig = module.make()
        save_figure(fig, Path(args.out_dir), name, description=module.DESCRIPTION)
        plt.close(fig)


if __name__ == "__main__":
    main()
