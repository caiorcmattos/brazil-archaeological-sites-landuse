"""Figure export in three formats.

- PDF (vector master); rasterized parts at `DPI_RASTER`;
- TIFF RGB without alpha channel, LZW, at `DPI_RASTER` (journal submission format);
- PNG preview at 300 dpi.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

from sitios_arq.style import DPI_RASTER

log = logging.getLogger("figures")

PREVIEW_DPI = 300


def save_figure(fig, out_dir: Path, name: str, *, description: str = "") -> None:
    """Save `<name>.pdf`, `<name>.tif` and `<name>.png` in `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", dpi=DPI_RASTER,
                metadata={"Title": name, "Subject": description})

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI_RASTER)
    buf.seek(0)
    rgb = Image.open(buf).convert("RGB")
    rgb.save(out_dir / f"{name}.tif", compression="tiff_lzw", dpi=(DPI_RASTER, DPI_RASTER))

    scale = PREVIEW_DPI / DPI_RASTER
    rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS).save(
        out_dir / f"{name}.png", dpi=(PREVIEW_DPI, PREVIEW_DPI))

    log.info("%s: %s.{pdf,tif,png} (%d×%d px at %d dpi)", name, out_dir / name,
             rgb.width, rgb.height, DPI_RASTER)
