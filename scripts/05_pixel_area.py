"""Pixel-area raster (m²) on the exact MapBiomas Collection 11 grid.

The area of a lon/lat cell depends only on its row's latitude: one value per row
is computed with pyproj.Geod (WGS84) and broadcast across the columns, written
in blocks of 256 rows (streaming).

Usage:
    python scripts/05_pixel_area.py
    python scripts/05_pixel_area.py --test-single-pixel
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.transform import Affine
from rasterio.windows import Window

from sitios_arq.paths import PIXEL_AREA_TIF as OUTPUT_FULL
from sitios_arq.paths import REF_RASTER_C11_1985 as REF_RASTER

OUTPUT_TEST = OUTPUT_FULL.with_name("pixel_area_c11_m2_TEST.tif")
BLOCK_LINES = 256
GEOD = Geod(ellps="WGS84")


def pixel_area_m2(lon_left: float, lat_top: float, dx: float, dy: float) -> float:
    """Geodesic area (WGS84) of one lon/lat pixel. dy < 0 (top-down raster)."""
    lat_bot = lat_top + dy
    lon_right = lon_left + dx
    lons = [lon_left, lon_right, lon_right, lon_left, lon_left]
    lats = [lat_top, lat_top, lat_bot, lat_bot, lat_top]
    area, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(area)


def row_areas(transform: Affine, n_rows: int) -> np.ndarray:
    """Area (m²) of the pixels of each of the first `n_rows` rows of `transform`."""
    dx, dy, lon0, lat0 = transform.a, transform.e, transform.c, transform.f
    return np.array([pixel_area_m2(lon0, lat0 + i * dy, dx, dy) for i in range(n_rows)],
                    dtype=np.float32)


def write_raster(src_path: Path, dst_path: Path, n_rows: int, n_cols: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        t, crs = src.transform, src.crs
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": n_cols, "height": n_rows,
        "count": 1, "crs": crs, "transform": t, "compress": "lzw", "predictor": 3,
        "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        start = 0
        while start < n_rows:
            n = min(BLOCK_LINES, n_rows - start)
            areas = row_areas(Affine(t.a, t.b, t.c, t.d, t.e, t.f + start * t.e), n)
            block = np.ascontiguousarray(np.broadcast_to(areas[:, None], (n, n_cols)))
            dst.write(block[None, :, :], window=Window(0, start, n_cols, n))
            start += n
            print(f"\r  {start}/{n_rows} rows ({100 * start / n_rows:.1f}%)", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-single-pixel", action="store_true",
                    help="1×1 raster (row 0, column 0) for a quick check")
    args = ap.parse_args()

    if not REF_RASTER.exists():
        sys.exit(f"Reference raster not found: {REF_RASTER}")
    with rasterio.open(REF_RASTER) as src:
        width, height = src.width, src.height

    if args.test_single_pixel:
        out, n_rows, n_cols = OUTPUT_TEST, 1, 1
    else:
        out, n_rows, n_cols = OUTPUT_FULL, height, width
    print(f"{n_cols}×{n_rows} pixels → {out}")

    write_raster(REF_RASTER, out, n_rows, n_cols)
    with rasterio.open(out) as dst:
        print(f"Pixel (0,0) = {dst.read(1, window=Window(0, 0, 1, 1))[0, 0]:.4f} m²")


if __name__ == "__main__":
    main()
