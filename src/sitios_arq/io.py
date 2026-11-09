"""Writing of tabular products."""

from __future__ import annotations

from pathlib import Path


def save(obj, out_path, **save_kwargs) -> Path:
    """Write a DataFrame or GeoDataFrame, creating the parent directory.

    `.parquet` → `to_parquet`; `.csv` → `to_csv(index=False)`; any other suffix of a
    GeoDataFrame → `to_file` (e.g. `.gpkg`, with `layer=`).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        obj.to_parquet(out_path, **save_kwargs)
    elif out_path.suffix == ".csv":
        obj.to_csv(out_path, **({"index": False} | save_kwargs))
    elif type(obj).__name__ == "GeoDataFrame":
        obj.to_file(out_path, **save_kwargs)
    else:
        raise TypeError(f"Cannot write {type(obj).__name__} to {out_path.suffix}")
    return out_path
