"""Thematic aggregation of MapBiomas C11 classes (Natural / Anthropic / Water / No data).

The legend lives in `config/mapbiomas_legend_c11.csv` (code, name_pt, name_en,
category), from the MapBiomas "Códigos das classes da legenda da Coleção 11"
table (13/08/2026, `docs/mapbiomas_c11_legenda_codigos.pdf`). Class 0 (no
data) is not in that table: it is the raster value outside the MapBiomas mask
(sea and, along the borders, neighbouring territory) or of pixels not observed.

Class 0 rule: class 0 belongs to no thematic category and
is left out of the denominator. Proportion = category area / (total area −
class 0 area).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sitios_arq.extraction import NATIONAL

# Category label in the legend → column slug in the products.
CATEGORIES = {"Natural": "natural", "Anthropic": "anthropic", "Water": "water", "No data": "nodata"}
WITH_DATA = ("natural", "anthropic", "water")


def load_legend(path: str | Path) -> pd.DataFrame:
    """Read and validate the C11 legend: unique codes, known categories, 0 is 'No data'."""
    legend = pd.read_csv(path, dtype={"code": "int64"})
    dup = legend.loc[legend["code"].duplicated(), "code"].tolist()
    if dup:
        raise ValueError(f"duplicated codes in legend {path}: {dup}")
    unknown = sorted(set(legend["category"]) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categories in legend {path}: {unknown}")
    nodata = legend.loc[legend["category"] == "No data", "code"].tolist()
    if nodata != [0]:
        raise ValueError(f"'No data' must be exactly code 0 in legend {path}, got {nodata}")
    return legend


def category_of(legend: pd.DataFrame) -> dict[int, str]:
    """code → category slug."""
    return {int(c): CATEGORIES[cat] for c, cat in zip(legend["code"], legend["category"])}


def aggregate_categories(df: pd.DataFrame, legend: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Long table (keys + mapbiomas_class + area_m2) → one row per key with category areas.

    Columns: keys, area_{natural,anthropic,water,nodata}_m2, area_total_m2,
    area_with_data_m2, prop_{natural,anthropic,water}. Proportions are over
    the area with data; NaN where that area is 0. Raises if any class code
    has no mapping in the legend.
    """
    mapping = category_of(legend)
    codes = df["mapbiomas_class"].astype("int64")
    unmapped = sorted(set(codes.unique()) - set(mapping))
    if unmapped:
        raise KeyError(f"MapBiomas codes without a thematic category in the legend: {unmapped}")

    cat = codes.map(mapping)
    wide = (df.assign(category=cat).groupby(keys + ["category"])["area_m2"].sum()
            .unstack("category", fill_value=0.0)
            .reindex(columns=list(CATEGORIES.values()), fill_value=0.0))
    wide.columns = [f"area_{c}_m2" for c in wide.columns]
    out = wide.reset_index()

    out["area_total_m2"] = df.groupby(keys)["area_m2"].sum().reindex(
        pd.MultiIndex.from_frame(out[keys])).to_numpy()
    out["area_with_data_m2"] = out["area_total_m2"] - out["area_nodata_m2"]
    denom = out["area_with_data_m2"].where(out["area_with_data_m2"] > 0)
    for c in WITH_DATA:
        out[f"prop_{c}"] = out[f"area_{c}_m2"] / denom
    return out


# ---------------------------------------------------------------------------
# Checks (used by 09_thematic_aggregation.py)
# ---------------------------------------------------------------------------

def closure_rel(out: pd.DataFrame) -> pd.Series:
    """(Σ four category areas − total area) / total area."""
    s = out[[f"area_{c}_m2" for c in CATEGORIES.values()]].sum(axis=1)
    return (s - out["area_total_m2"]) / out["area_total_m2"]


def national_vs_biomes(out: pd.DataFrame) -> pd.DataFrame:
    """Per (radius, year, category): Σ biome areas vs NATIONAL, relative difference."""
    cols = [f"area_{c}_m2" for c in CATEGORIES.values()]
    g = out.assign(group=np.where(out["unit"].eq(NATIONAL), "national", "biomes"))
    p = g.groupby(["radius_m", "year", "group"])[cols].sum()
    nat, bio = p.xs("national", level="group"), p.xs("biomes", level="group")
    diff = (bio - nat).abs()
    # A category absent everywhere (e.g. nodata at 100 m) gives 0/0: no difference.
    rel = (diff / nat.where(nat > 0)).where(diff > 0, 0.0)
    return rel.melt(ignore_index=False, var_name="category", value_name="rel_diff").reset_index()


def class0_site_summary(out: pd.DataFrame) -> pd.DataFrame:
    """Per (radius, year): sites with class 0 > 0 and the distribution of their class 0 share.

    Also, per radius, the number of sites with class 0 in at least one year
    (`n_sites_any_year`), the stable subset to drop in with/without analyses.
    """
    d = out.assign(nodata_frac=out["area_nodata_m2"] / out["area_total_m2"])
    d["has_nodata"] = d["area_nodata_m2"] > 0
    rows = []
    for (r, y), g in d.groupby(["radius_m", "year"]):
        f = g.loc[g["has_nodata"], "nodata_frac"]
        rows.append({
            "radius_m": r, "year": y, "n_sites": len(g), "n_sites_nodata": int(g["has_nodata"].sum()),
            "share_sites_nodata": g["has_nodata"].mean(),
            "nodata_frac_median": f.median() if len(f) else np.nan,
            "nodata_frac_p90": f.quantile(0.9) if len(f) else np.nan,
            "nodata_frac_max": f.max() if len(f) else np.nan,
            "n_sites_nodata_gt_0.1": int((f > 0.1).sum()),
            "n_sites_nodata_gt_0.5": int((f > 0.5).sum()),
            "nodata_share_of_total_area": g["area_nodata_m2"].sum() / g["area_total_m2"].sum(),
        })
    s = pd.DataFrame(rows)
    any_year = d.groupby(["radius_m", "id_bem"])["has_nodata"].any().groupby("radius_m").sum()
    s["n_sites_any_year"] = s["radius_m"].map(any_year).astype("int64")
    return s
