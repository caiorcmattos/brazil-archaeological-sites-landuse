"""Filtering of the IPHAN database to continental Brazil: metrics, flags and status.

Steps (every rule runs on every site):

1. `compute_metrics()` — spatial (EPSG:5880) and text metrics.
2. `build_flags()` — hard and soft flags from metrics and config.
3. `apply_decisions()` — status (`removed` / `pending_review` / `kept`) and
   `primary_reason`; manual decisions override rules in both directions.
4. `apply_exclusions()` — data-availability exclusions (e.g. no MapBiomas
   data in the buffer) override rules and manual decisions.

Parameters come from `config/filtering.toml`.

No coordinate is ever corrected. `swapped_coord` only flags the site.

Distances are computed in EPSG:5880 (Brazil Polyconic), which is not
equidistant: against the geodesic distance (GRS80), `dist_coast_m` overestimates
by 0.9% in median and up to 6.9%, and `dist_border_m` by up to 1.4%.
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

CRS_METRIC = 5880
CRS_GEOGRAPHIC = 4674

HARD_FLAGS = [
    "invalid_coord",
    "swapped_coord",
    "outside_brazil",
    "offshore_far",
    "shipwreck_text",
    "outside_mapbiomas",
]
SOFT_FLAGS = ["maritime_island", "offshore_near", "near_border"]

VALID_DECISIONS = {"keep", "remove"}
DECISION_COLUMNS = ["id_bem", "decision", "reviewer", "date"]
EXCLUSION_COLUMNS = ["id_bem", "reason", "reviewer", "date"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Read the TOML parameter file. Each top-level table is one parameter (description + value)."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for name, p in cfg.items():
        if not isinstance(p, dict) or "value" not in p:
            raise ValueError(f"Parameter '{name}' needs a 'value'")
    for order, flags in (("hard_order", HARD_FLAGS), ("soft_order", SOFT_FLAGS)):
        if sorted(cfg[order]["value"]) != sorted(flags):
            raise ValueError(f"'{order}' must contain exactly {flags}")
    if cfg["hard_order"]["value"][-1] != "outside_mapbiomas":
        raise ValueError("'outside_mapbiomas' must be the last entry of 'hard_order'")
    return cfg


def values(cfg: dict) -> dict:
    """{name: value} — used by the rules."""
    return {k: v["value"] for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def normalize_text(s) -> str:
    """No accents, lowercase, hyphen becomes space, whitespace collapsed."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    """A trailing `*` makes a prefix match; otherwise whole word/phrase."""
    out = []
    for term in terms:
        prefix = term.endswith("*")
        base = normalize_text(term.rstrip("*"))
        pattern = r"\b" + re.escape(base) + (r"\w*" if prefix else r"\b")
        out.append((term, re.compile(pattern)))
    return out


def find_terms(normalized_text: str, compiled_terms) -> str:
    """Matched terms (as written in the config), joined by `|`."""
    return "|".join(t for t, rx in compiled_terms if rx.search(normalized_text))


# ---------------------------------------------------------------------------
# Spatial references
# ---------------------------------------------------------------------------

@dataclass
class References:
    """Reference layers in EPSG:5880.

    brazil             : Brazil polygon (BC250 `lml_pais_a`)
    land               : brazil ∪ every island of `hid_ilha_a`
    neighbors          : other countries of `lml_pais_a` (column `name`)
    coastline          : outline of `land` minus land-border stretches
    border             : Brazil's land-border lines
    islands            : `hid_ilha_a` with `name`, `island_type`, `area_km2`
    municipalities     : `lml_municipio_a` with `geocode` (optional)
    biomes             : IBGE biomes with `Bioma` (optional)
    """

    brazil: object
    land: object
    neighbors: gpd.GeoDataFrame
    coastline: gpd.GeoDataFrame
    border: gpd.GeoDataFrame
    islands: gpd.GeoDataFrame
    municipalities: gpd.GeoDataFrame | None = None
    biomes: gpd.GeoDataFrame | None = None

    def __post_init__(self):
        shapely.prepare(self.brazil)
        shapely.prepare(self.land)


def _segments(lines) -> np.ndarray:
    """Split (Multi)LineStrings into 2-vertex segments."""
    parts = shapely.get_parts(np.atleast_1d(lines))
    segs = []
    for ln in parts:
        c = shapely.get_coordinates(ln)
        if len(c) >= 2:
            segs.append(shapely.linestrings(np.stack([c[:-1], c[1:]], axis=1)))
    return np.concatenate(segs) if segs else np.array([], dtype=object)


def _chunk(lines, n_vertices: int = 200) -> np.ndarray:
    """Merge segments and cut into pieces of up to `n_vertices` (efficient spatial index)."""
    if len(lines) == 0:
        return np.array([], dtype=object)
    merged = shapely.line_merge(shapely.multilinestrings(lines))
    out = []
    for ln in shapely.get_parts(merged):
        c = shapely.get_coordinates(ln)
        for i in range(0, len(c) - 1, n_vertices - 1):
            out.append(shapely.linestrings(c[i : i + n_vertices]))
    return np.array(out, dtype=object)


def derive_lines(brazil, land, neighbor_geoms, eps_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Coastline and land-border lines.

    A segment is border if it lies within `eps_m` of a neighbor's outline.
    Coastline = segments of the `land` outline that are not border;
    border = segments of the `brazil` outline that are.
    """
    neighbor_edges = shapely.get_parts(shapely.boundary(np.asarray(neighbor_geoms)))
    neighbor_edges = _segments(neighbor_edges)
    tree = shapely.STRtree(neighbor_edges)

    def near_neighbor(segs):
        idx, _ = tree.query(segs, predicate="dwithin", distance=eps_m)
        m = np.zeros(len(segs), dtype=bool)
        m[np.unique(idx)] = True
        return m

    land_segs = _segments(shapely.boundary(land))
    brazil_segs = _segments(shapely.boundary(brazil))
    coastline = land_segs[~near_neighbor(land_segs)]
    border = brazil_segs[near_neighbor(brazil_segs)]
    return _chunk(coastline), _chunk(border)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_TO_METRIC = Transformer.from_crs(CRS_GEOGRAPHIC, CRS_METRIC, always_xy=True)


def _min_distance(points: np.ndarray, lines: gpd.GeoDataFrame) -> np.ndarray:
    d = np.full(len(points), np.nan)
    if len(points) == 0 or len(lines) == 0:
        return d
    tree = shapely.STRtree(lines.geometry.values)
    (ip, _), dist = tree.query_nearest(points, return_distance=True, all_matches=False)
    d[ip] = dist
    return d


def _join_attribute(points: gpd.GeoDataFrame, layer, column: str) -> pd.Series:
    """First value of `column` from the feature containing each point."""
    if layer is None:
        return pd.Series(None, index=points.index, dtype=object)
    j = gpd.sjoin(points[["geometry"]], layer[[column, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j[column].reindex(points.index)


def _island_of_point(points: gpd.GeoDataFrame, islands: gpd.GeoDataFrame, flagged_types) -> pd.DataFrame:
    """Island containing each point; prefers flagged types and, among them, the smallest."""
    cols = ["name", "island_type", "area_km2", "geometry"]
    j = gpd.sjoin(points[["geometry"]], islands[cols], how="inner", predicate="within")
    j["_prio"] = ~j["island_type"].isin(flagged_types)
    j = j.sort_values(["_prio", "area_km2"])
    j = j[~j.index.duplicated(keep="first")]
    out = pd.DataFrame(index=points.index)
    out["island_name"] = j["name"].reindex(points.index)
    out["island_type"] = j["island_type"].reindex(points.index)
    out["island_area_km2"] = j["area_km2"].reindex(points.index)
    out["maritime_island"] = out["island_type"].isin(flagged_types)
    return out


def compute_metrics(
    sites: gpd.GeoDataFrame,
    refs: References,
    cfg: dict,
    mapbiomas_bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Per-site metrics.

    `sites` needs `id_bem`, `lon`, `lat` (EPSG:4674, raw), `co_iphan`,
    `identifica`, `sintese_be`. `mapbiomas_bounds` = (left, bottom, right,
    top) in degrees, read from the C11 raster metadata.
    """
    v = values(cfg)
    m = pd.DataFrame(index=sites.index)
    m["id_bem"] = sites["id_bem"].values
    lon = sites["lon"].to_numpy(dtype=float)
    lat = sites["lat"].to_numpy(dtype=float)
    m["lon"], m["lat"] = lon, lat

    # --- validity -----------------------------------------------------------
    invalid = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90)
    m["invalid_coord"] = invalid
    ok = ~invalid

    x = np.full(len(m), np.nan)
    y = np.full(len(m), np.nan)
    x[ok], y[ok] = _TO_METRIC.transform(lon[ok], lat[ok])
    pts = shapely.points(x, y)
    pts[invalid] = None
    gpts = gpd.GeoDataFrame(index=m.index, geometry=pts, crs=CRS_METRIC)
    valid = gpts[ok]

    # --- coast side ---------------------------------------------------------
    in_brazil = np.zeros(len(m), dtype=bool)
    in_land = np.zeros(len(m), dtype=bool)
    in_brazil[ok] = shapely.contains_xy(refs.brazil, x[ok], y[ok])
    in_land[ok] = shapely.contains_xy(refs.land, x[ok], y[ok])

    country = _join_attribute(valid, refs.neighbors, "name").reindex(m.index)
    in_neighbor = country.notna().to_numpy() & ~in_brazil

    side = np.where(in_brazil, "land",
           np.where(in_neighbor, "other_country",
           np.where(in_land, "land", "sea"))).astype(object)
    side[invalid] = None
    m["inside_brazil"] = pd.array(np.where(ok, in_brazil, pd.NA), dtype="boolean")
    m["coast_side"] = side
    m["spatial_country"] = country.where(in_neighbor, None)

    # --- swapped coordinate (lat/lon) --------------------------------------
    swapped = np.zeros(len(m), dtype=bool)
    cand = ok & (side != "land") & (np.abs(lat) <= 180) & (np.abs(lon) <= 90)
    if cand.any():
        xs, ys = _TO_METRIC.transform(lat[cand], lon[cand])
        swapped[cand] = shapely.contains_xy(refs.brazil, xs, ys)
    m["swapped_coord"] = swapped

    # --- distances ----------------------------------------------------------
    m["dist_coast_m"] = np.nan
    m["dist_border_m"] = np.nan
    m.loc[ok, "dist_coast_m"] = _min_distance(valid.geometry.values, refs.coastline)
    m.loc[ok, "dist_border_m"] = _min_distance(valid.geometry.values, refs.border)

    # --- islands ------------------------------------------------------------
    island = _island_of_point(valid, refs.islands, v["island_types_flagged"]).reindex(m.index)
    island["maritime_island"] = island["maritime_island"].eq(True)
    m = m.join(island)

    # --- municipality and biome --------------------------------------------
    co = sites["co_iphan"].astype("string")
    code = co.str.slice(2, 9)
    m["declared_mun"] = code.where(code.str.fullmatch(r"\d{7}"), None)
    m["spatial_mun"] = _join_attribute(valid, refs.municipalities, "geocode").reindex(m.index)
    m["biome"] = _join_attribute(valid, refs.biomes, "Bioma").reindex(m.index)

    # --- MapBiomas (collection bounds only) ---------------------------------
    left, bottom, right, top = mapbiomas_bounds
    in_mb = (lon >= left) & (lon <= right) & (lat >= bottom) & (lat <= top)
    m["inside_mapbiomas"] = pd.array(np.where(ok, in_mb, pd.NA), dtype="boolean")

    # --- text ---------------------------------------------------------------
    text = sites["sintese_be"].map(normalize_text) + " " + sites["identifica"].map(normalize_text)
    for key in ("shipwreck_terms", "ambiguous_terms"):
        rx = compile_terms(v[key])
        m[key] = text.map(lambda t: find_terms(t, rx))

    return m


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def build_flags(m: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One boolean column per flag (`flag_<name>`)."""
    v = values(cfg)
    side = m["coast_side"]
    dc = m["dist_coast_m"]
    db = m["dist_border_m"]
    at_sea = side.eq("sea")
    abroad = side.eq("other_country")

    f = pd.DataFrame(index=m.index)
    f["invalid_coord"] = m["invalid_coord"].astype(bool)
    f["swapped_coord"] = m["swapped_coord"].astype(bool)
    f["outside_brazil"] = abroad & (db > v["border_tolerance_m"])
    f["offshore_far"] = at_sea & (dc > v["offshore_hard_m"])
    f["shipwreck_text"] = m["shipwreck_terms"].fillna("").ne("")
    f["outside_mapbiomas"] = m["inside_mapbiomas"].eq(False).fillna(False).astype(bool)
    # Small islands only: sites on islands at or above the threshold are kept.
    f["maritime_island"] = m["maritime_island"].astype(bool) & (m["island_area_km2"] < v["max_island_area_km2"])
    f["offshore_near"] = at_sea & (dc > v["coast_tolerance_m"]) & (dc <= v["offshore_hard_m"])
    f["near_border"] = abroad & (db <= v["border_tolerance_m"])
    return f.add_prefix("flag_").astype(bool)


# ---------------------------------------------------------------------------
# Manual decisions and status
# ---------------------------------------------------------------------------

def load_decisions(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"id_bem": "Int64"}, keep_default_na=False, na_values=[""])
    missing = set(DECISION_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f"Decisions file missing columns: {sorted(missing)}")
    d["decision"] = d["decision"].astype("string").str.strip().str.lower()
    bad = d.loc[~d["decision"].isin(VALID_DECISIONS), "id_bem"].tolist()
    if bad:
        raise ValueError(f"Invalid decision (use keep|remove) for id_bem {bad}")
    return d


def apply_decisions(m: pd.DataFrame, flags: pd.DataFrame, decisions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Status, primary reason, flags as text and current decision.

    - removed        : hard flag without a `keep` decision, or a `remove` decision
    - pending_review : soft flags only, no decision
    - kept           : no flags, or a `keep` decision

    `primary_reason` is always the first flag raised (hard before soft, in
    config order), regardless of the decision; `status_source` tells whether
    the status came from the rules or from a manual decision.
    """
    v = values(cfg)
    order = v["hard_order"] + v["soft_order"]

    dup = decisions["id_bem"][decisions["id_bem"].duplicated()].unique().tolist()
    if dup:
        warnings.warn(f"Duplicate id_bem in decisions file (last row wins): {dup}")
    dec = decisions.drop_duplicates("id_bem", keep="last").set_index("id_bem")

    orphans = sorted(set(dec.index) - set(m["id_bem"]))
    if orphans:
        warnings.warn(f"Orphan decisions (id_bem not in snapshot): {orphans}")

    fl = flags.rename(columns=lambda c: c.removeprefix("flag_"))
    hard = fl[v["hard_order"]].any(axis=1)
    soft = fl[v["soft_order"]].any(axis=1)

    s = pd.DataFrame(index=m.index)
    s["id_bem"] = m["id_bem"]
    s["flags"] = fl[order].apply(lambda r: "|".join(r.index[r.values]), axis=1)
    first = s["flags"].str.split("|").str[0]
    s["primary_reason"] = first.where(first.ne(""), None)

    by_rule = np.where(hard, "removed", np.where(soft, "pending_review", "kept"))
    decision = m["id_bem"].map(dec["decision"]).astype(object)
    s["current_decision"] = decision.where(decision.notna(), None)
    s["status"] = np.where(decision.eq("keep"), "kept",
                  np.where(decision.eq("remove"), "removed", by_rule))
    s["status_source"] = np.where(decision.notna(), "manual_decision", "rule")
    return s


def load_exclusions(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"id_bem": "Int64"}, keep_default_na=False, na_values=[""])
    missing = set(EXCLUSION_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f"Exclusions file missing columns: {sorted(missing)}")
    empty = d.loc[d[EXCLUSION_COLUMNS].isna().any(axis=1), "id_bem"].tolist()
    if empty:
        raise ValueError(f"Empty field in exclusions file for id_bem {empty}")
    dup = d["id_bem"][d["id_bem"].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"Duplicate id_bem in exclusions file: {dup}")
    return d


def apply_exclusions(s: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    """Excluded sites become `removed` (`status_source` = `exclusion`), whatever the
    rule or manual decision; `primary_reason` keeps the first flag raised and the
    exclusion reason goes to `exclusion_reason`."""
    orphans = sorted(set(exclusions["id_bem"]) - set(s["id_bem"]))
    if orphans:
        warnings.warn(f"Orphan exclusions (id_bem not in snapshot): {orphans}")
    reason = s["id_bem"].map(exclusions.set_index("id_bem")["reason"]).astype(object)
    hit = reason.notna()
    s = s.copy()
    s["exclusion_reason"] = reason.where(hit, None)
    s.loc[hit, "status"] = "removed"
    s.loc[hit, "status_source"] = "exclusion"
    return s


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

ID_COLUMNS = ["identifica", "co_iphan", "sintese_be"]


def status_table(sites: pd.DataFrame, m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Every site: identification, status, reason, flags, decision and metrics."""
    ident = sites[["id_bem"] + ID_COLUMNS]
    cols_s = ["status", "primary_reason", "flags", "current_decision", "status_source", "exclusion_reason"]
    t = ident.join(s.reindex(columns=cols_s)).join(m.drop(columns=["id_bem"]))
    return t.reset_index(drop=True)


def review_register(table: pd.DataFrame) -> pd.DataFrame:
    """Removed and pending sites (read-only; decisions are edited in the decisions file)."""
    r = table[table["status"].isin(["removed", "pending_review"])]
    return r.sort_values(["status", "primary_reason", "island_name", "id_bem"], na_position="last")
