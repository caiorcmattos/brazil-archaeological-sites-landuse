"""Filtering tests on synthetic geometries.

Scenario (lon/lat, SIRGAS 2000 → EPSG:5880):
- "Brazil"  : box lon -52..-48, lat -12..-8
- neighbor  : box lon -56..-52 (border at lon -52)
- sea       : east of lon -48 (coast)
- islands at sea near lon -47.9: maritime (lat -10), untyped (lat -11), fluvial (lat -9),
  large maritime (lat -11.5; declared area ≥ threshold)
- MapBiomas bounds: top at lat -8.5 (northern strip of "Brazil" is outside)

1° of longitude at lat ~-10 ≈ 109.6 km.
"""

import copy
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from sitios_arq import filtering as F

ROOT = Path(__file__).resolve().parents[1]

CFG = {
    "coast_tolerance_m": {"value": 250},
    "offshore_hard_m": {"value": 1000},
    "border_tolerance_m": {"value": 100},
    "island_types_flagged": {"value": ["maritime", "untyped"]},
    "max_island_area_km2": {"value": 100},
    "shipwreck_terms": {"value": ["naufrag*", "socobr*", "ao largo", "alto-mar"]},
    "ambiguous_terms": {"value": ["vapor", "navio"]},
    "hard_order": {"value": list(F.HARD_FLAGS)},
    "soft_order": {"value": list(F.SOFT_FLAGS)},
}
MB_BOUNDS = (-60.0, -60.0, -30.0, -8.5)

# id_bem: (lon, lat, sintese)
POINTS = {
    1: (-50.0, -10.0, ""),                 # land
    2: (-48 + 0.001, -10.5, ""),           # sea, ~110 m
    3: (-48 + 0.005, -10.5, ""),           # sea, ~550 m
    4: (-48 + 0.02, -10.5, ""),            # sea, ~2.2 km
    5: (-52 - 0.0005, -10.0, ""),          # neighbor, ~55 m
    6: (-52 - 0.01, -10.0, ""),            # neighbor, ~1.1 km
    7: (-47.925, -10.025, ""),             # maritime island
    8: (-47.925, -11.025, ""),             # untyped island
    9: (-47.925, -9.025, ""),              # fluvial island
    10: (530046.0, 9710233.0, ""),         # UTM typed as degrees
    11: (-10.0, -50.0, ""),                # lat/lon swapped from (-50, -10)
    12: (-50.0, -9.5, "Naufrágio do vapor Tal"),
    13: (-50.0, -8.2, ""),                 # outside MapBiomas bounds
    14: (-48 + 0.02, -10.6, "naufragou ao largo"),
    15: (-47.925, -11.525, ""),            # large maritime island
}


def _proj(geom):
    # Densify before projecting: in the polyconic, long lon/lat edges become curves.
    return gpd.GeoSeries([geom.segmentize(0.001)], crs=4674).to_crs(5880).iloc[0]


@pytest.fixture(scope="module")
def refs():
    brazil = _proj(box(-52, -12, -48, -8))
    neighbor = _proj(box(-56, -12, -52, -8))
    islands = gpd.GeoDataFrame(
        {
            "name": ["Island M", "Island U", "Island F", "Island L"],
            "island_type": ["maritime", "untyped", "fluvial", "maritime"],
            # Declared areas (not the geometry's) to test the threshold.
            "area_km2": [30.0, 30.0, 30.0, 100.0],
        },
        geometry=[
            _proj(box(-47.95, -10.05, -47.90, -10.0)),
            _proj(box(-47.95, -11.05, -47.90, -11.0)),
            _proj(box(-47.95, -9.05, -47.90, -9.0)),
            _proj(box(-47.95, -11.55, -47.90, -11.5)),
        ],
        crs=5880,
    )
    land = shapely.union_all([brazil, *islands.geometry])
    coastline, border = F.derive_lines(brazil, land, [neighbor], eps_m=10)
    return F.References(
        brazil=brazil,
        land=land,
        neighbors=gpd.GeoDataFrame({"name": ["Neighbor"]}, geometry=[neighbor], crs=5880),
        coastline=gpd.GeoDataFrame(geometry=coastline, crs=5880),
        border=gpd.GeoDataFrame(geometry=border, crs=5880),
        islands=islands,
        municipalities=gpd.GeoDataFrame({"geocode": ["1100015"]}, geometry=[brazil], crs=5880),
        biomes=gpd.GeoDataFrame({"Bioma": ["Amazônia"]}, geometry=[brazil], crs=5880),
    )


@pytest.fixture(scope="module")
def sites():
    ids = list(POINTS)
    return pd.DataFrame(
        {
            "id_bem": ids,
            "lon": [POINTS[i][0] for i in ids],
            "lat": [POINTS[i][1] for i in ids],
            "sintese_be": [POINTS[i][2] for i in ids],
            "identifica": ["" for _ in ids],
            "co_iphan": ["RO1100015BAST00001" for _ in ids],
        }
    )


@pytest.fixture(scope="module")
def metrics(sites, refs):
    return F.compute_metrics(sites, refs, CFG, MB_BOUNDS)


@pytest.fixture(scope="module")
def flags(metrics):
    return F.build_flags(metrics, CFG)


def _decisions(rows=()):
    return pd.DataFrame(rows, columns=F.DECISION_COLUMNS).astype({"id_bem": "Int64"})


@pytest.fixture(scope="module")
def status(metrics, flags):
    return F.apply_decisions(metrics, flags, _decisions(), CFG).set_index("id_bem")


def _flags_of(status, i):
    f = status.loc[i, "flags"]
    return set(f.split("|")) if f else set()


# --- text -------------------------------------------------------------------

def test_normalization_and_word_boundaries():
    rx = F.compile_terms(["naufrag*", "vapor", "socobr*", "ao largo", "alto-mar"])
    assert F.find_terms(F.normalize_text("NAUFRÁGIO"), rx) == "naufrag*"
    assert F.find_terms(F.normalize_text("naufragio"), rx) == "naufrag*"
    assert F.find_terms(F.normalize_text("vaporizar"), rx) == ""
    assert F.find_terms(F.normalize_text("o vapor"), rx) == "vapor"
    assert F.find_terms(F.normalize_text("Soçobrou"), rx) == "socobr*"
    assert F.find_terms(F.normalize_text("ao  largo da ilha"), rx) == "ao largo"
    assert F.find_terms(F.normalize_text("aolargo"), rx) == ""
    assert F.find_terms(F.normalize_text("em alto mar"), rx) == "alto-mar"
    assert F.find_terms(F.normalize_text("em alto-mar"), rx) == "alto-mar"
    assert F.normalize_text(None) == ""


# --- rules ------------------------------------------------------------------

@pytest.mark.parametrize(
    "i, expected_flags, expected_status",
    [
        (1, set(), "kept"),
        (2, set(), "kept"),
        (3, {"offshore_near"}, "pending_review"),
        (4, {"offshore_far"}, "removed"),
        (5, {"near_border"}, "pending_review"),
        (6, {"outside_brazil"}, "removed"),
        (7, {"maritime_island"}, "pending_review"),
        (8, {"maritime_island"}, "pending_review"),
        (9, set(), "kept"),
        (15, set(), "kept"),  # maritime island with area = threshold (not < 100 km²)
        (10, {"invalid_coord"}, "removed"),
        (12, {"shipwreck_text"}, "removed"),
        (13, {"outside_mapbiomas"}, "removed"),
    ],
)
def test_rules(status, i, expected_flags, expected_status):
    assert _flags_of(status, i) == expected_flags
    assert status.loc[i, "status"] == expected_status


def test_basic_metrics(metrics):
    m = metrics.set_index("id_bem")
    assert m.loc[1, "coast_side"] == "land" and m.loc[1, "inside_brazil"]
    assert m.loc[2, "coast_side"] == "sea" and 50 < m.loc[2, "dist_coast_m"] < 250
    assert m.loc[5, "coast_side"] == "other_country" and m.loc[5, "dist_border_m"] < 100
    assert m.loc[5, "spatial_country"] == "Neighbor"
    assert m.loc[7, "coast_side"] == "land" and not m.loc[7, "inside_brazil"]
    assert m.loc[7, "island_name"] == "Island M" and m.loc[8, "island_type"] == "untyped"
    assert m.loc[9, "island_type"] == "fluvial" and not m.loc[9, "maritime_island"]
    assert m.loc[15, "maritime_island"] and m.loc[15, "island_area_km2"] == 100.0  # metric keeps the type
    assert m.loc[1, "spatial_mun"] == "1100015" and m.loc[1, "declared_mun"] == "1100015"
    assert m.loc[1, "biome"] == "Amazônia"
    assert m.loc[12, "ambiguous_terms"] == "vapor"
    assert np.isnan(m.loc[10, "dist_coast_m"]) and pd.isna(m.loc[10, "coast_side"])


def test_swapped_coord_not_corrected(sites, metrics, status):
    m = metrics.set_index("id_bem")
    assert m.loc[11, "swapped_coord"]
    assert (m.loc[11, "lon"], m.loc[11, "lat"]) == (-10.0, -50.0)
    assert m.loc[11, "coast_side"] == "sea"  # metrics at the original point
    assert status.loc[11, "primary_reason"] == "swapped_coord"
    assert status.loc[11, "status"] == "removed"
    assert sites.set_index("id_bem").loc[11, "lon"] == -10.0


# --- precedence -------------------------------------------------------------

def test_primary_reason_precedence(metrics, flags, status):
    assert _flags_of(status, 14) == {"offshore_far", "shipwreck_text"}
    assert status.loc[14, "primary_reason"] == "offshore_far"
    assert status.loc[14, "flags"] == "offshore_far|shipwreck_text"

    cfg = copy.deepcopy(CFG)
    cfg["hard_order"]["value"] = [
        "invalid_coord", "swapped_coord", "outside_brazil",
        "shipwreck_text", "offshore_far", "outside_mapbiomas",
    ]
    s = F.apply_decisions(metrics, flags, _decisions(), cfg).set_index("id_bem")
    assert s.loc[14, "primary_reason"] == "shipwreck_text"


def test_hard_before_soft(metrics):
    m = metrics.copy()
    m.loc[m["id_bem"] == 7, "shipwreck_terms"] = "naufrag*"
    f = F.build_flags(m, CFG)
    s = F.apply_decisions(m, f, _decisions(), CFG).set_index("id_bem")
    assert s.loc[7, "primary_reason"] == "shipwreck_text"
    assert s.loc[7, "status"] == "removed"


# --- manual decisions -------------------------------------------------------

def test_manual_decision_overrides(metrics, flags):
    dec = _decisions([
        (4, "keep", "cm", "2026-09-25"),
        (1, "remove", "cm", "2026-09-25"),
    ])
    s = F.apply_decisions(metrics, flags, dec, CFG).set_index("id_bem")
    assert s.loc[4, "status"] == "kept" and s.loc[4, "status_source"] == "manual_decision"
    assert s.loc[4, "primary_reason"] == "offshore_far"
    assert s.loc[1, "status"] == "removed" and s.loc[1, "status_source"] == "manual_decision"
    assert s.loc[2, "status_source"] == "rule"


def test_orphan_decision_warns(metrics, flags):
    dec = _decisions([(999999, "keep", "cm", "2026-09-25")])
    with pytest.warns(UserWarning, match="Orphan.*999999"):
        F.apply_decisions(metrics, flags, dec, CFG)


def test_no_decisions_no_warning(metrics, flags):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        F.apply_decisions(metrics, flags, _decisions(), CFG)


def test_current_decision_in_register(sites, metrics, flags):
    dec = _decisions([(3, "remove", "cm", "2026-09-25")])
    s = F.apply_decisions(metrics, flags, dec, CFG)
    reg = F.review_register(F.status_table(sites, metrics, s)).set_index("id_bem")
    assert reg.loc[3, "current_decision"] == "remove"
    assert reg.loc[3, "status"] == "removed"
    assert pd.isna(reg.loc[6, "current_decision"])
    assert "decision" not in reg.columns
    assert 1 not in reg.index  # kept sites stay out of the register


def test_load_decisions_rejects_invalid_value(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("id_bem,decision,reviewer,date\n1,maybe,cm,2026-09-25\n")
    with pytest.raises(ValueError, match="keep\\|remove"):
        F.load_decisions(p)


def test_load_decisions_rejects_portuguese_value(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("id_bem,decision,reviewer,date\n1,manter,cm,2026-09-25\n")
    with pytest.raises(ValueError, match="keep\\|remove"):
        F.load_decisions(p)


def test_load_decisions_requires_reviewer(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("id_bem,decision,date\n1,keep,2026-09-25\n")
    with pytest.raises(ValueError, match="reviewer"):
        F.load_decisions(p)


def _exclusions(rows=()):
    return pd.DataFrame(rows, columns=F.EXCLUSION_COLUMNS).astype({"id_bem": "Int64"})


def test_exclusion_overrides_rule_and_keep_decision(sites, metrics, flags):
    dec = _decisions([(4, "keep", "cm", "2026-09-25")])
    s = F.apply_decisions(metrics, flags, dec, CFG)
    s = F.apply_exclusions(s, _exclusions([(4, "no_mapbiomas_data", "cm", "2026-10-12"),
                                           (2, "no_mapbiomas_data", "cm", "2026-10-12")]))
    t = s.set_index("id_bem")
    assert t.loc[4, "status"] == "removed" and t.loc[4, "status_source"] == "exclusion"
    assert t.loc[4, "primary_reason"] == "offshore_far" and t.loc[4, "current_decision"] == "keep"
    assert t.loc[2, "status"] == "removed" and t.loc[2, "exclusion_reason"] == "no_mapbiomas_data"
    assert t["exclusion_reason"].isna().sum() == len(t) - 2
    reg = F.review_register(F.status_table(sites, metrics, s)).set_index("id_bem")
    assert reg.loc[2, "exclusion_reason"] == "no_mapbiomas_data"


def test_orphan_exclusion_warns(metrics, flags):
    s = F.apply_decisions(metrics, flags, _decisions(), CFG)
    with pytest.warns(UserWarning, match="Orphan exclusions.*999999"):
        F.apply_exclusions(s, _exclusions([(999999, "no_mapbiomas_data", "cm", "2026-10-12")]))


def test_load_exclusions_rejects_duplicates_and_empty_reason(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("id_bem,reason,reviewer,date\n1,no_mapbiomas_data,cm,2026-10-12\n1,x,cm,2026-10-12\n")
    with pytest.raises(ValueError, match="Duplicate"):
        F.load_exclusions(p)
    p.write_text("id_bem,reason,reviewer,date\n1,,cm,2026-10-12\n")
    with pytest.raises(ValueError, match="Empty field"):
        F.load_exclusions(p)


def test_repository_exclusions_file_loads():
    e = F.load_exclusions(ROOT / "config/class0_exclusions.csv")
    assert list(e.columns) == F.EXCLUSION_COLUMNS
    assert len(e) == 43 and set(e["reason"]) == {"no_mapbiomas_data"}


def test_repository_decisions_file_loads():
    d = F.load_decisions(ROOT / "config/filtering_decisions.csv")
    assert list(d.columns) == F.DECISION_COLUMNS
    assert set(d["decision"]) <= F.VALID_DECISIONS
    assert d["id_bem"].is_unique and d["id_bem"].notna().all()


# --- config -----------------------------------------------------------------

def test_repository_config_loads():
    cfg = F.load_config(ROOT / "config/filtering.toml")
    assert cfg["hard_order"]["value"][-1] == "outside_mapbiomas"
    terms = cfg["shipwreck_terms"]["value"]
    assert not any(t.startswith(("submers", "inundad")) for t in terms)
