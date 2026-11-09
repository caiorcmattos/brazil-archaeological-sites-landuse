"""Summary tables for figures and manuscript text.

    python scripts/14_summary_tables.py
    python scripts/14_summary_tables.py --out-dir /tmp/smoke

Figure modules only draw; every number shown in a figure or quoted in the text
comes from a table written here.

Outputs (in --out-dir)
----------------------
- sites_per_biome.csv   biome, biome_en, n_sites, pct — sites per biome from the
  100 m buffers (`biome` assigned by 04_buffers.py)
- anthropic_change_periods.csv   change in anthropic cover per period (config/hex.toml,
  main + test) for NATIONAL and each biome, from the dissolved aggregate (no double
  counting): shares at start/end, delta (m², pp), mean annual rate (pp/yr, m²/yr) and
  share of the full-period change (sitios_arq.change)
- landcover_composition_national.csv   Natural/Anthropic/Water area and share of the NATIONAL
  dissolve (100 m) in the first and last year of the series (Fig 2 A)
- landcover_composition_biomes.csv   the same for each biome (dissolve clipped by biome, 100 m;
  Fig 3)
- landcover_change_biomes.csv   per biome × category: shares in the first and last year and
  delta_pp = difference of shares (equals change.delta_pp at 100 m); Fig 5
- reservoir_water_biomes.csv   reservoir-associated sites per water status
  (sitios_arq.reservoirs.STATUSES) for NATIONAL and every biome, total and hydropower-linked;
  Fig 4 A and text
- reservoir_water_hydro.csv   the same per hydropower reservoir (ANA gid), ranked by buffers
  entirely classified as water in the end year; `in_figure` marks the top N of
  config/reservoirs.toml (Fig 4 B)
- reservoir_water_share_hist.csv   reservoir-associated sites per class of water share in the
  end year (10% classes and exactly 100%), NATIONAL and biomes (Fig. S2 A)
- reservoir_water_tolerance.csv, reservoir_water_tolerance_hydro.csv   sensitivity of the
  entirely-water rule: the counts above with entirely water = share ≥ min_share
  (sensitivity_min_share of config/reservoirs.toml; 1.0 = the rule), per unit and per reservoir
- reservoir_water_radius.csv   sensitivity to the buffer radius (Fig. S2 B): the counts per radius,
  sites associated at 100 m, with the share of the 100 m count and the tolerated displacement
- radius_sensitivity.csv   buffer-radius sensitivity (Fig. S1), long format: per radius ×
  unit (NATIONAL + biomes) × category, composition (metric = pct) in the sensitivity years
  and delta_pp (pct_end − pct_start) for the main periods of config/hex.toml
- radius_sensitivity_concordance.csv   sign of each delta_pp vs the main radius (100 m) per
  radius × unit × category × period; n_periods_agree per radius, all_agree per unit ×
  category (no tolerance; raw deltas kept)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sitios_arq import change as C
from sitios_arq import hexgrid as H
from sitios_arq import extraction as E
from sitios_arq import reservoirs as R
from sitios_arq.extraction import NATIONAL
from sitios_arq.paths import BUFFER_DIR, RESERVOIRS_DIR, SUMMARY_DIR, THEMATIC_C11_DIR
from sitios_arq.io import save
from sitios_arq.style import BIOME_LABELS_EN

ROOT = Path(__file__).resolve().parents[1]
HEX_CONFIG = ROOT / "config/hex.toml"
EXTRACTION_CONFIG = ROOT / "config/extraction.toml"
RESERVOIRS_CONFIG = ROOT / "config/reservoirs.toml"

log = logging.getLogger("summary")


def sites_per_biome(buffers: pd.DataFrame) -> pd.DataFrame:
    """Unique sites per biome, in descending order; fails on a null or unknown biome."""
    if buffers["biome"].isna().any():
        raise ValueError(f"{buffers['biome'].isna().sum()} sites without a biome")
    unknown = set(buffers["biome"]) - set(BIOME_LABELS_EN)
    if unknown:
        raise ValueError(f"Biomes without a label in style.BIOME_LABELS_EN: {sorted(unknown)}")
    if buffers["id_bem"].duplicated().any():
        raise ValueError("Duplicate id_bem in the buffers")

    out = (buffers.groupby("biome")["id_bem"].nunique().rename("n_sites").reset_index()
           .sort_values(["n_sites", "biome"], ascending=[False, True], ignore_index=True))
    out.insert(1, "biome_en", out["biome"].map(BIOME_LABELS_EN))
    out["pct"] = 100 * out["n_sites"] / out["n_sites"].sum()
    return out


def anthropic_change_periods(aggregate: pd.DataFrame, cfg: dict, radius_m: int) -> pd.DataFrame:
    """Change per period for NATIONAL and the biomes; checks that sub-periods add up."""
    agg = aggregate[aggregate["radius_m"] == radius_m]
    periods = H.periods(cfg)
    out = C.period_change(agg, periods)

    C.check_periods_additivity(out, periods)
    full = C.full_span(periods)

    whole = out[out["period"] == C.period_label(*full)].set_index("unit")["delta_anthropic_m2"]
    # NaN when the full-span change is zero; the share is unstable when it is small
    out["share_of_full_change_pct"] = 100 * out["delta_anthropic_m2"] / out["unit"].map(whole.where(whole != 0))
    out.insert(1, "unit_en", out["unit"].map(BIOME_LABELS_EN | {NATIONAL: "Brazil"}))
    out.insert(2, "radius_m", radius_m)
    order = {u: i for i, u in enumerate([NATIONAL, *sorted(BIOME_LABELS_EN)])}
    out["_u"], out["_e"] = out["unit"].map(order), -out["end"]  # full span first
    return (out.sort_values(["_u", "start", "_e"], ignore_index=True)
            .drop(columns=["_u", "_e"]))


def landcover_composition(aggregate: pd.DataFrame, years: list[int], radius_m: int,
                          unit: str = NATIONAL) -> pd.DataFrame:
    """Category areas and shares (over the area with data) of `unit` in `years`."""
    nat = aggregate[(aggregate["unit"] == unit) & (aggregate["radius_m"] == radius_m)]
    nat = nat.set_index("year").loc[years]
    cats = {"Natural": "natural", "Anthropic": "anthropic", "Water": "water"}
    out = pd.DataFrame([{"year": y, "category": c,
                         "area_m2": nat.loc[y, f"area_{k}_m2"],
                         "pct": 100 * nat.loc[y, f"area_{k}_m2"] / nat.loc[y, "area_with_data_m2"]}
                        for y in years for c, k in cats.items()])
    s = out.groupby("year")["pct"].sum()
    if not ((s - 100).abs() < 1e-9).all():
        raise ValueError(f"{unit}: shares do not add up to 100: {s.to_dict()}")
    return out


def landcover_composition_biomes(aggregate: pd.DataFrame, years: list[int], radius_m: int) -> pd.DataFrame:
    units = set(aggregate["unit"]) - {NATIONAL}
    if units != set(BIOME_LABELS_EN):
        raise ValueError(f"Aggregate biomes {sorted(units)} != style.BIOME_LABELS_EN")
    out = pd.concat([landcover_composition(aggregate, years, radius_m, b).assign(biome=b)
                     for b in sorted(units)], ignore_index=True)
    out.insert(1, "biome_en", out["biome"].map(BIOME_LABELS_EN))
    return out[["biome", "biome_en", *out.columns.drop(["biome", "biome_en"])]]


def landcover_change_biomes(comp_b: pd.DataFrame) -> pd.DataFrame:
    """Change in share (pp) of each category per biome, first → last year of `comp_b`."""
    first, last = sorted(comp_b["year"].unique())
    keys = ["biome", "biome_en", "category"]
    w = comp_b.pivot_table(index=keys, columns="year", values="pct").reset_index()
    w = w.rename(columns={first: "pct_start", last: "pct_end"})
    w.insert(3, "start", first)
    w.insert(4, "end", last)
    w["delta_pp"] = w["pct_end"] - w["pct_start"]
    s = w.groupby("biome")["delta_pp"].sum()
    if not (s.abs() < 1e-9).all():
        raise ValueError(f"Category changes do not add up to 0: {s.to_dict()}")
    return w


CATEGORY_KEYS = {"Natural": "natural", "Anthropic": "anthropic", "Water": "water"}


def radius_sensitivity(aggregate: pd.DataFrame, radii: list[int], years: list[int],
                       periods: list[tuple[int, int]]) -> pd.DataFrame:
    """Composition (%) in `years` and Δpp per period, per radius × unit × category (long format).

    Shares over the area with data; delta_pp = pct_end − pct_start (equals change.delta_pp at
    100 m, where the area with data does not change between years).
    """
    agg = aggregate[aggregate["radius_m"].isin(radii) & aggregate["year"].isin(years)]
    have = agg.groupby(["radius_m", "unit"])["year"].nunique()
    units = [NATIONAL, *sorted(BIOME_LABELS_EN)]
    expected = pd.MultiIndex.from_product([radii, units], names=["radius_m", "unit"])
    if not have.index.equals(expected.sort_values()) or not (have == len(years)).all():
        raise ValueError(f"Aggregate lacks radius × unit × year combinations for {radii}, {years}")
    if {y for p in periods for y in p} - set(years):
        raise ValueError(f"Period endpoints {periods} not in years {years}")

    pct = pd.concat([agg[["radius_m", "unit", "year"]].assign(category=c, pct=100 * agg[f"prop_{k}"])
                     for c, k in CATEGORY_KEYS.items()], ignore_index=True)
    s = pct.groupby(["radius_m", "unit", "year"])["pct"].sum()
    if not ((s - 100).abs() < 1e-9).all():
        raise ValueError("Shares do not add up to 100")

    w = pct.pivot_table(index=["radius_m", "unit", "category"], columns="year", values="pct")
    delta = pd.concat([pd.DataFrame({"start": a, "end": b, "period": C.period_label(a, b),
                                     "value": w[b] - w[a]}).reset_index()
                       for a, b in periods], ignore_index=True)
    s = delta.groupby(["radius_m", "unit", "period"])["value"].sum()
    if not (s.abs() < 1e-9).all():
        raise ValueError("Category changes do not add up to 0")

    out = pd.concat([pct.rename(columns={"pct": "value"}).assign(metric="pct"),
                     delta.assign(metric="delta_pp")], ignore_index=True)
    out["year"] = out["year"].astype("Int64")
    out["start"] = out["start"].astype("Int64")
    out["end"] = out["end"].astype("Int64")
    out["unit_en"] = out["unit"].map(BIOME_LABELS_EN | {NATIONAL: "Brazil"})
    out["_u"] = out["unit"].map({u: i for i, u in enumerate(units)})
    out["_c"] = out["category"].map({c: i for i, c in enumerate(CATEGORY_KEYS)})
    out["_m"] = out["metric"].map({"pct": 0, "delta_pp": 1})
    out["_p"] = out["period"].map({C.period_label(*p): i for i, p in enumerate(periods)})
    out = out.sort_values(["radius_m", "_u", "_c", "_m", "year", "_p"], ignore_index=True)
    return out[["radius_m", "unit", "unit_en", "category", "metric", "year",
                "period", "start", "end", "value"]]


def radius_concordance(sens: pd.DataFrame, reference: int) -> pd.DataFrame:
    """Sign of each Δ vs the `reference` radius, per radius × unit × category × period.

    n_periods_agree: periods with the same sign at that radius (per unit × category × radius);
    all_agree: same sign at every radius and period (per unit × category). No tolerance.
    """
    d = sens[sens["metric"] == "delta_pp"]
    keys = ["unit", "unit_en", "category", "period", "start", "end"]
    ref = d[d["radius_m"] == reference].set_index(keys)["value"].rename("delta_pp_ref")
    out = (d[d["radius_m"] != reference].rename(columns={"value": "delta_pp"})
           .join(ref, on=keys).drop(columns=["metric", "year"]))
    if out["delta_pp_ref"].isna().any():
        raise ValueError("Δ without reference value")
    out.insert(0, "reference_radius_m", reference)
    out["sign_agrees"] = np.sign(out["delta_pp"]) == np.sign(out["delta_pp_ref"])
    g = out.groupby(["unit", "category", "radius_m"])["sign_agrees"]
    out["n_periods_agree"] = g.transform("sum").astype("int64")
    out["n_periods"] = g.transform("size").astype("int64")
    out["all_agree"] = out.groupby(["unit", "category"])["sign_agrees"].transform("all")
    return out.reset_index(drop=True)


def status_counts(sites: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sites per water status (total and hydropower-linked) for each group of `keys`."""
    def count(df, suffix):
        t = (df.groupby(keys + ["status"]).size().unstack("status", fill_value=0)
             .reindex(columns=list(R.STATUSES), fill_value=0))
        t.columns = [f"n_{c}{suffix}" for c in t.columns]
        t.insert(0, f"n_associated{suffix}", t.sum(axis=1))
        t[f"n_entirely_water{suffix}"] = t[f"n_entirely_water_both{suffix}"] + t[f"n_entirely_water_end{suffix}"]
        return t
    return count(sites, "").join(count(sites[sites["is_hydro"]], "_hydro")).fillna(0).astype("int64")


def reservoir_water_biomes(sites: pd.DataFrame) -> pd.DataFrame:
    """Status counts for NATIONAL and every biome (biomes without associated sites get zeros)."""
    unknown = set(sites["biome"]) - set(BIOME_LABELS_EN)
    if unknown:
        raise ValueError(f"Biomes without label in style.BIOME_LABELS_EN: {sorted(unknown)}")
    by_biome = status_counts(sites.rename(columns={"biome": "unit"}), ["unit"])
    by_biome = by_biome.reindex(sorted(BIOME_LABELS_EN), fill_value=0)
    national = status_counts(sites.assign(unit=NATIONAL), ["unit"])
    out = pd.concat([national, by_biome]).reset_index(names="unit")
    if not (out.loc[1:, out.columns[1:]].sum() == out.loc[0, out.columns[1:]]).all():
        raise ValueError("Biome counts do not add up to NATIONAL")
    out.insert(1, "unit_en", out["unit"].map(BIOME_LABELS_EN | {NATIONAL: "Brazil"}))
    out.insert(2, "start", sites["start"].iloc[0])
    out.insert(3, "end", sites["end"].iloc[0])
    return out


def reservoir_water_hydro(sites: pd.DataFrame, links: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Status counts per hydropower reservoir (ANA gid); a site linked to two reservoirs counts in both."""
    hl = links[links["is_hydro"]].merge(sites[["id_bem", "status", "is_hydro"]], on="id_bem",
                                        suffixes=("_link", ""))
    out = status_counts(hl, ["res_id"]).filter(regex=r"^n_(?!.*_hydro$)")
    meta = hl.groupby("res_id").agg(res_name=("res_name", "first"),
                                    biomes=("biome", lambda b: "; ".join(sorted(set(b)))))
    out = meta.join(out).reset_index()
    out = out.sort_values(["n_entirely_water", "n_entirely_water_end", "res_name"],
                          ascending=[False, False, True], ignore_index=True)
    out["rank"] = range(1, len(out) + 1)
    out["in_figure"] = out["rank"] <= top_n
    if out.loc[out["in_figure"], "res_name"].isna().any():
        raise ValueError("Unnamed hydropower reservoir among those shown in the figure")
    out.insert(3, "start", sites["start"].iloc[0])
    out.insert(4, "end", sites["end"].iloc[0])
    return out


SENS_COLS = ["unit", "unit_en", "start", "end", "n_associated", "n_entirely_water_both",
             "n_entirely_water_end", "n_entirely_water"]


def reservoir_water_tolerance(sites: pd.DataFrame, links: pd.DataFrame, min_shares: list[float],
                              top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Status counts with the entirely-water rule relaxed to each `min_shares` (1.0 = the rule).

    Returns (NATIONAL and biomes, hydropower reservoirs), one block per min_share; pct_vs_rule is
    the change of n_entirely_water relative to the rule, per unit.
    """
    levels = [1.0, *sorted(min_shares, reverse=True)]
    b, h = [], []
    for m in levels:
        s = sites.assign(status=R.water_status(sites["water_pct_start"] / 100, sites["water_pct_end"] / 100, m))
        b.append(reservoir_water_biomes(s)[SENS_COLS].assign(min_share=m))
        h.append(reservoir_water_hydro(s, links, top_n).assign(min_share=m))
    b, h = pd.concat(b, ignore_index=True), pd.concat(h, ignore_index=True)
    ref = b[b["min_share"] == 1.0].set_index("unit")["n_entirely_water"]
    b["pct_vs_rule"] = 100 * (b["n_entirely_water"] / b["unit"].map(ref) - 1)
    b.loc[b["unit"].map(ref) == 0, "pct_vs_rule"] = float("nan")
    return b, h


SHARE_BIN_WIDTH = 10  # %


def reservoir_water_share_hist(sites: pd.DataFrame) -> pd.DataFrame:
    """Reservoir-associated sites per class of water share in the end year, NATIONAL and biomes.

    Classes [0, 10), [10, 20), ..., [90, 100) and exactly 100 (entirely water, the rule).
    """
    lows = list(range(0, 100, SHARE_BIN_WIDTH))
    w = sites["water_pct_end"]
    if ((w < 0) | (w > 100)).any():
        raise ValueError("water share outside [0, 100]")
    cls = np.where(w == 100, 100, np.minimum((w // SHARE_BIN_WIDTH) * SHARE_BIN_WIDTH, lows[-1])).astype(int)
    s = sites.assign(bin_low=cls)
    units = [NATIONAL, *sorted(BIOME_LABELS_EN)]
    rows = []
    for unit in units:
        g = s if unit == NATIONAL else s[s["biome"] == unit]
        n = g["bin_low"].value_counts()
        for lo in [*lows, 100]:
            rows.append({"unit": unit, "bin_low": lo, "bin_high": 100 if lo == 100 else lo + SHARE_BIN_WIDTH,
                         "entirely_water": lo == 100, "n_sites": int(n.get(lo, 0))})
    out = pd.DataFrame(rows)
    out.insert(1, "unit_en", out["unit"].map(BIOME_LABELS_EN | {NATIONAL: "Brazil"}))
    out.insert(2, "year", int(sites["end"].iloc[0]))
    return out


def reservoir_water_radius(radius_sites: pd.DataFrame, reference: int) -> pd.DataFrame:
    """Status counts per radius (sites associated at `reference`); share of the reference count and
    tolerated displacement (a buffer of radius r entirely water implies the `reference` buffer of
    every point within r − reference of the site is entirely water too)."""
    entire = radius_sites["status"].str.startswith("entirely")
    ids = radius_sites[entire].groupby("radius_m")["id_bem"].apply(set)
    for r, s in ids.items():
        if not s <= ids[reference]:
            raise ValueError(f"Buffers entirely water at {r} m that are not at {reference} m")
    out = pd.concat([reservoir_water_biomes(g.drop(columns="radius_m"))[SENS_COLS].assign(radius_m=r)
                     for r, g in radius_sites.groupby("radius_m")], ignore_index=True)
    ref = out[out["radius_m"] == reference].set_index("unit")["n_entirely_water"]
    out["pct_of_reference"] = 100 * out["n_entirely_water"] / out["unit"].map(ref).replace(0, float("nan"))
    out["tolerated_displacement_m"] = out["radius_m"] - reference
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffer-dir", default=str(BUFFER_DIR))
    p.add_argument("--out-dir", default=str(SUMMARY_DIR))
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    src = Path(args.buffer_dir) / "buffers_100m.parquet"
    buffers = pd.read_parquet(src, columns=["id_bem", "biome"])
    table = sites_per_biome(buffers)
    for r in table.itertuples():
        log.info("%-15s %6d  %5.1f%%", r.biome_en, r.n_sites, r.pct)
    log.info("Total: %d sites", table["n_sites"].sum())

    save(table, out_dir / "sites_per_biome.csv")

    cfg = H.load_config(HEX_CONFIG)
    r = cfg["radius_m"]["value"]
    agg_src = THEMATIC_C11_DIR / "thematic_aggregate_c11.parquet"
    change = anthropic_change_periods(pd.read_parquet(agg_src), cfg, r)
    log.info("Anthropic change, %s (%d m):", "Brazil", r)
    for row in change[change["unit"] == NATIONAL].itertuples():
        log.info("  %-10s %5.1f%% → %5.1f%%  Δ %+5.2f pp  %+.3f pp/yr  (%5.1f%% of full)",
                 row.period, row.anthropic_pct_start, row.anthropic_pct_end,
                 row.delta_pp, row.rate_pp_per_yr, row.share_of_full_change_pct)
    save(change, out_dir / "anthropic_change_periods.csv")

    first, last = E.load_config(EXTRACTION_CONFIG)["series_years"]["value"]
    agg = pd.read_parquet(agg_src)
    comp = landcover_composition(agg, [first, last], r)
    for row in comp.itertuples():
        log.info("Composition %d %-9s %5.1f%%", row.year, row.category, row.pct)
    save(comp, out_dir / "landcover_composition_national.csv")

    comp_b = landcover_composition_biomes(agg, [first, last], r)
    for row in comp_b[comp_b["category"] == "Anthropic"].itertuples():
        log.info("Composition %-15s %d Anthropic %5.1f%%", row.biome_en, row.year, row.pct)
    save(comp_b, out_dir / "landcover_composition_biomes.csv")

    chg_b = landcover_change_biomes(comp_b)
    for row in chg_b[chg_b["category"] != "Natural"].itertuples():
        log.info("Change %-15s %-9s %+6.2f pp", row.biome_en, row.category, row.delta_pp)
    save(chg_b, out_dir / "landcover_change_biomes.csv")

    ecfg = E.load_config(EXTRACTION_CONFIG)
    radii, sens_years = ecfg["radii_m"]["value"], ecfg["sensitivity_years"]["value"]
    main_radius = ecfg["main_radius_m"]["value"]
    main_periods = H.periods(cfg)
    sens = radius_sensitivity(agg, radii, sens_years, main_periods)
    # same metric as anthropic_change_periods.csv at the main radius; at the others the area with
    # data varies with class 0 and change.delta_pp (over the mean) differs from the difference of shares
    ref = pd.concat([C.period_change(agg[agg["radius_m"] == rr], main_periods)
                     .assign(radius_m=rr) for rr in radii])
    d_ant = sens[(sens["metric"] == "delta_pp") & (sens["category"] == "Anthropic")]
    diff = d_ant.merge(ref[["radius_m", "unit", "period", "delta_pp"]], on=["radius_m", "unit", "period"])
    if len(diff) != len(d_ant):
        raise ValueError("Radius sensitivity rows without change.period_change counterpart")
    diff["d"] = (diff["value"] - diff["delta_pp"]).abs()
    if diff.loc[diff["radius_m"] == main_radius, "d"].max() > 1e-9:
        raise ValueError(f"{main_radius} m: delta_pp differs from change.period_change")
    log.info("Radius sensitivity: max |Δ shares − change.delta_pp| (anthropic) per radius: %s",
             diff.groupby("radius_m")["d"].max().map("{:.2e}".format).to_dict())
    save(sens, out_dir / "radius_sensitivity.csv")

    conc = radius_concordance(sens, main_radius)
    summ = (conc.drop_duplicates(["unit", "category", "radius_m"])
            .pivot(index=["unit_en", "category"], columns="radius_m", values="n_periods_agree"))
    log.info("Radius sensitivity: periods with the same sign as %d m (of %d):\n%s",
             main_radius, len(main_periods), summ.to_string())
    save(conc, out_dir / "radius_sensitivity_concordance.csv")

    rcfg = R.load_config(RESERVOIRS_CONFIG)
    sites_src = RESERVOIRS_DIR / "reservoir_sites_water.parquet"
    links_src = RESERVOIRS_DIR / "site_reservoir_links.parquet"
    sites = pd.read_parquet(sites_src)
    res_b = reservoir_water_biomes(sites)
    for row in res_b.itertuples():
        log.info("Reservoirs %-15s associated %5d | entirely water: both %4d, end only %4d | "
                 "partial %4d | no gain %4d", row.unit_en, row.n_associated, row.n_entirely_water_both,
                 row.n_entirely_water_end, row.n_partial_gain, row.n_no_gain)
    save(res_b, out_dir / "reservoir_water_biomes.csv")

    res_h = reservoir_water_hydro(sites, pd.read_parquet(links_src), rcfg["top_hydro"]["value"])
    for row in res_h[res_h["in_figure"]].itertuples():
        log.info("Hydropower #%d %-28s entirely water %4d (both %3d, end only %3d) | partial %3d",
                 row.rank, row.res_name, row.n_entirely_water, row.n_entirely_water_both,
                 row.n_entirely_water_end, row.n_partial_gain)
    save(res_h, out_dir / "reservoir_water_hydro.csv")

    hist = reservoir_water_share_hist(sites)
    tot = hist.groupby("unit")["n_sites"].sum()
    full = hist[hist["entirely_water"]].set_index("unit")["n_sites"]
    rb = res_b.set_index("unit")
    if not (tot.equals(rb.loc[tot.index, "n_associated"]) and full.equals(rb.loc[full.index, "n_entirely_water"])):
        raise ValueError("Water-share classes do not add up to the reservoir_water_biomes counts")
    log.info("Water share in %d, Brazil: %s", hist["year"].iloc[0],
             dict(zip(hist.loc[hist["unit"] == NATIONAL, "bin_low"], hist.loc[hist["unit"] == NATIONAL, "n_sites"])))
    save(hist, out_dir / "reservoir_water_share_hist.csv")

    tol_b, tol_h = reservoir_water_tolerance(sites, pd.read_parquet(links_src),
                                             rcfg["sensitivity_min_share"]["value"], rcfg["top_hydro"]["value"])
    rule = tol_b[tol_b["min_share"] == 1.0].drop(columns=["min_share", "pct_vs_rule"]).reset_index(drop=True)
    if not rule.equals(res_b[SENS_COLS]):
        raise ValueError("Tolerance table at the rule differs from reservoir_water_biomes")
    for row in tol_b[tol_b["unit"] == NATIONAL].itertuples():
        top = tol_h[(tol_h["min_share"] == row.min_share) & tol_h["in_figure"]]["res_name"].tolist()
        log.info("Tolerance ≥ %.2f: entirely water %4d (%+5.1f%%) | top %s", row.min_share,
                 row.n_entirely_water, row.pct_vs_rule, top)
    save(tol_b, out_dir / "reservoir_water_tolerance.csv")
    save(tol_h, out_dir / "reservoir_water_tolerance_hydro.csv")

    rad_src = RESERVOIRS_DIR / "reservoir_sites_water_radius.parquet"
    rad = reservoir_water_radius(pd.read_parquet(rad_src), rcfg["radius_m"]["value"])
    for row in rad[rad["unit"] == NATIONAL].itertuples():
        log.info("Radius %4d m (displacement ≤ %3d m): entirely water %4d (%5.1f%%)", row.radius_m,
                 row.tolerated_displacement_m, row.n_entirely_water, row.pct_of_reference)
    save(rad, out_dir / "reservoir_water_radius.csv")


if __name__ == "__main__":
    main()
