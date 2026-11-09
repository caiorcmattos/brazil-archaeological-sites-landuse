"""Change in anthropic cover between two years, per spatial unit.

Same metric for the national/biome aggregate and for the hexagons:

    delta_pp       = 100 · (A_end − A_start) / mean(V_start, V_end)
    rate_pp_per_yr = delta_pp / (end − start)

A = anthropic area, V = area with MapBiomas data (class 0 out of the
denominator). At 100 m V does not change between years,
so delta_pp equals the difference of the anthropic shares. Rates are mean annual (linear) changes over the period.
"""

from __future__ import annotations

import pandas as pd

COLS = ("year", "area_anthropic_m2", "area_with_data_m2")


def period_label(start: int, end: int) -> str:
    return f"{start}–{end}"


def period_change(df: pd.DataFrame, periods: list[tuple[int, int]], key: str = "unit") -> pd.DataFrame:
    """One row per (unit, period). `periods` = [(start, end), ...]."""
    missing = [c for c in (key, *COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.duplicated([key, "year"]).any():
        raise ValueError(f"Duplicated ({key}, year) rows")
    by_year = {y: g.set_index(key) for y, g in df.groupby("year")}

    rows = []
    for start, end in periods:
        for y in (start, end):
            if y not in by_year:
                raise ValueError(f"Year {y} not in the data")
        s, e = by_year[start], by_year[end]
        if set(s.index) != set(e.index):
            raise ValueError(f"{period_label(start, end)}: different units in {start} and {end}")
        e = e.loc[s.index]
        delta = e["area_anthropic_m2"] - s["area_anthropic_m2"]
        valid = (s["area_with_data_m2"] + e["area_with_data_m2"]) / 2
        n = end - start
        out = pd.DataFrame({
            key: s.index,
            "period": period_label(start, end), "start": start, "end": end,
            "n_years": n,
            "anthropic_pct_start": (100 * s["area_anthropic_m2"] / s["area_with_data_m2"]).to_numpy(),
            "anthropic_pct_end": (100 * e["area_anthropic_m2"] / e["area_with_data_m2"]).to_numpy(),
            "area_with_data_mean_m2": valid.to_numpy(),
            "delta_anthropic_m2": delta.to_numpy(),
            "delta_pp": (100 * delta / valid).to_numpy(),
        })
        out["rate_pp_per_yr"] = out["delta_pp"] / n
        out["rate_m2_per_yr"] = out["delta_anthropic_m2"] / n
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def check_additivity(change: pd.DataFrame, whole: tuple[int, int], parts: list[tuple[int, int]],
                     key: str = "unit", rtol: float = 1e-9) -> None:
    """Σ delta_anthropic_m2 of consecutive `parts` == delta of `whole`, per unit."""
    if parts[0][0] != whole[0] or parts[-1][1] != whole[1] or any(
            a[1] != b[0] for a, b in zip(parts, parts[1:])):
        raise ValueError(f"{parts} do not tile {whole}")
    d = change.set_index([key, "period"])["delta_anthropic_m2"]
    w = d.xs(period_label(*whole), level="period")
    p = sum(d.xs(period_label(*q), level="period") for q in parts)
    scale = change["area_with_data_mean_m2"].max()
    bad = (p - w).abs() > rtol * scale
    if bad.any():
        raise RuntimeError(f"Periods {parts} do not add up to {whole} for {list(w.index[bad])}")


def full_span(periods: list[tuple[int, int]]) -> tuple[int, int]:
    """Span of the periods; it must itself be one of them (e.g. 1985–2025)."""
    periods = [tuple(p) for p in periods]
    full = (min(s for s, _ in periods), max(e for _, e in periods))
    if full not in periods:
        raise ValueError(f"Periods must include the full span {full}")
    return full


def check_periods_additivity(change: pd.DataFrame, periods: list[tuple[int, int]],
                             key: str = "unit") -> None:
    """The sub-periods add up to the full span."""
    full = full_span(periods)
    check_additivity(change, full, sorted(tuple(p) for p in periods if tuple(p) != full), key)
