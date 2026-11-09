"""Figure infrastructure tests: style palette, summary tables and export."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from sitios_arq import style as S  # noqa: E402
from sitios_arq.figures import FIGURE_NUMBERS, FIGURES  # noqa: E402
from sitios_arq.figures.export import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIOMES = {"Amazônia", "Mata Atlântica", "Caatinga", "Cerrado", "Pampa", "Pantanal"}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_biome_palette_complete_and_valid():
    assert set(S.BIOME_COLORS) == set(S.BIOME_LABELS_EN) == set(S.BIOME_ORDER) == BIOMES
    assert all(mcolors.is_color_like(c) for c in S.BIOME_COLORS.values())
    assert len(set(S.BIOME_COLORS.values())) == len(BIOMES)


def test_tint_bounds():
    assert S.tint("#1F5C45", 1.0) == "#1f5c45"
    assert S.tint("#1F5C45", 0.0) == "#ffffff"


def test_font_registered():
    S.apply_style()
    assert matplotlib.rcParams["font.family"] == [S.FONT_FAMILY]
    assert matplotlib.rcParams["axes.linewidth"] >= 2.0


def test_figure_registry():
    assert set(FIGURE_NUMBERS.values()) <= set(FIGURES)


def test_sites_per_biome():
    summary = _load_script("14_summary_tables")
    buffers = pd.DataFrame({"id_bem": [1, 2, 3, 4, 5],
                            "biome": ["Caatinga", "Pampa", "Caatinga", "Caatinga", "Pampa"]})
    t = summary.sites_per_biome(buffers)
    assert list(t["biome"]) == ["Caatinga", "Pampa"]
    assert list(t["n_sites"]) == [3, 2]
    assert t["n_sites"].sum() == len(buffers)
    assert t["pct"].sum() == pytest.approx(100)
    assert list(t["biome_en"]) == ["Caatinga", "Pampa"]


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"id_bem": [1, 2], "biome": ["Caatinga", None]}),
    pd.DataFrame({"id_bem": [1, 2], "biome": ["Caatinga", "Atlantis"]}),
    pd.DataFrame({"id_bem": [1, 1], "biome": ["Caatinga", "Caatinga"]}),
])
def test_sites_per_biome_rejects_bad_input(bad):
    summary = _load_script("14_summary_tables")
    with pytest.raises(ValueError):
        summary.sites_per_biome(bad)


def _composition(pcts):
    rows = [{"biome": "Pampa", "biome_en": "Pampa", "year": y, "category": c, "pct": v}
            for y, shares in pcts.items() for c, v in shares.items()]
    return pd.DataFrame(rows)


def test_landcover_change_biomes():
    summary = _load_script("14_summary_tables")
    comp = _composition({1985: {"Natural": 70, "Anthropic": 25, "Water": 5},
                         2025: {"Natural": 60, "Anthropic": 32, "Water": 8}})
    t = summary.landcover_change_biomes(comp).set_index("category")
    assert t.loc["Anthropic", "delta_pp"] == pytest.approx(7)
    assert t.loc["Water", "delta_pp"] == pytest.approx(3)
    assert t.loc["Natural", "delta_pp"] == pytest.approx(-10)
    assert set(t["start"]) == {1985} and set(t["end"]) == {2025}


def test_landcover_change_biomes_rejects_shares_not_closing():
    summary = _load_script("14_summary_tables")
    comp = _composition({1985: {"Natural": 70, "Anthropic": 25, "Water": 5},
                         2025: {"Natural": 60, "Anthropic": 32, "Water": 9}})
    with pytest.raises(ValueError):
        summary.landcover_change_biomes(comp)


def test_export_writes_rgb_tiff_pdf_and_png(tmp_path):
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1], [0, 1])
    save_figure(fig, tmp_path / "out", "demo", description="test")
    plt.close(fig)

    tif = Image.open(tmp_path / "out/demo.tif")
    assert tif.mode == "RGB"
    assert round(tif.info["dpi"][0]) == S.DPI_RASTER
    assert (tmp_path / "out/demo.png").stat().st_size > 0
    assert (tmp_path / "out/demo.pdf").stat().st_size > 0


def test_drop_offshore_islands_keeps_coastal_and_inhabited():
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Point, box

    from sitios_arq.figures.sites_biomes import drop_offshore_islands

    mainland = box(0, 0, 100_000, 100_000)
    coastal = box(105_000, 0, 106_000, 1_000)        # 5 km off the coast
    far_empty = box(160_000, 0, 161_000, 1_000)      # 60 km, no site
    far_site = box(160_000, 50_000, 161_000, 51_000)  # 60 km, with a site
    biomes = gpd.GeoDataFrame({"Bioma": ["Mata Atlântica"]},
                              geometry=[MultiPolygon([mainland, coastal, far_empty, far_site])],
                              crs=5880)
    sites = gpd.GeoDataFrame(geometry=[Point(50_000, 50_000), Point(160_500, 50_500)], crs=5880)

    out = drop_offshore_islands(biomes, sites, offshore_km=20)
    geom = out.geometry.iloc[0]
    assert geom.area == pytest.approx(mainland.area + coastal.area + far_site.area)
    assert not geom.intersects(far_empty)


def test_anthropic_change_scale():
    b, c = S.ANTHROPIC_CHANGE_BOUNDS, S.ANTHROPIC_CHANGE_COLORS
    assert len(c) == len(b) + 1          # under + bins + over
    assert b == sorted(b) and 0 in b
    assert all(mcolors.is_color_like(x) for x in c) and len(set(c)) == len(c)


def test_landcover_composition():
    summary = _load_script("14_summary_tables")
    agg = pd.DataFrame({"unit": ["NATIONAL", "NATIONAL", "Pampa"], "radius_m": [100, 100, 100],
                        "year": [1985, 2025, 1985],
                        "area_natural_m2": [6.0, 4.0, 1.0], "area_anthropic_m2": [3.0, 5.0, 1.0],
                        "area_water_m2": [1.0, 1.0, 1.0], "area_with_data_m2": [10.0, 10.0, 3.0]})
    t = summary.landcover_composition(agg, [1985, 2025], 100).set_index(["year", "category"])
    assert t.loc[(1985, "Natural"), "pct"] == pytest.approx(60)
    assert t.loc[(2025, "Anthropic"), "pct"] == pytest.approx(50)
    assert t.groupby(level="year")["pct"].sum().tolist() == pytest.approx([100, 100])


def _reservoir_sites():
    return pd.DataFrame({
        "id_bem": [1, 2, 3, 4, 5],
        "start": 1985, "end": 2025,
        "biome": ["Cerrado", "Cerrado", "Amazônia", "Amazônia", "Amazônia"],
        "status": ["entirely_water_both", "entirely_water_end", "entirely_water_end",
                   "partial_gain", "no_gain"],
        "is_hydro": [True, True, False, True, False],
    })


def test_reservoir_water_biomes():
    summary = _load_script("14_summary_tables")
    t = summary.reservoir_water_biomes(_reservoir_sites()).set_index("unit")
    assert set(t.index) == {"NATIONAL"} | BIOMES                  # biomes without sites get zeros
    assert t.loc["NATIONAL", "n_associated"] == 5
    assert t.loc["NATIONAL", "n_entirely_water"] == 3
    assert t.loc["NATIONAL", "n_entirely_water_hydro"] == 2
    assert t.loc["Amazônia", ["n_entirely_water_end", "n_partial_gain", "n_no_gain"]].tolist() == [1, 1, 1]
    assert t.loc["Pantanal", "n_associated"] == 0


def test_reservoir_water_hydro_rank():
    summary = _load_script("14_summary_tables")
    links = pd.DataFrame({
        "id_bem": [1, 2, 2, 4], "res_id": [7, 7, 8, 8], "res_name": ["UHE X", "UHE X", "UHE Y", "UHE Y"],
        "biome": ["Cerrado", "Cerrado", "Cerrado", "Amazônia"], "is_hydro": True,
    })
    t = summary.reservoir_water_hydro(_reservoir_sites(), links, top_n=1)
    assert list(t["res_name"]) == ["UHE X", "UHE Y"]              # site 2 counts in both
    assert t["n_entirely_water"].tolist() == [2, 1]
    assert t["in_figure"].tolist() == [True, False]
    assert t.loc[1, "biomes"] == "Amazônia; Cerrado"
