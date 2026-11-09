"""sitios_arq.paths: path resolution with and without SITIOS_DATA_ROOT/SITIOS_RASTER_DIR.

Each test sets (or clears) the env vars via monkeypatch and reloads the module,
since the constants are resolved at import time.
"""

import importlib

from sitios_arq import paths as P


def reload_paths():
    return importlib.reload(P)


def test_defaults_without_env_vars(monkeypatch):
    monkeypatch.delenv("SITIOS_DATA_ROOT", raising=False)
    monkeypatch.delenv("SITIOS_RASTER_DIR", raising=False)
    mod = reload_paths()

    assert mod.DATA_ROOT == mod.REPO_ROOT / "data"
    assert mod.RAW_DIR == mod.DATA_ROOT / "raw"
    assert mod.PROCESSED_DIR == mod.DATA_ROOT / "processed"
    assert mod.SHAPEFILES_DIR == mod.DATA_ROOT / "shapefiles"
    assert mod.RASTER_DIR == mod.RAW_DIR


def test_data_root_env_var_moves_every_derived_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SITIOS_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("SITIOS_RASTER_DIR", raising=False)
    mod = reload_paths()

    assert mod.DATA_ROOT == tmp_path
    assert mod.RAW_DIR == tmp_path / "raw"
    assert mod.PROCESSED_DIR == tmp_path / "processed"
    assert mod.RASTER_DIR == tmp_path / "raw"  # follows SITIOS_DATA_ROOT by default
    assert mod.BUFFER_DIR == tmp_path / "processed" / "buffers"
    assert mod.IPHAN_SITIOS_SHP == tmp_path / "raw/iphan_geoserver_2026-09-08/sitios/sitios.shp"


def test_raster_dir_env_var_overrides_default_independently(monkeypatch, tmp_path):
    monkeypatch.delenv("SITIOS_DATA_ROOT", raising=False)
    monkeypatch.setenv("SITIOS_RASTER_DIR", str(tmp_path / "LULC-Rasters/collection-11"))
    mod = reload_paths()

    assert mod.DATA_ROOT == mod.REPO_ROOT / "data"  # unaffected
    assert mod.RASTER_DIR == tmp_path / "LULC-Rasters/collection-11"


def test_both_env_vars_set(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    raster_dir = tmp_path / "rasters"
    monkeypatch.setenv("SITIOS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SITIOS_RASTER_DIR", str(raster_dir))
    mod = reload_paths()

    assert mod.DATA_ROOT == data_root
    assert mod.RASTER_DIR == raster_dir  # does not inherit from SITIOS_DATA_ROOT when set


def test_all_derived_paths_are_absolute(monkeypatch):
    monkeypatch.delenv("SITIOS_DATA_ROOT", raising=False)
    monkeypatch.delenv("SITIOS_RASTER_DIR", raising=False)
    mod = reload_paths()

    for name in [
        "DATA_ROOT", "RAW_DIR", "PROCESSED_DIR", "SHAPEFILES_DIR", "RASTER_DIR",
        "REF_RASTER_C11_1985", "IPHAN_SITIOS_SHP", "SHP_ESTADOS", "SHP_BIOMAS",
        "GPKG_BC250", "OUT_ESTADOS", "OUT_BIOMAS", "SITES_FILTERED", "BUFFER_DIR",
        "PIXEL_AREA_TIF", "EXTRACTION_C11_DIR",
    ]:
        assert getattr(mod, name).is_absolute(), name
