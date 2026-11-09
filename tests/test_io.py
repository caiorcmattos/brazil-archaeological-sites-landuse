"""io.save tests: the three output kinds used by the pipeline scripts."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sitios_arq.io import save


@pytest.fixture
def gdf():
    return gpd.GeoDataFrame({"id_bem": [1, 2]}, geometry=[Point(-50, -10), Point(-51, -11)], crs=4674)


def test_save_parquet_csv_gpkg(tmp_path, gdf):
    df = pd.DataFrame({"a": [1, 2]})
    save(df, tmp_path / "x" / "t.parquet", index=False)
    save(df, tmp_path / "t.csv")
    save(gdf, tmp_path / "t.gpkg", layer="points")
    assert pd.read_parquet(tmp_path / "x" / "t.parquet").equals(df)
    assert pd.read_csv(tmp_path / "t.csv").equals(df)  # no index column by default
    assert len(gpd.read_file(tmp_path / "t.gpkg", layer="points")) == 2


def test_save_rejects_unknown_suffix(tmp_path):
    with pytest.raises(TypeError):
        save(pd.DataFrame({"a": [1]}), tmp_path / "t.gpkg")
