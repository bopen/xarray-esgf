from pathlib import Path

import pytest
import xarray as xr


@pytest.mark.parametrize("download", [True, False])
def test_open_dataset(tmp_path: Path, index_node: str, download: bool) -> None:
    esgpull_path = tmp_path / "esgpull"
    selection = {
        "query": [
            '"tas_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_201901-201912.nc"',
            '"tas_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_202001-202012.nc"',
            '"tas_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_201901-201912.nc"',
            '"tas_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_202001-202012.nc"',
            '"pr_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_201901-201912.nc"',
            '"pr_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_202001-202012.nc"',
            '"pr_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_201901-201912.nc"',
            '"pr_Amon_EC-Earth3-CC_ssp585_r1i1p1f1_gr_202001-202012.nc"',
        ]
    }
    ds = xr.open_dataset(
        selection,  # type: ignore[arg-type]
        esgpull_path=esgpull_path,
        concat_dims="experiment_id",
        engine="esgf",
        index_node=index_node,
        download=download,
        chunks={},
    )
    assert (esgpull_path / "data" / "CMIP6").exists() is download

    # Dims
    assert ds.sizes == {
        "experiment_id": 2,
        "time": 24,
        "bnds": 2,
        "lat": 256,
        "lon": 512,
    }

    # Coords
    assert set(ds.coords) == {
        "experiment_id",
        "height",
        "lat",
        "lat_bnds",
        "lon",
        "lon_bnds",
        "time",
        "time_bnds",
    }
    assert set(ds[["lat_bnds", "lon_bnds", "time_bnds"]].dims) == {
        "bnds",
        "lat",
        "lon",
        "time",
    }

    # Data vars
    assert set(ds.data_vars) == {"tas", "pr"}

    # Attributes
    assert ds.dataset_ids == [
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.tas.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.tas.gr.v20210113",
    ]


def test_open_dataset_check_dims(tmp_path: Path) -> None:
    esgpull_path = tmp_path / "esgpull"
    selection = {
        "query": [
            '"tos_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_201501-201512.nc"',
            '"tos_Omon_EC-Earth3-CC_ssp245_r1i1p1f1_gn_201501-201512.nc"',
        ]
    }
    with pytest.raises(ValueError, match="Dimensions do not match"):
        xr.open_dataset(
            selection,  # type: ignore[arg-type]
            esgpull_path=esgpull_path,
            engine="esgf",
            download=True,
            chunks={},
        )
