from pathlib import Path

import xarray as xr


def test_open_dataset(tmp_path: Path) -> None:
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
        selection,
        esgpull_path=str(tmp_path / "esgpull"),
        concat_dims="experiment_id",
        engine="esgf",
    )
    assert ds.sizes == {
        "experiment_id": 2,
        "time": 24,
        "bnds": 2,
        "lat": 256,
        "lon": 512,
    }
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
    assert set(ds.data_vars) == {"tas", "pr"}
    assert ds.dataset_ids == [
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.tas.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.tas.gr.v20210113",
    ]
