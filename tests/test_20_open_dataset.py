from pathlib import Path

import dask
import pytest
import xarray as xr


class CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    Reference: https://stackoverflow.com/questions/53289286/"""

    def __init__(self, max_computes: int = 0) -> None:
        self.total_computes = 0
        self.max_computes = max_computes

    def __call__(self, dsk, keys, **kwargs):  # type: ignore[no-untyped-def]
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            msg = f"Too many computes. Total: {self.total_computes:d} > max: {self.max_computes:d}."
            raise RuntimeError(msg)
        return dask.get(dsk, keys, **kwargs)


def raise_if_dask_computes(max_computes: int = 0) -> dask.config.set:
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)


@pytest.mark.parametrize(
    "download",
    [True, False],
)
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
            '"CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.fx.areacella.gr.v20210113.areacella_fx_EC-Earth3-CC_ssp245_r1i1p1f1_gr.nc"',
            '"CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.fx.areacella.gr.v20210113.areacella_fx_EC-Earth3-CC_ssp585_r1i1p1f1_gr.nc"',
        ]
    }
    with raise_if_dask_computes():
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

    # Chunks
    for dim in ds.dims:
        assert not ds[dim].chunks
    assert ds.chunksizes == {
        "experiment_id": (1, 1),
        "time": (24,),
        "lat": (256,),
        "lon": (512,),
        "bnds": (2,),
    }

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
        "areacella",
        "experiment_id",
        "height",
        "lat",
        "lat_bnds",
        "lon",
        "lon_bnds",
        "time",
        "time_bnds",
    }
    assert all(
        "experiment_id" not in coord.dims
        for name, coord in ds.coords.items()
        if name != "experiment_id"
    )

    # Data vars
    assert set(ds.data_vars) == {"tas", "pr"}

    # Attributes
    assert ds.dataset_ids == [
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.tas.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.fx.areacella.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.tas.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.fx.areacella.gr.v20210113",
    ]

    # Compute
    with raise_if_dask_computes(1):
        ds.compute()
