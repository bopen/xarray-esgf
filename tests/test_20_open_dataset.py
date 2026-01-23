import contextlib
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pytest
import xarray as xr
from xarray import AlignmentError

does_not_raise = contextlib.nullcontext


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
            '"CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.fx.areacella.gr.v20210113.areacella_fx_EC-Earth3-CC_ssp245_r1i1p1f1_gr.nc"',
            '"CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.fx.areacella.gr.v20210113.areacella_fx_EC-Earth3-CC_ssp585_r1i1p1f1_gr.nc"',
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

    # Chunks
    for dim in ds.dims:
        assert not ds[dim].chunks
    assert ds.chunksizes == {
        "experiment_id": (1, 1),
        "time": (12, 12),
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

    # Dimensionless coords
    assert "height" not in ds["pr"].attrs
    assert ds["tas"].attrs["height"] == 2.0

    # Data vars
    assert set(ds.data_vars) == {"pr", "tas"}
    assert ds["pr"].coordinates == "areacella experiment_id lat lon time"
    assert ds["tas"].coordinates == "areacella experiment_id lat lon time"

    # Attributes
    assert (
        ds.coordinates
        == "areacella experiment_id lat lat_bnds lon lon_bnds time time_bnds"
    )
    assert ds.dataset_ids == [
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.Amon.tas.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp245.r1i1p1f1.fx.areacella.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.pr.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.Amon.tas.gr.v20210113",
        "CMIP6.ScenarioMIP.EC-Earth-Consortium.EC-Earth3-CC.ssp585.r1i1p1f1.fx.areacella.gr.v20210113",
    ]


def test_combine_coords(tmp_path: Path, index_node: str) -> None:
    esgpull_path = tmp_path / "esgpull"
    selection = {
        "query": [
            '"areacella_fx_IPSL-CM6A-LR_historical_r1i1p1f1_gr.nc"',
            '"orog_fx_IPSL-CM6A-LR_historical_r1i1p1f1_gr.nc"',
        ]
    }
    ds = xr.open_dataset(
        selection,  # type: ignore[arg-type]
        esgpull_path=esgpull_path,
        concat_dims="experiment_id",
        engine="esgf",
        index_node=index_node,
        chunks={},
    )
    assert set(ds.coords) == {"areacella", "lat", "lon", "experiment_id", "orog"}
    assert not ds.data_vars


@pytest.mark.parametrize(
    "sel,expected_size",
    [
        ({}, 12),
        ({"time": "2019-01"}, 1),
        ({"time": {"slice": ["2019-01", "2019-02"]}}, 2),
    ],
)
def test_time_selection(
    tmp_path: Path,
    index_node: str,
    sel: dict[Hashable, Any],
    expected_size: int,
) -> None:
    esgpull_path = tmp_path / "esgpull"
    selection = {
        "query": [
            '"tas_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_201901-201912.nc"',
        ]
    }
    ds = xr.open_dataset(
        selection,  # type: ignore[arg-type]
        esgpull_path=esgpull_path,
        engine="esgf",
        index_node=index_node,
        chunks={},
        sel=sel,
    )
    assert ds.sizes["time"] == expected_size


@pytest.mark.parametrize(
    "ignore_spatial_coords, raises",
    [
        ("areacella", does_not_raise()),
        (None, pytest.raises(AlignmentError)),
    ],
)
def test_ignore_spatial_coords(
    tmp_path: Path,
    index_node: str,
    ignore_spatial_coords: str | None,
    raises: contextlib.nullcontext,
) -> None:
    esgpull_path = tmp_path / "esgpull"
    selection = {
        "query": [
            '"CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical.r1i1p1f1.fx.areacella.gn.v20190710.areacella_fx_MPI-ESM1-2-HR_historical_r1i1p1f1_gn.nc"',
            '"CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR.historical.r1i1p1f1.Amon.tas.gn.v20190710.tas_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001-185412.nc"',
        ]
    }

    with raises:
        ds = xr.open_dataset(
            selection,  # type: ignore[arg-type]
            esgpull_path=esgpull_path,
            engine="esgf",
            index_node=index_node,
            chunks={},
            ignore_spatial_coords=ignore_spatial_coords,
        )
        assert {"lat", "lon"} <= set(ds.variables)
