from pathlib import Path

from xarray_esgf import Client


def test_missing_files(tmp_path: Path) -> None:
    selection = {"query": '"tas_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_201901-201912.nc"'}
    client = Client(
        selection,
        esgpull_path=str(tmp_path / "esgpull"),
    )
    assert len(client.missing_files) == 1
    downloaded = client.download()
    assert len(downloaded) == 1
    assert len(client.missing_files) == 0

    downloaded = client.download()
    assert len(downloaded) == 0
