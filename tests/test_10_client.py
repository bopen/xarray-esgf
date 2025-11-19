from pathlib import Path

from xarray_esgf import Client


def test_download(tmp_path: Path, index_node: str) -> None:
    selection: dict[str, str | list[str]] = {
        "query": '"tas_Amon_EC-Earth3-CC_ssp245_r1i1p1f1_gr_201901-201912.nc"'
    }
    client = Client(
        selection,
        esgpull_path=str(tmp_path / "esgpull"),
        index_node=index_node,
    )

    downloaded = client.download()
    assert len(downloaded) == 1

    downloaded = client.download()
    assert len(downloaded) == 0
