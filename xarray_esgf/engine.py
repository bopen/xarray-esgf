from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any

from xarray import Dataset
from xarray.backends import BackendEntrypoint

from .client import DATASET_ID_KEYS, Client


class EsgfBackendEntrypoint(BackendEntrypoint):
    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj: dict[str, str | list[str]],
        *,
        drop_variables: str | Iterable[str] | None = None,
        esgpull_path: str | Path | None = None,
        index_node: str | None = None,
        retries: int = 0,
        check_files: bool = True,
        verify_ssl: bool = False,
        concat_dims: DATASET_ID_KEYS | Iterable[DATASET_ID_KEYS] | None = None,
        download: bool = False,
        show_progress: bool = True,
        sel: dict[Hashable, Any] | None = None,
    ) -> Dataset:
        client = Client(
            selection=filename_or_obj,
            esgpull_path=esgpull_path,
            index_node=index_node,
            retries=retries,
            check_files=check_files,
            verify_ssl=verify_ssl,
        )
        return client.open_dataset(
            concat_dims=concat_dims,
            drop_variables=drop_variables,
            download=download,
            show_progress=show_progress,
            sel=sel,
        )

    open_dataset_parameters = (
        "filename_or_obj",
        "esgpull_path",
        "index_node",
        "concat_dims",
    )

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        return isinstance(filename_or_obj, dict)

    description = "Open ESGF data using Xarray"

    url = "https://github.com/bopen/xarray-esgf"
