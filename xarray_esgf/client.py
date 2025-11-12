import asyncio
import dataclasses
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Literal, get_args

import xarray as xr
from esgpull import Esgpull, File, Query
from xarray import DataArray, Dataset

DATASET_ID_KEYS = Literal[
    "project",
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "variant_label",
    "table_id",
    "variable_id",
    "grid_label",
    "version",
]


def use_new_combine_kwarg_defaults[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwds: P.kwargs) -> T:
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            return func(*args, **kwds)

    return wrapper


def dataset_id_to_dict(dataset_id: str) -> dict[DATASET_ID_KEYS, str]:
    return dict(zip(get_args(DATASET_ID_KEYS), dataset_id.split("."), strict=True))


@dataclasses.dataclass
class Client:
    esgpull_path: str
    index_node: str
    selection: dict[str, str | list[str]]

    @cached_property
    def _client(self) -> Esgpull:
        client = Esgpull(path=self.esgpull_path, install=True)
        client.config.download.disable_ssl = True
        client.config.api.index_node = self.index_node
        return client

    @cached_property
    def _query(self) -> Query:
        return Query(
            selection=self.selection,
            options={"distrib": True, "latest": True},
        )

    @cached_property
    def files(self) -> list[File]:
        return self._client.context.files(
            self._query, max_hits=None, keep_duplicates=False
        )

    def _get_local_path(self, file: File) -> Path:
        return self._client.fs.paths.data / file.local_path / file.filename

    @cached_property
    def local_paths(self) -> dict[str, list[Path]]:
        datasets = defaultdict(list)
        for file in self.files:
            datasets[file.dataset_id].append(self._get_local_path(file))
        return dict(datasets)

    def download(self) -> None:
        _, errors = asyncio.run(self._client.download(self.files, use_db=False))
        exceptions = []
        for error in errors:
            err = error.err
            if isinstance(err, Exception):
                exceptions.append(err)
        if exceptions:
            raise ExceptionGroup("Errors", exceptions)

    @use_new_combine_kwarg_defaults
    def open_dataset(self, concat_dims: list[DATASET_ID_KEYS]) -> Dataset:
        datasets = []
        for dataset_id, paths in self.local_paths.items():
            dataset_id_dict = dataset_id_to_dict(dataset_id)
            ds = xr.open_mfdataset(paths, chunks={}, engine="netcdf4")
            ds = ds.expand_dims({dim: [dataset_id_dict[dim]] for dim in concat_dims})
            datasets.append(ds)
        obj = xr.combine_by_coords(
            datasets, join="exact", combine_attrs="drop_conflicts"
        )
        if isinstance(obj, DataArray):
            return obj.to_dataset()
        return obj
