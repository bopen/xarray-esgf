import asyncio
import dataclasses
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import cached_property
from pathlib import Path
from typing import Literal, get_args

import tqdm
import xarray as xr
from esgpull import Esgpull, File, Query
from esgpull.fs import FileCheck
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
    keys = get_args(DATASET_ID_KEYS)
    return dict(zip(keys, dataset_id.split("."), strict=True))


@dataclasses.dataclass
class Client:
    selection: dict[str, str | list[str]]
    esgpull_path: str | Path | None = None
    index_node: str | None = None
    n_tries: int = 1

    @cached_property
    def _client(self) -> Esgpull:
        client = Esgpull(
            path=self.esgpull_path,
            install=True,
            load_db=False,
        )
        client.config.download.disable_ssl = True
        if self.index_node is not None:
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
            self._query,
            max_hits=None,
            keep_duplicates=False,
        )

    @property
    def missing_files(self) -> list[File]:
        return [
            file for file in self.files if self._client.fs.check(file) != FileCheck.Ok
        ]

    def download(self) -> list[File]:
        files = []
        for _ in range(self.n_tries):
            downloaded, errors = asyncio.run(
                self._client.download(self.missing_files, use_db=False)
            )
            files.extend(downloaded)
            if not errors:
                break

        exceptions = []
        for error in errors:
            err = error.err
            if isinstance(err, Exception):
                exceptions.append(err)
        if exceptions:
            msg = "Download errors"
            raise ExceptionGroup(msg, exceptions)
        return files

    @use_new_combine_kwarg_defaults
    def open_dataset(
        self,
        concat_dims: DATASET_ID_KEYS | Iterable[DATASET_ID_KEYS] | None,
        drop_variables: str | Iterable[str] | None = None,
        download: bool = False,
        show_progress: bool = True,
    ) -> Dataset:
        if isinstance(concat_dims, str):
            concat_dims = [concat_dims]
        concat_dims = concat_dims or []

        if download:
            self.download()

        grouped_objects = defaultdict(list)
        for file in tqdm.tqdm(
            self.files, disable=not show_progress, desc="Opening datasets"
        ):
            ds = xr.open_dataset(
                self._client.fs[file].drs if download else file.url,
                chunks={},
                engine="h5netcdf",
                drop_variables=drop_variables,
            )
            grouped_objects[file.dataset_id].append(ds)

        combined_datasets = []
        for dataset_id, datasets in grouped_objects.items():
            dataset_id_dict = dataset_id_to_dict(dataset_id)
            ds = xr.concat(
                datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="drop_conflicts",
            )
            ds = ds.expand_dims({dim: [dataset_id_dict[dim]] for dim in concat_dims})
            ds = ds.set_coords([
                name for name, da in ds.variables.items() if "bnds" in da.dims
            ])
            combined_datasets.append(ds)

        obj = xr.combine_by_coords(
            combined_datasets,
            join="exact",
            combine_attrs="drop_conflicts",
        )
        if isinstance(obj, DataArray):
            obj = obj.to_dataset()
        obj.attrs["dataset_ids"] = sorted(grouped_objects)
        return obj
