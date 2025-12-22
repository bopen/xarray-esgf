import asyncio
import dataclasses
import logging
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

LOGGER = logging.getLogger()


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
    retries: int = 0
    check_files: bool = True
    verify_ssl: bool = False

    @cached_property
    def _client(self) -> Esgpull:
        client = Esgpull(
            path=self.esgpull_path,
            install=True,
            load_db=False,
        )
        client.config.download.disable_ssl = not self.verify_ssl
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
    def n_tries(self) -> int:
        return self.retries + 1 if self.retries >= 0 else 1

    @cached_property
    def files(self) -> list[File]:
        return self._client.context.files(
            self._query,
            max_hits=None,
            keep_duplicates=False,
        )

    @property
    def missing_files(self) -> list[File]:
        missing_files = []
        for file in tqdm.tqdm(self.files, desc="Looking for missing files"):
            file_path = Path(str(self._client.fs[file]))
            if (self.check_files and self._client.fs.check(file) != FileCheck.Ok) or (
                not self.check_files and not file_path.exists()
            ):
                missing_files.append(file)
        return missing_files

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
                chunks=-1,
                engine="h5netcdf",
                drop_variables=drop_variables,
                storage_options={"verify_ssl": self.verify_ssl},
            )
            grouped_objects[file.dataset_id].append(ds.drop_encoding())

        combined_datasets = {}
        for dataset_id, datasets in grouped_objects.items():
            dataset_id_dict = dataset_id_to_dict(dataset_id)
            if len(datasets) == 1:
                (ds,) = datasets
            else:
                ds = xr.concat(
                    datasets,
                    dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                )
            ds = ds.set_coords([
                name
                for name, da in ds.variables.items()
                if "bnds" in da.dims or "time" not in da.dims
            ])
            ds = ds.expand_dims({dim: [dataset_id_dict[dim]] for dim in concat_dims})
            combined_datasets[dataset_id] = ds
            LOGGER.debug(f"{dataset_id}: {dict(ds.sizes)}")

        obj = xr.combine_by_coords(
            combined_datasets.values(),
            join="exact",
            combine_attrs="drop_conflicts",
        )
        if isinstance(obj, DataArray):
            obj = obj.to_dataset()
        obj.attrs["dataset_ids"] = sorted(grouped_objects)

        for name, var in obj.variables.items():
            if name not in obj.dims:
                var.encoding["preferred_chunks"] = dict(var.chunksizes)

        return obj
