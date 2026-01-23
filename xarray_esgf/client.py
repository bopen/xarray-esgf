import asyncio
import dataclasses
import logging
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, get_args

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
BOUNDS_DIMS = {"axis_nbounds", "bnds", "nbnd"}

LOGGER = logging.getLogger()


def use_new_combine_kwarg_defaults[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwds: P.kwargs) -> T:
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            return func(*args, **kwds)

    return wrapper


def dataset_id_to_dict(dataset_id: str) -> dict[DATASET_ID_KEYS, str]:
    keys = get_args(DATASET_ID_KEYS)
    return dict(zip(keys, dataset_id.split("."), strict=True))


def combine_datasets(datasets: list[Dataset]) -> Dataset:
    obj = xr.combine_by_coords(
        datasets,
        join="exact",
        combine_attrs="drop_conflicts",
    )
    if isinstance(obj, DataArray):
        return obj.to_dataset()
    return obj


def move_dimensionless_coords_to_attrs(ds: Dataset) -> Dataset:
    attrs = {}
    for var, da in ds.coords.items():
        if not da.dims:
            attrs[var] = da.item()
    ds = ds.drop_vars(list(attrs))
    for da in ds.data_vars.values():
        da.attrs.update(attrs)
    return ds


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

    def _open_datasets(
        self,
        concat_dims: DATASET_ID_KEYS | Iterable[DATASET_ID_KEYS] | None,
        drop_variables: str | Iterable[str] | None,
        download: bool,
        show_progress: bool,
        sel: dict[Hashable, Any],
        ignore_spatial_coords: str | Iterable[str],
    ) -> dict[str, Dataset]:
        sel = {
            k: slice(*v["slice"]) if isinstance(v, dict) else v for k, v in sel.items()
        }

        if isinstance(concat_dims, str):
            concat_dims = [concat_dims]
        concat_dims = concat_dims or []

        if isinstance(ignore_spatial_coords, str):
            ignore_spatial_coords = {ignore_spatial_coords}
        ignore_spatial_coords = set(ignore_spatial_coords)

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
                storage_options={"ssl": self.verify_ssl},
            )

            ds = ds.sel({k: v for k, v in sel.items() if k in ds.dims})

            if ignore_spatial_coords.intersection(ds.variables):
                ds = ds.drop_vars(set(ds.variables) & {"lat", "lon"})

            if all(ds.sizes.values()):
                grouped_objects[file.dataset_id].append(ds.drop_encoding())

        combined_datasets = {}
        for dataset_id, datasets in grouped_objects.items():
            dataset_id_dict = dataset_id_to_dict(dataset_id)
            if len(datasets) == 1:
                (ds,) = datasets
            else:
                ds = combine_datasets(datasets)

            ds = ds.set_coords([
                name
                for name, da in ds.variables.items()
                if BOUNDS_DIMS.intersection(da.dims) or "time" not in da.dims
            ])

            ds = move_dimensionless_coords_to_attrs(ds)

            ds = ds.expand_dims({dim: [dataset_id_dict[dim]] for dim in concat_dims})
            combined_datasets[dataset_id] = ds
            LOGGER.debug(f"{dataset_id}: {dict(ds.sizes)}")

        return combined_datasets

    @use_new_combine_kwarg_defaults
    def open_dataset(
        self,
        concat_dims: DATASET_ID_KEYS | Iterable[DATASET_ID_KEYS] | None,
        drop_variables: str | Iterable[str] | None = None,
        download: bool = False,
        show_progress: bool = True,
        sel: dict[Hashable, Any] | None = None,
        ignore_spatial_coords: str | Iterable[str] | None = None,
    ) -> Dataset:
        combined_datasets = self._open_datasets(
            concat_dims=concat_dims,
            drop_variables=drop_variables,
            download=download,
            show_progress=show_progress,
            sel=sel or {},
            ignore_spatial_coords=ignore_spatial_coords or {},
        )

        obj = combine_datasets([ds.reset_coords() for ds in combined_datasets.values()])

        coords: set[Hashable] = set()
        for ds in combined_datasets.values():
            coords.update(ds.coords)
        obj = obj.set_coords(coords)

        for name, var in obj.variables.items():
            if name not in obj.dims:
                var.encoding["preferred_chunks"] = dict(var.chunksizes)

        obj.attrs["coordinates"] = " ".join(sorted(str(coord) for coord in obj.coords))
        obj.attrs["dataset_ids"] = sorted(combined_datasets)
        return obj
