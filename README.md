# xarray-esgf

Xarray backend for ESGF data

## Quick start

```python
import xarray as xr

ESGPULL_PATH =  # Set path to download data

ds = xr.open_dataset(
    {
        "project": "CMIP6",
        "experiment_id": "ssp*",
        "source_id": "EC-Earth3-CC",
        "frequency": "mon",
        "variable_id": ["tas", "pr"],
        "variant_label": "r1i1p1f1",
    },
    concat_dims="experiment_id",
    esgpull_path=ESGPULL_PATH,
    index_node="esgf.ceda.ac.uk",
    engine="esgf",
)
```
