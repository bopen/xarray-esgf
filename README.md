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

## License

```
Copyright 2025, B-Open Solutions srl.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
