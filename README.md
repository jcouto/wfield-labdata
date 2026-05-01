# wfield-labdata

[``labdata``](https://github.com/jcouto/labdata) plugin for widefield calcium imaging analysis

Tables:
- `WfieldParameters` — manual table holding analysis configuration
- `WfieldStack` — computed table that runs the full pipeline and stores results

---

## Prerequisites

- `labdata` installed and configured (`database`, `storage`, `scratch_path`, `local_paths` in `user_preferences.json`)
- `wfield` package installed (`pip install wfield` or from source)
- Widefield data already ingested into the `Widefield` table

---

## Plugin registration

Add the plugin to `~/.labdata/user_preferences.json` under `plugins`:

```json
{
  "plugins": {
    "pwfield@your_project": "/path/to/wfield-labdata/analysis"
  }
}
```

Then load it at the start of a session:

```python
# this will load the plugin to the namespace
from labdata import *
# Show the WfieldParameters table
pwfield.WfieldParameters()

```

---

## Processing a widefield dataset

### Step 1 — Find the Widefield entry

```python
Widefield()   # list all entries

key = dict(subject_name='subject001', session_name='2024-01-15')
key = (Widefield & key).fetch1("KEY")
```

### Step 2 — Insert WfieldParameters

```python
WfieldParameters.insert1(dict(**key, wfield_analysis_id=0))
```

All fields have sensible defaults (see reference table below). Override only what you need:

```python
WfieldParameters.insert1(dict(
    **key,
    wfield_analysis_id=1,
    k=100,                    # fewer SVD components
    mask_std_threshold=3,     # auto-generate brain mask
    motion_correction='none', # skip motion correction
))
```

`wfield_analysis_id` lets you store multiple parameter sets for the same dataset — increment it for each run.

### Step 3 — Run the analysis

```python
# Run for this specific entry
WfieldStack.populate(key & 'wfield_analysis_id=0')

# Or populate all pending entries
WfieldStack.populate(display_progress=True)
```

The pipeline:
1. Motion correction (default: ECC algorithm)
2. Mean projection across frames
3. Brain mask generation
4. Approximate SVD decomposition
5. Hemodynamic correction (multi-channel data only)
6. Computes pixelwise projections (`mean`, `std`, `var`) stored in `WfieldStack.Projection`
7. Saves `U`, `SVT`, `motion`, `mean_proj` (+ `SVTcorr`/`T`/`rcoeffs` if multi-channel) to a `.npz` file uploaded via `AnalysisFile`

### Step 4 — Load results

```python
entry = WfieldStack & key & 'wfield_analysis_id=0'

# Load as SVDStack (reconstructs frames on demand)
stack = entry.open()

stack.shape          # (nframes, height, width)
stack[0]             # single frame, shape (H, W)
stack[0:100]         # 100 frames, shape (100, H, W)
stack.mean()         # pixelwise mean image
stack.std()          # pixelwise std image
stack.mean_proj      # mean projection stored during make()
stack.motion         # XY motion shifts, shape (nframes, 2, nchannels)
stack.fs             # frame rate (Hz)

# Get timecourse for a region
import numpy as np
mask = stack.mean_proj[0] > 500   # example mask
timecourse = stack.get_timecourse(np.where(mask))

# Load the raw npz (access all stored arrays)
res = entry.load()
res.files             # ['U', 'SVT', 'motion', 'mean_proj', ...]
res['SVTcorr']        # hemodynamic-corrected SVT (multi-channel only)
res['rcoeffs']        # regression coefficients from hemodynamic correction
```

`open()` uses `SVTcorr` by default for multi-channel data. Pass `use_corrected=False` to use the raw `SVT` instead.

Projections stored in `WfieldStack.Projection` are attached to the stack as a dict keyed by `proj_name`:

```python
stack.projections['var']    # pixelwise variance (computed during make())
```

You can query or add projections directly — `proj_name` is a free-form string, so any name is valid:

```python
# Fetch all projections for an entry
(WfieldStack.Projection & key & 'wfield_analysis_id=0').fetch(as_dict=True)

# Insert a custom projection
WfieldStack.Projection.insert1(dict(**key, wfield_analysis_id=0,
                                    proj_name='my_mask',
                                    proj=my_image))
```

---

## WfieldParameters field reference

| Field | Default | Description |
|---|---|---|
| `wfield_analysis_id` | — | Integer key; increment to store multiple runs per dataset |
| `motion_correction` | `'ecc'` | Algorithm: `'2d'`, `'ecc'`, `'normcorr'`, `'none'` |
| `motion_conv_kernel` | `NULL` | Optional convolution kernel applied before motion correction |
| `decomposition` | `'approx'` | SVD algorithm: `'approx'` (fast) or `'pmd'` |
| `k` | `200` | Number of SVD components |
| `atlas` | `'dorsal_cortex'` | Brain atlas for registration |
| `exclude_mask` | `NULL` | Binary mask (same shape as frame) — pixels set to 1 are excluded from SVD |
| `functional_channel` | `0` | Channel index carrying the functional signal (470 nm) |
| `nframes_decimate` | `15` | Temporal decimation factor used to compute the mean frame for SVD |
| `chunk_size` | `512` | Number of frames processed per chunk |
| `mask_std_threshold` | `NULL` | If set, automatically generates a brain mask by thresholding pixel std |
| `match_to_session` | `NULL` | Optional link to a `Session` entry for behavioral alignment |
