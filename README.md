# wfield-labdata

[``labdata``](https://github.com/jcouto/labdata) plugin for widefield calcium imaging analysis

**Analysis pipeline:** `WfieldParameters`, `WfieldStack`

**Atlas registration:** `WidefieldAtlas`, `WidefieldAtlasTransform`

**Stimulus responses:** `WidefieldResponse`, `WidefieldResponse.Projection`

---

## Prerequisites

- `labdata` installed and configured (`database`, `storage`, `scratch_path`, `local_paths` in `user_preferences.json`)
- `wfield` package installed (`pip install wfield` or from source)
- Widefield data already ingested into the `Widefield` table

---

## Plugin instalation

Like any other ``labdata`` plugin: Clone the repository and add the path to `~/.labdata/user_preferences.json` under `plugins`:

```json
{
  "plugins": {
    "pwfield@your_project": "/path/to/wfield-labdata/analysis"
  }
}
```

Then load it at the start of a session:

```python
from labdata import *
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
6. Computes pixelwise projections (`var`) stored in `WfieldStack.Projection`
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

# Load the raw npz (access all stored arrays)
res = entry.load()
res.files             # ['U', 'SVT', 'motion', 'mean_proj', ...]
res['SVTcorr']        # hemodynamic-corrected SVT (multi-channel only)
```

`open()` uses `SVTcorr` by default for multi-channel data. Pass `use_corrected=False` to use the raw `SVT` instead.

Add or query projections via the part table — `proj_name` is a free-form string:

```python
stack.projections['var']    # pixelwise variance (computed during make())

WfieldStack.Projection.insert1(dict(**key, wfield_analysis_id=0,
                                    proj_name='my_mask', proj=my_image))
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
| `atlas` | `'dorsal_cortex'` | Brain atlas name (legacy field, see `WidefieldAtlas`) |
| `exclude_mask` | `NULL` | Binary mask — pixels set to 1 are excluded from SVD |
| `functional_channel` | `0` | Channel index carrying the functional signal (470 nm) |
| `nframes_decimate` | `15` | Temporal decimation factor for the mean frame used in SVD |
| `chunk_size` | `512` | Frames processed per chunk |
| `mask_std_threshold` | `NULL` | If set, auto-generates a brain mask by thresholding pixel std |
| `match_to_session` | `NULL` | Optional link to a `Session` entry for behavioural alignment |

---

## Atlas registration

### WidefieldAtlas

Stores a pre-built CCF brain atlas reference. One entry per atlas name; shared across all recordings.

```python
# Load from pre-built wfield files (~/.wfield/)
WidefieldAtlas().from_wfield('dorsal_cortex')

# Or build directly from the Allen SDK (downloads CCF annotation if needed)
WidefieldAtlas().from_allensdk('dorsal_cortex') # this takes time
# Custom region list or resolution:
WidefieldAtlas().from_allensdk('my_atlas', structures=['VISp', 'SSp-bfd', 'AUDp'],
                                resolution=25)
```

Visualise:

```python
atlas = WidefieldAtlas & 'atlas_name="dorsal_cortex"'

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
atlas.plot_atlas(ax=ax, alpha=0.6)          # flat projection with mm extent, bregma at origin
atlas.plot_regions(ax=ax)                   # all region contours
atlas.plot_regions(ax=ax, acronyms=['VISp', 'SSp-bfd', 'RSPv'])  # selected regions only
```

| Field | Description |
|---|---|
| `atlas_name` | Unique identifier, e.g. `'dorsal_cortex'` |
| `ccf_regions` | Region contours as a dict (mm from bregma); reconstruct with `pd.DataFrame(row['ccf_regions'])` |
| `projection` | 2D ndarray — flattened dorsal projection of the annotation volume |
| `brain_outline` | 2D ndarray — brain boundary contour |
| `reference_point` | `[row, col]` of bregma in the atlas projection image (pixels) |
| `resolution` | mm per atlas projection pixel |

---

### WidefieldAtlasTransform

Links a `Widefield` recording to a `WidefieldAtlas` entry with a spatial transform. Supports two registration modes.

**Manual** — specify where bregma is in the widefield image and adjust scale/rotation until the contours align:

```python
WidefieldAtlasTransform.insert1(dict(
    **widefield_key,
    atlas_name='dorsal_cortex',
    atlas_transform_id=0,
    transform_type='manual',
    reference_point=[320, 240],   # bregma [col, row] in widefield pixels
    resolution=0.025,             # mm per widefield pixel
    rotation=5.0,                 # degrees CCW
    scale=1.0,
    ratio=1.0,
))
```

**Landmarks** — match named point pairs between atlas and widefield image; the transform is computed automatically:

```python
import pandas as pd

# atlas landmarks in mm from bregma (x, y)
lm = pd.DataFrame({'x': [-1.5, 0.0,  2.0], 'y': [-2.0, 0.5, -1.0],
                   'name': ['lm0', 'lm1', 'lm2'], 'color': ['k']*3})
# corresponding pixel coordinates in the widefield image (col, row)
lm_match = pd.DataFrame({'x': [150, 320, 480], 'y': [310, 240, 290],
                          'name': ['lm0', 'lm1', 'lm2'], 'color': ['k']*3})

WidefieldAtlasTransform.insert1(dict(
    **widefield_key,
    atlas_name='dorsal_cortex',
    atlas_transform_id=0,
    transform_type='landmarks',
    landmarks=lm.to_dict(orient='list'),
    landmarks_match=lm_match.to_dict(orient='list'),
))
```

Use the resulting transform:

```python
xfm = WidefieldAtlasTransform & widefield_key & 'atlas_name="dorsal_cortex"' & 'atlas_transform_id=0'

M = xfm.get_transform()          # 3×3 ndarray, atlas mm → widefield pixels
regions = xfm.transform_regions() # ccf_regions DataFrame in widefield pixel coordinates

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(mean_proj, cmap='gray')
xfm.plot_regions(ax=ax)
xfm.plot_regions(ax=ax, acronyms=['VISp', 'RSPv'])
```

The dashboard **Atlas Alignment** tab provides an interactive GUI for both modes, with live preview.

| Field | Description |
|---|---|
| `atlas_transform_id` | Integer; allows multiple transforms per recording × atlas pair |
| `transform_type` | `'manual'` or `'landmarks'` |
| `reference_point` | `[col, row]` of bregma in the widefield image (manual path) |
| `resolution` | mm per widefield pixel (manual path) |
| `rotation` | Degrees counter-clockwise (manual path) |
| `scale` | Isotropic scale on top of `1/resolution` (manual path) |
| `ratio` | X/Y aspect ratio correction (manual path) |
| `landmarks` | Atlas-space landmarks as dict with keys `x`, `y`, `name`, `color` (landmarks path) |
| `landmarks_match` | Corresponding widefield pixel coordinates, same format (landmarks path) |
| `transform_matrix` | Cached 3×3 float64 matrix (atlas mm → widefield px); populated on first `get_transform()` call |

---

## Stimulus responses

### WidefieldResponse

Stores named analysis results tied to a `Widefield` recording — independent of whether a `WfieldStack` was computed. Typical use cases: retinotopy maps, orientation preference maps, dF/F triggered averages.

```python
# Insert an entry with projections (e.g. retinotopy)
WidefieldResponse.insert1(dict(
    **widefield_key,
    stim_name='retinotopy',
))

entry = WidefieldResponse & widefield_key & 'stim_name="retinotopy"'

# Add projections one at a time
entry.save_projection(phase_map,     'phase')
entry.save_projection(magnitude_map, 'magnitude')
entry.save_projection(sign_map,      'sign_map')

# Load them all back
projs = entry.load_projections()   # {'phase': ..., 'magnitude': ..., 'sign_map': ...}
```

Insert with a movie (numpy array or dict of arrays — saved as `.npz` and uploaded automatically):

```python
WidefieldResponse.insert1(
    dict(**widefield_key, stim_name='retinotopy'),
    movie=np.random.rand(100, 256, 256).astype('float32'),   # single array
)

# Or store multiple arrays in one file
WidefieldResponse.insert1(
    dict(**widefield_key, stim_name='retinotopy'),
    movie={'azimuth': azi_movie, 'elevation': elev_movie},
)

# Load back
data = entry.load_movie()   # numpy file (.npz) access fields by key
data['azimuth']
```

To record which `WfieldStack` was used to compute the response, set the optional `wfield_analysis_id` attribute:

```python
WidefieldResponse.insert1(dict(
    **widefield_key,
    stim_name='retinotopy',
    wfield_analysis_id=0,
))
```

Deleting an entry also removes its movie from `AnalysisFile` and S3. Pass `keep_analysis=True` to skip that:

```python
(WidefieldResponse & widefield_key & 'stim_name="retinotopy"').delete()
(WidefieldResponse & widefield_key & 'stim_name="retinotopy"').delete(keep_analysis=True)
```

| Field | Description |
|---|---|
| `stim_name` | Primary key; descriptive name, e.g. `'retinotopy'`, `'orientation'` |
| `wfield_analysis_id` | Optional; records which `WfieldStack` the response was derived from |
| `file_path` / `storage` | Auto-populated from `AnalysisFile` when a movie is provided |

**`WidefieldResponse.Projection`** stores 2D images under arbitrary names:

| Field | Description |
|---|---|
| `proj_name` | Free-form name, e.g. `'phase'`, `'magnitude'`, `'sign_map'`, `'dff'` |
| `proj` | 2D ndarray (any dtype — uint8 RGB images are supported) |

`WidefieldResponse` projections are also available as background images in the dashboard **Atlas Alignment** tab, making it straightforward to overlay atlas regions on retinotopy or other functional maps.
