import base64
import io
import numpy as np
import pandas as pd
import streamlit as st

_ANALYSED_COLOR = '#d4f5d4'  # light green for sessions with completed WfieldStack


def _normalize_image(img):
    img = np.asarray(img, dtype=float)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return (img * 255).clip(0, 255).astype(np.uint8)


def _normalize_with_pct(img, lo_pct=2.0, hi_pct=99.0):
    img = np.asarray(img, dtype=float)
    lo = np.percentile(img, lo_pct)
    hi = np.percentile(img, hi_pct)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _to_base64(arr):
    from PIL import Image as PILImage
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        pass  # already (H, W, 3) RGB — use as-is
    elif arr.ndim == 3:
        arr = arr[0]   # multi-channel widefield: take first channel
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = np.stack([arr] * 3, axis=-1)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _altair_image(url, w, h, title='', width=400, height=None):
    import altair as alt
    if height is None:
        height = max(1, int(width * h / w))
    img_df = pd.DataFrame([{'x': 0, 'y': 0, 'x2': w, 'y2': h, 'url': url}])
    return (
        alt.Chart(img_df).mark_image(aspect=False).encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[0, w]), axis=None),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[h, 0]), axis=None),
            x2=alt.X2('x2'), y2=alt.Y2('y2'), url='url:N',
            tooltip=alt.value(None),
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def _tab_cache_factory(refresh_key):
    """Return a decorator like ``st.cache_data`` that also wires a cache into a
    per-tab refresh button. When the tab's refresh flag is set (the button was
    clicked on the previous run), each cache is cleared as it is recreated — so
    only the current tab's data reloads, leaving other tabs' caches warm."""
    do_clear = st.session_state.pop(refresh_key, False)

    def cache(func):
        cached = st.cache_data(func)
        if do_clear:
            cached.clear()
        return cached
    return cache


def _refresh_button(refresh_key, label='↻ Refresh'):
    """Render a per-tab refresh button that reloads this tab's data on next run."""
    if st.button(label, key=f'{refresh_key}_btn',
                 help="Reload this tab's data from the database"):
        st.session_state[refresh_key] = True
        st.rerun()


# Quick-reference descriptions of the pluginschema classes, shown at the bottom
# of each tab. Each entry: (one-line description, {method signature: purpose},
# {part-table name: purpose}).
SCHEMA_REFERENCE = {
    'WfieldParameters': (
        'Manual — analysis parameters for a widefield recording; one row per '
        '`wfield_analysis_id` (motion correction, SVD `k`, masks, atlas).',
        {'create_exclude_mask(widefieldkey)': 'stub for interactively building an exclude mask'},
        {},
    ),
    'WfieldStack': (
        'Computed — motion correction + SVD decomposition for a WfieldParameters set; '
        'run with `.populate()`. Stores U / SVT / mean_proj in an AnalysisFile.',
        {'open(key=None, use_corrected=True)': 'return an SVDStack (reconstructs frames on demand)',
         'load(key=None)': 'load the raw .npz arrays',
         'save_projection(proj, proj_name, key=None)': 'store a 2D projection',
         'make(key)': 'compute the stack (called by populate)',
         'delete(...)': 'delete rows and the linked AnalysisFile'},
        {'Projection': 'named 2D projections (`proj_name`, `proj`)'},
    ),
    'ImagingWindow': (
        'Manual — cranial-window geometry for a recording: circle parameters, '
        'resolution (mm/px), and the points sampled around the window.',
        {'apply_window_mask(image=None)':
            'fill outside the window circle with NaN — image, colour image, or N x H x W '
            'movie (defaults to the session mean projection)'},
        {},
    ),
    'ImagingReference': (
        'Manual — a reference widefield image for a subject (per `ref_num`); other '
        'recordings (e.g. 2P) are aligned to it.',
        {'overlay_projections_on_reference(ax=None, proj_name="mean", cell_seg_params=None, ...)':
            'overlay aligned 2P CellSegmentation projections + FOV boxes on the reference image'},
        {},
    ),
    'TwoPhotonReferenceAlignment': (
        'Manual — affine alignment of a TwoPhoton dataset to an ImagingReference '
        '(rotation / scale / ratio / transpose / origin / fov_offset).',
        {'get_transform(fw, fh)': 'return (M_fwd, transpose, fov_offset): 2P (col,row) → reference px',
         'apply_transform(image, output_shape=None)':
            'warp a 2P image or N x H x W movie into the reference image pixel space',
         'points_to_atlas(xy, atlas_transform=None, fov_dims=None, ...)':
            'map 2P (col,row) points to atlas mm (atlas transform auto-resolved if omitted)',
         'plot_fov_on_atlas(ax=None, atlas_transform=None, ...)':
            'plot the imaged FOV outline(s) in atlas mm over the region contours; '
            'iterates over all rows in the query and finds each atlas transform automatically'},
        {},
    ),
    'WidefieldAtlas': (
        'Manual — a stored Allen CCF atlas (region contours in mm, flat projection, '
        'bregma). One row per `atlas_name`, shared across recordings.',
        {'from_wfield(atlas_name)': 'load from local wfield reference files',
         'from_allensdk(atlas_name, structures=None, resolution=10, reference=None)':
            'build from the Allen SDK',
         'plot_atlas(ax=None, **kw)': 'show the flat projection (mm extent, bregma origin)',
         'plot_regions(acronyms=None, ax=None, labels=True, **kw)': 'plot region contours in mm',
         'load()': 'return (ccf_regions, projection, brain_outline)'},
        {},
    ),
    'WidefieldAtlasTransform': (
        'Manual — registration of a widefield session to a WidefieldAtlas (manual '
        'sliders or landmark pairs); per (session, atlas, `atlas_transform_id`).',
        {'get_transform()': '3×3 matrix mapping atlas mm → widefield pixels',
         'transform_regions(ccf_regions=None)': 'region contours warped to widefield pixels',
         'plot_regions(acronyms=None, ax=None, labels=True, **kw)':
            'plot warped region contours on the image',
         'load_reference()': 'return the atlas (ccf_regions, projection, outline)'},
        {},
    ),
    'WidefieldResponse': (
        'Manual — named analysis results tied to a recording (e.g. retinotopy), '
        'optionally with a movie file.',
        {'insert1(row, movie=None, **kw)': 'insert, optionally saving/uploading a movie .npz',
         'save_projection(proj, proj_name, key=None)': 'store a 2D response projection',
         'load_projections(key=None)': 'return {proj_name: image}',
         'load_movie(key=None)': 'load the stored movie .npz',
         'delete(..., keep_analysis=False)': 'delete rows and (optionally) the movie file'},
        {'Projection': 'named 2D images (`proj_name`, `proj`)'},
    ),
    'CellSegmentationAtlas': (
        'Computed — places each CellSegmentation.ROI in atlas mm and assigns a '
        'cortical region; joins CellSegmentation × TwoPhotonReferenceAlignment × '
        'WidefieldAtlasTransform. Run with `.populate()`.',
        {'make(key)': 'compute ROI atlas positions (called by populate)'},
        {'ROI': 'per ROI: `atlas_x`, `atlas_y`, `hemisphere`, `acronym`, `region_distance`'},
    ),
}


def _schema_reference(*names):
    """Render a collapsed quick-reference of the given pluginschema classes."""
    with st.expander('Schema reference (pluginschema)'):
        for name in names:
            entry = SCHEMA_REFERENCE.get(name)
            if entry is None:
                continue
            desc, methods, parts = entry
            st.markdown(f"**`{name}`** — {desc}")
            for sig, purpose in methods.items():
                st.markdown(f"- `{sig}` — {purpose}")
            for part, purpose in parts.items():
                st.markdown(f"- part `{name}.{part}` — {purpose}")
