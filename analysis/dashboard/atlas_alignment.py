import base64
import numpy as np
import pandas as pd
import streamlit as st

from .common import (_normalize_with_pct, _altair_image,
                     _tab_cache_factory, _refresh_button)


def _render_atlas_overlay(mean_proj, ccf_regions_dict, reference_point,
                          resolution, rotation=0., scale=1., ratio=1.,
                          circle_params=None, mirror=False):
    """Render atlas contours on mean_proj. Returns base64 PNG URL."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd_
    from ..utils import build_atlas_transform, transform_atlas_regions

    regions = pd_.DataFrame(ccf_regions_dict)
    M = build_atlas_transform(
        bregma_xy=reference_point,
        resolution=float(resolution),
        rotation=float(rotation),
        scale=float(scale),
        ratio=float(ratio),
        mirror=bool(mirror),
    )
    transformed = transform_atlas_regions(regions, M)

    mean_proj = np.squeeze(mean_proj)
    img_h, img_w = mean_proj.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5 * img_h / img_w), dpi=100, facecolor='k')
    ax.set_facecolor('k')
    is_rgb = mean_proj.ndim == 3 and mean_proj.shape[2] == 3
    if is_rgb:
        ax.imshow(mean_proj, origin='upper', aspect='auto')
    else:
        ax.imshow(mean_proj, cmap='gray', origin='upper', aspect='auto')
    for _, row in transformed.iterrows():
        ax.plot(np.asarray(row['left_x']),  np.asarray(row['left_y']),  '-', color='orange', lw=0.8)
        ax.plot(np.asarray(row['right_x']), np.asarray(row['right_y']), '-', color='orange', lw=0.8)
    ax.plot(reference_point[0], reference_point[1], 'r+', markersize=14, markeredgewidth=2)
    if circle_params is not None:
        import matplotlib.patches as mpatches
        xc, yc, r = circle_params
        ax.add_patch(mpatches.Circle((xc, yc), r, fill=False,
                                      edgecolor='black', linewidth=2))
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis('off')
    fig.tight_layout(pad=0.05)

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='k')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _render_img_with_landmarks(arr, points, dot_color=(0, 140, 255)):
    """Render float/uint image as base64 PNG with coloured landmark dots."""
    from PIL import Image as PILImage, ImageDraw
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr if arr.dtype == np.uint8 else _normalize_with_pct(arr).reshape(arr.shape[:2] + (3,)).astype(np.uint8)
    else:
        norm = _normalize_with_pct(arr.astype(float))
        rgb = np.stack([norm] * 3, axis=-1).astype(np.uint8)
    pil = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    for x, y in points:
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=dot_color, outline=(255, 255, 255), width=1)
    import io
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _altair_clickable(url, img_w, img_h, width=380, title=''):
    """Altair image chart with a transparent grid overlay for click detection."""
    import altair as alt
    x_sc = alt.Scale(domain=[0, img_w])
    y_sc = alt.Scale(domain=[img_h, 0])
    step = max(4, min(img_w, img_h) // 64)
    xs, ys = np.arange(0, img_w, step, dtype=float), np.arange(0, img_h, step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    height = max(200, int(width * img_h / img_w))
    img_layer = (
        alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': img_w, 'y2': img_h, 'url': url}]))
        .mark_image(aspect=False)
        .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                y=alt.Y('y:Q', scale=y_sc, axis=None),
                x2='x2:Q', y2='y2:Q', url='url:N',
                tooltip=alt.value(None))
    )
    click_sel = alt.selection_point(name='pt_click', on='click',
                                    nearest=True, encodings=['x', 'y'])
    grid_layer = (
        alt.Chart(pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()}))
        .mark_point(opacity=0, size=step * step * 4)
        .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                y=alt.Y('y:Q', scale=y_sc, axis=None))
        .add_params(click_sel)
    )
    return (alt.layer(img_layer, grid_layer)
            .properties(width=width, height=height, title=title)
            .interactive())


def _extract_click(event):
    """Return (x, y) from an Altair on_select event, or None."""
    sel = (event.selection or {}).get('pt_click', [])
    if sel:
        x, y = sel[0].get('x'), sel[0].get('y')
        if x is not None and y is not None:
            return float(x), float(y)
    return None


@st.fragment
def _atlas_alignment_tab(schema, WfieldParameters, WfieldStack):
    from ..pluginschema import WidefieldAtlas, WidefieldAtlasTransform, WidefieldResponse

    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return

    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    cache = _tab_cache_factory('refresh_atlas')
    _refresh_button('refresh_atlas')

    @cache
    def get_atlases():
        return list((WidefieldAtlas()).fetch('atlas_name'))

    @cache
    def get_stack_ids_at(subject_name, session_name, dataset_name):
        return list((WfieldStack & dict(subject_name=subject_name,
                                        session_name=session_name,
                                        dataset_name=dataset_name)
                     ).fetch('wfield_analysis_id'))

    @cache
    def get_mean_proj_at(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        mp = np.squeeze((WfieldStack & key).fetch1('mean_proj'))
        return (mp[0] if mp.ndim == 3 else mp).astype(np.float32)

    @cache
    def get_response_projections(subject_name, session_name, dataset_name):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name)
        rows = (WidefieldResponse.Projection & key).fetch(
            'stim_name', 'proj_name', as_dict=True)
        return [(r['stim_name'], r['proj_name']) for r in rows]

    @cache
    def get_response_proj_image(subject_name, session_name, dataset_name,
                                response_name, proj_name):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, stim_name=response_name,
                   proj_name=proj_name)
        arr = np.squeeze(np.asarray((WidefieldResponse.Projection & key).fetch1('proj')))
        return arr  # preserve original dtype (uint8 RGB or float)

    @cache
    def get_atlas_row(atlas_name):
        return (WidefieldAtlas & dict(atlas_name=atlas_name)).fetch1()

    @cache
    def get_existing_transforms(subject_name, session_name, dataset_name, atlas_name):
        return (WidefieldAtlasTransform & dict(
            subject_name=subject_name, session_name=session_name,
            dataset_name=dataset_name, atlas_name=atlas_name,
        )).fetch(as_dict=True)

    @cache
    def get_imaging_window(subject_name, session_name, dataset_name):
        from ..pluginschema import ImagingWindow
        rows = (ImagingWindow & dict(subject_name=subject_name,
                                     session_name=session_name,
                                     dataset_name=dataset_name)).fetch(as_dict=True)
        return rows[0] if rows else None

    atlas_names = get_atlases()
    if not atlas_names:
        st.warning('No WidefieldAtlas entries. '
                   'Create one with `WidefieldAtlas().from_wfield(name)` '
                   'or `.from_allensdk(name)`.')
        return

    stack_ids = get_stack_ids_at(subject_name, session_name, dataset_name)

    hc1, hc2, hc3 = st.columns(3)
    atlas_name   = hc1.selectbox('Atlas', atlas_names, key='at_atlas')
    analysis_id  = hc2.selectbox('WfieldStack', stack_ids if stack_ids else ['—'],
                                  disabled=not stack_ids, key='at_aid')
    existing     = get_existing_transforms(subject_name, session_name, dataset_name, atlas_name)
    # Default to the latest saved transform so its parameters load on open;
    # bump the number to create a new one.
    default_id   = max((r['atlas_transform_id'] for r in existing), default=0) if existing else 0
    transform_id = int(hc3.number_input('atlas_transform_id', value=default_id,
                                         step=1, min_value=0, key='at_tid'))
    if existing:
        disp_cols = [k for k in existing[0]
                     if k not in ('landmarks', 'landmarks_match',
                                  'transform_matrix', 'transform_matrix_inverse')]
        with st.expander(f'{len(existing)} existing transform(s)'):
            st.dataframe(pd.DataFrame(existing)[disp_cols], hide_index=True)

    # Reference image: mean_proj (if stack available) or any WidefieldResponse projection
    _DEFAULT_IMG = 'mean_proj (WfieldStack)'
    response_projs = get_response_projections(subject_name, session_name, dataset_name)
    img_options = ([_DEFAULT_IMG] if stack_ids else []) + [
        f'{rn} → {pn}' for rn, pn in response_projs]
    if not img_options:
        st.info('No reference image available. Run a WfieldStack analysis or add a '
                'WidefieldResponse projection first.')
        return
    sel_img = st.selectbox('Reference image', img_options, key='at_ref_img',
                            help='Use a WidefieldResponse projection (e.g. retinotopy phase map) '
                                 'as the background for atlas alignment.')

    with st.spinner('Loading…'):
        if sel_img == _DEFAULT_IMG:
            ref_image = get_mean_proj_at(subject_name, session_name, dataset_name,
                                         int(analysis_id))
        else:
            rn, pn = response_projs[img_options.index(sel_img) - (1 if stack_ids else 0)]
            ref_image = get_response_proj_image(subject_name, session_name, dataset_name,
                                                rn, pn)
        atlas_row      = get_atlas_row(atlas_name)
        imaging_window = get_imaging_window(subject_name, session_name, dataset_name)

    manual_tab, landmarks_tab = st.tabs(['Manual', 'Landmarks'])
    with manual_tab:
        _atlas_manual_subtab(sel_key, atlas_name, transform_id,
                              ref_image, atlas_row, imaging_window,
                              WidefieldAtlasTransform, get_existing_transforms)
    with landmarks_tab:
        _atlas_landmarks_subtab(sel_key, atlas_name, transform_id,
                                ref_image, atlas_row,
                                WidefieldAtlasTransform, get_existing_transforms)


def _atlas_manual_subtab(sel_key, atlas_name, transform_id,
                          mean_proj, atlas_row, imaging_window,
                          WidefieldAtlasTransform, get_existing_transforms):
    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    img_h, img_w = mean_proj.shape[:2]
    # transform_id is part of the scope so picking a saved transform reloads its parameters
    scope = f'{subject_name}|{session_name}|{dataset_name}|{atlas_name}|{transform_id}'

    # Drain pending bregma click into slider state before widgets render
    for k in ('rx', 'ry'):
        pkey = f'at_m_{scope}_{k}_pend'
        if pkey in st.session_state:
            st.session_state[f'at_m_{scope}_{k}'] = st.session_state.pop(pkey)

    # Default resolution: from ImagingWindow if available, else 0.025
    iw_res = float(imaging_window['resolution']) if (
        imaging_window and imaging_window.get('resolution') is not None) else 0.025
    iw_circle = (np.asarray(imaging_window['circle_parameters']).ravel()
                 if imaging_window and imaging_window.get('circle_parameters') is not None
                 else None)

    # Pre-populate from DB on first load
    if f'at_m_{scope}_init' not in st.session_state:
        saved = next((r for r in get_existing_transforms(
                          subject_name, session_name, dataset_name, atlas_name)
                      if r['atlas_transform_id'] == transform_id
                      and r['transform_type'] == 'manual'), None)
        if saved:
            rp = np.asarray(saved['reference_point']).ravel()
            st.session_state[f'at_m_{scope}_rx']    = float(rp[0])
            st.session_state[f'at_m_{scope}_ry']    = float(rp[1])
            st.session_state[f'at_m_{scope}_rot']    = float(saved.get('rotation') or 0.)
            st.session_state[f'at_m_{scope}_scale']  = float(saved.get('scale')    or 1.)
            st.session_state[f'at_m_{scope}_ratio']  = float(saved.get('ratio')    or 1.)
            st.session_state[f'at_m_{scope}_res']    = float(saved.get('resolution') or iw_res)
            st.session_state[f'at_m_{scope}_mirror'] = bool(saved.get('mirror') or False)
        else:
            st.session_state[f'at_m_{scope}_res'] = iw_res
        st.session_state[f'at_m_{scope}_init'] = True

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
    resolution = c1.number_input('Resolution (mm/px)', value=iw_res, step=0.001,
                                  format='%.4f', key=f'at_m_{scope}_res',
                                  help='mm per widefield pixel')
    rotation   = c2.slider('Rotation (°)', -180., 180., 0., step=0.5,
                             key=f'at_m_{scope}_rot')
    scale      = c3.slider('Scale',          0.1,   5.0, 1., step=0.05,
                             key=f'at_m_{scope}_scale')
    ratio      = c4.slider('X/Y ratio',      0.5,   2.0, 1., step=0.01,
                             key=f'at_m_{scope}_ratio')
    mirror     = c5.checkbox('Mirror X', value=False, key=f'at_m_{scope}_mirror',
                              help='Flip the atlas left/right (for imaging setups where the x-axis is reversed)')

    ref_x = float(st.session_state.get(f'at_m_{scope}_rx', img_w / 2))
    ref_y = float(st.session_state.get(f'at_m_{scope}_ry', img_h / 2))
    st.caption(f'Bregma: ({ref_x:.0f}, {ref_y:.0f}) px — click the image to reposition')

    url = _render_atlas_overlay(mean_proj, atlas_row['ccf_regions'],
                                 [ref_x, ref_y], resolution, rotation, scale, ratio,
                                 circle_params=iw_circle, mirror=mirror)
    chart = _altair_clickable(url, img_w, img_h, width=460,
                               title='Click to set bregma · atlas contours overlaid')
    last_key = f'at_m_{scope}_last'
    event = st.altair_chart(chart, on_select='rerun',
                             key=f'at_m_{scope}_chart', width='content')
    click = _extract_click(event)
    if click and list(click) != st.session_state.get(last_key):
        st.session_state[last_key] = list(click)
        st.session_state[f'at_m_{scope}_rx_pend'] = click[0]
        st.session_state[f'at_m_{scope}_ry_pend'] = click[1]
        st.rerun(scope='fragment')

    if st.button('Save manual transform', type='primary', key=f'at_m_{scope}_save'):
        WidefieldAtlasTransform.insert1(dict(
            **sel_key,
            atlas_name=atlas_name,
            atlas_transform_id=transform_id,
            transform_type='manual',
            reference_point=[ref_x, ref_y],
            resolution=float(resolution),
            rotation=float(rotation),
            scale=float(scale),
            ratio=float(ratio),
            mirror=int(mirror),
        ), replace=True)
        get_existing_transforms.clear()
        st.success(f'Saved manual transform id={transform_id}.')
        st.rerun(scope='fragment')


def _atlas_landmarks_subtab(sel_key, atlas_name, transform_id,
                              mean_proj, atlas_row,
                              WidefieldAtlasTransform, get_existing_transforms):
    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    img_h, img_w = mean_proj.shape[:2]
    proj         = atlas_row['projection']
    atlas_res    = float(atlas_row['resolution'])          # mm / atlas pixel
    ref_row_a, ref_col_a = atlas_row['reference_point']   # [row, col]
    atlas_h, atlas_w = proj.shape
    scope = f'{subject_name}|{session_name}|{dataset_name}|{atlas_name}|{transform_id}'

    wf_key   = f'at_l_{scope}_wf'
    at_key   = f'at_l_{scope}_at'
    wf_last  = f'at_l_{scope}_wf_last'
    at_last  = f'at_l_{scope}_at_last'

    # On first load of this transform, pre-populate points from a saved landmark transform.
    if f'{at_key}_init' not in st.session_state:
        st.session_state[f'{at_key}_init'] = True
        saved = next((r for r in get_existing_transforms(
                          subject_name, session_name, dataset_name, atlas_name)
                      if r['atlas_transform_id'] == transform_id
                      and r['transform_type'] == 'landmarks'), None)
        wf_init, at_init = [], []
        if saved and saved.get('landmarks') is not None \
                and saved.get('landmarks_match') is not None:
            lm  = pd.DataFrame(saved['landmarks'])        # atlas-space, mm from bregma (x, y)
            lmm = pd.DataFrame(saved['landmarks_match'])  # widefield pixels (x=col, y=row)
            at_init = [[float(x) / atlas_res + ref_col_a, float(y) / atlas_res + ref_row_a]
                       for x, y in zip(lm['x'], lm['y'])]
            wf_init = [[float(x), float(y)] for x, y in zip(lmm['x'], lmm['y'])]
        st.session_state[wf_key] = wf_init
        st.session_state[at_key] = at_init
    st.session_state.setdefault(wf_key, [])
    st.session_state.setdefault(at_key, [])

    wf_pts    = st.session_state[wf_key]    # [[col, row], …] in widefield pixels
    atlas_pts = st.session_state[at_key]    # [[col, row], …] in atlas pixels
    n_wf, n_at = len(wf_pts), len(atlas_pts)
    n_pairs   = min(n_wf, n_at)

    st.caption(f'Image landmarks: **{n_wf}**  ·  Atlas landmarks: **{n_at}**  ·  '
               f'Pairs: **{n_pairs}** (need ≥ 3)')

    lm_col1, lm_col2 = st.columns(2)

    with lm_col1:
        st.write('**Widefield image** — click to add landmark')
        url_wf = _render_img_with_landmarks(mean_proj, wf_pts, dot_color=(0, 140, 255))
        ev_wf = st.altair_chart(
            _altair_clickable(url_wf, img_w, img_h, width=340,
                              title='Widefield landmarks (blue)'),
            on_select='rerun', key=f'at_l_{scope}_wf_chart', width='content')
        click_wf = _extract_click(ev_wf)
        if click_wf and list(click_wf) != st.session_state.get(wf_last):
            st.session_state[wf_last] = list(click_wf)
            wf_pts.append(list(click_wf))
            st.session_state[wf_key] = wf_pts
            st.rerun(scope='fragment')

    with lm_col2:
        st.write('**Atlas projection** — click to add landmark')
        url_at = _render_img_with_landmarks(proj, atlas_pts, dot_color=(255, 140, 0))
        ev_at = st.altair_chart(
            _altair_clickable(url_at, atlas_w, atlas_h, width=340,
                              title='Atlas landmarks (orange)'),
            on_select='rerun', key=f'at_l_{scope}_at_chart', width='content')
        click_at = _extract_click(ev_at)
        if click_at and list(click_at) != st.session_state.get(at_last):
            st.session_state[at_last] = list(click_at)
            atlas_pts.append(list(click_at))
            st.session_state[at_key] = atlas_pts
            st.rerun(scope='fragment')

    bc1, bc2, bc3, bc4 = st.columns(4)
    if bc1.button('Clear image pts',  key=f'at_l_{scope}_clwf'):
        st.session_state[wf_key] = []; st.session_state.pop(wf_last, None)
        st.rerun(scope='fragment')
    if bc2.button('Undo image pt',    key=f'at_l_{scope}_unwf',  disabled=n_wf == 0):
        st.session_state[wf_key] = wf_pts[:-1]; st.rerun(scope='fragment')
    if bc3.button('Clear atlas pts',  key=f'at_l_{scope}_clat'):
        st.session_state[at_key] = []; st.session_state.pop(at_last, None)
        st.rerun(scope='fragment')
    if bc4.button('Undo atlas pt',    key=f'at_l_{scope}_unat',  disabled=n_at == 0):
        st.session_state[at_key] = atlas_pts[:-1]; st.rerun(scope='fragment')

    if n_pairs > 0:
        pairs_df = pd.DataFrame({
            '#':           range(1, n_pairs + 1),
            'wf_col':      [p[0] for p in wf_pts[:n_pairs]],
            'wf_row':      [p[1] for p in wf_pts[:n_pairs]],
            'atlas_col':   [p[0] for p in atlas_pts[:n_pairs]],
            'atlas_row':   [p[1] for p in atlas_pts[:n_pairs]],
            'atlas_x_mm':  [(p[0] - ref_col_a) * atlas_res for p in atlas_pts[:n_pairs]],
            'atlas_y_mm':  [(p[1] - ref_row_a) * atlas_res for p in atlas_pts[:n_pairs]],
        })
        st.dataframe(pairs_df, hide_index=True)

    if n_pairs < 3:
        st.info(f'Add at least 3 matched pairs to preview the transform ({n_pairs} so far).')
        return

    # Build transform and render live preview
    try:
        from wfield import allen_transform_from_landmarks, allen_landmarks_to_image_space
        from ..utils import transform_atlas_regions
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        lm_df = pd.DataFrame({
            'x':     [(p[0] - ref_col_a) * atlas_res for p in atlas_pts[:n_pairs]],
            'y':     [(p[1] - ref_row_a) * atlas_res for p in atlas_pts[:n_pairs]],
            'name':  [f'lm{i}' for i in range(n_pairs)],
            'color': ['k'] * n_pairs,
        })
        lm_match_df = pd.DataFrame({
            'x':     [p[0] for p in wf_pts[:n_pairs]],
            'y':     [p[1] for p in wf_pts[:n_pairs]],
            'name':  [f'lm{i}' for i in range(n_pairs)],
            'color': ['k'] * n_pairs,
        })
        ref_offset = np.array([ref_col_a, ref_row_a], dtype=float)  # [x=col, y=row]
        lm_im  = allen_landmarks_to_image_space(lm_df.copy(), ref_offset, atlas_res)
        M_lm   = allen_transform_from_landmarks(lm_im, lm_match_df)
        T_res  = np.array([[1/atlas_res, 0, ref_col_a],
                            [0, 1/atlas_res, ref_row_a],
                            [0, 0, 1]], dtype=float)
        M_full = M_lm.params @ T_res
        transformed = transform_atlas_regions(pd.DataFrame(atlas_row['ccf_regions']), M_full)

        import io as _io
        fig, ax = plt.subplots(figsize=(5, 5 * img_h / img_w), dpi=100, facecolor='k')
        ax.set_facecolor('k')
        ax.imshow(mean_proj, cmap='gray', origin='upper', aspect='auto')
        for _, row in transformed.iterrows():
            rgb = row.get('allen_rgb')
            c = [v / 255 for v in rgb] if rgb is not None else 'cyan'
            ax.plot(np.asarray(row['left_x']),  np.asarray(row['left_y']),  '-', color=c, lw=0.8)
            ax.plot(np.asarray(row['right_x']), np.asarray(row['right_y']), '-', color=c, lw=0.8)
        for p in wf_pts[:n_pairs]:
            ax.plot(p[0], p[1], 'bo', markersize=8, markeredgecolor='w', markeredgewidth=1)
        ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.axis('off')
        fig.tight_layout(pad=0.05)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='k')
        plt.close(fig)
        buf.seek(0)
        prev_url = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()
        st.altair_chart(_altair_image(prev_url, img_w, img_h,
                                      title='Transform preview', width=460),
                        width='content')

        if st.button('Save landmark transform', type='primary',
                      key=f'at_l_{scope}_save'):
            WidefieldAtlasTransform.insert1(dict(
                **sel_key,
                atlas_name=atlas_name,
                atlas_transform_id=transform_id,
                transform_type='landmarks',
                landmarks=lm_df.to_dict(orient='list'),
                landmarks_match=lm_match_df.to_dict(orient='list'),
            ), replace=True)
            get_existing_transforms.clear()
            st.success(f'Saved landmark transform id={transform_id} '
                       f'with {n_pairs} pairs.')
            st.rerun(scope='fragment')

    except Exception as exc:
        st.error(f'Transform preview failed: {exc}')
