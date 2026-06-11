import numpy as np
import pandas as pd
import streamlit as st

from .common import _normalize_image, _to_base64, _tab_cache_factory, _refresh_button


@st.fragment
def _window_mask_tab(schema, WfieldParameters, WfieldStack):
    from ..pluginschema import ImagingWindow
    import altair as alt
    from skimage.measure import CircleModel

    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return

    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    widefield_key = dict(subject_name=subject_name,
                         session_name=session_name,
                         dataset_name=dataset_name)
    cache = _tab_cache_factory('refresh_window')
    _refresh_button('refresh_window')

    @cache
    def get_imaging_window(subject_name, session_name, dataset_name):
        rows = (ImagingWindow & widefield_key).fetch(as_dict=True)
        return rows[0] if rows else None

    existing = get_imaging_window(subject_name, session_name, dataset_name)
    if existing:
        cp = existing.get('circle_parameters')
        res = existing.get('resolution')
        info_parts = []
        if cp is not None:
            cp = np.asarray(cp)
            info_parts.append(f'center=({cp[0]:.0f}, {cp[1]:.0f})  radius={cp[2]:.1f} px')
        if res is not None:
            info_parts.append(f'resolution={res:.4f} mm/px')
        st.success('ImagingWindow entry exists — ' + ('  |  '.join(info_parts) if info_parts else ''))


    @cache
    def get_stack_ids(subject_name, session_name, dataset_name):
        return list((WfieldStack & widefield_key).fetch('wfield_analysis_id', as_dict=False))

    analysis_ids = get_stack_ids(subject_name, session_name, dataset_name)

    col_sel, col_win = st.columns([2, 1])
    windowsize = col_win.number_input('Window size (mm)', value=5.0, step=0.5, min_value=0.1)

    if analysis_ids:
        analysis_id = col_sel.selectbox('Reference image (wfield_analysis_id)', analysis_ids,
                                        key='wf_mask_aid')
    else:
        col_sel.info('No WfieldStack results — cannot display a reference image.')
        return

    @cache
    def get_image(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        mp = (WfieldStack & key).fetch1('mean_proj')
        arr = np.squeeze(mp)
        if arr.ndim == 3:
            arr = arr[0]
        return _normalize_image(arr)

    with st.spinner('Loading image…'):
        arr = get_image(subject_name, session_name, dataset_name, int(analysis_id))

    h, w = arr.shape


    scope    = f'{subject_name}|{session_name}|{dataset_name}'
    pts_key  = f'wf_mask_pts_{scope}'
    last_key = f'wf_mask_last_{scope}'
    if pts_key not in st.session_state:
        # Pre-populate from existing DB entry so the circle is shown immediately
        if existing and existing.get('points') is not None:
            st.session_state[pts_key] = np.asarray(existing['points']).tolist()
        else:
            st.session_state[pts_key] = []

    points = st.session_state[pts_key]

    # Downsample PNG for display — domain stays [0,w]×[0,h]
    _display_size = 256
    _scale = _display_size / max(h, w)
    if _scale < 1.0:
        dh, dw = max(1, int(h * _scale)), max(1, int(w * _scale))
        display_arr = arr[::max(1, h // dh), ::max(1, w // dw)]
    else:
        display_arr = arr
    # Fit circle whenever we have ≥ 3 points
    circlepar = None
    if len(points) >= 3:
        cmod = CircleModel()
        cmod.estimate(np.array(points, dtype=float))
        circlepar = cmod.params  # (xc, yc, r)

    # Bake circle outline into display image using PIL
    dh_actual, dw_actual = display_arr.shape
    scale_x_d = dw_actual / w
    scale_y_d = dh_actual / h
    disp_rgb = np.stack([display_arr] * 3, axis=-1)
    if circlepar is not None:
        from PIL import Image as PILImage, ImageDraw
        xc_d = circlepar[0] * scale_x_d
        yc_d = circlepar[1] * scale_y_d
        r_d = circlepar[2] * (scale_x_d + scale_y_d) / 2
        pil_d = PILImage.fromarray(disp_rgb)
        ImageDraw.Draw(pil_d).ellipse(
            [xc_d - r_d, yc_d - r_d, xc_d + r_d, yc_d + r_d],
            outline=(255, 60, 60), width=2)
        disp_rgb = np.array(pil_d)
    url = _to_base64(disp_rgb)


    x_scale = alt.Scale(domain=[0, w])
    y_scale = alt.Scale(domain=[h, 0])

    img_layer = (
        alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': w, 'y2': h, 'url': url}]))
        .mark_image(aspect=False)
        .encode(
            x=alt.X('x:Q', scale=x_scale, axis=None),
            y=alt.Y('y:Q', scale=y_scale, axis=None),
            x2='x2:Q', y2='y2:Q', url='url:N',
            tooltip=alt.value(None),
        )
    )

    step = max(4, min(w, h) // 64)
    xs, ys = np.arange(0, w, step, dtype=float), np.arange(0, h, step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    grid_df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})
    click_sel = alt.selection_point(name='img_click', on='click', nearest=True,
                                    encodings=['x', 'y'])
    grid_layer = (
        alt.Chart(grid_df)
        .mark_point(opacity=0, size=step * step * 4)
        .encode(x=alt.X('x:Q', scale=x_scale, axis=None),
                y=alt.Y('y:Q', scale=y_scale, axis=None))
        .add_params(click_sel)
    )

    layers = [img_layer]

    if points:
        pts_df = pd.DataFrame(points, columns=['x', 'y'])
        layers.append(
            alt.Chart(pts_df).mark_point(color='lime', size=80, filled=True,
                                         strokeWidth=1.5, stroke='black').encode(
                x=alt.X('x:Q', scale=x_scale, axis=None),
                y=alt.Y('y:Q', scale=y_scale, axis=None),
                tooltip=alt.value(None),
            )
        )

    layers.append(grid_layer)
    chart = (alt.layer(*layers)
             .properties(width=400, height=400,
                         title='Click the boundary of the imaging window')
             .interactive())

    event = st.altair_chart(chart, on_select='rerun', key='wf_circle_chart', width='content')

    sel = (event.selection or {}).get('img_click', [])
    if sel:
        x_click = sel[0].get('x')
        y_click = sel[0].get('y')
        if x_click is not None and y_click is not None:
            new_pt = [float(x_click), float(y_click)]
            if new_pt != st.session_state.get(last_key):
                st.session_state[last_key] = new_pt
                points.append(new_pt)
                st.session_state[pts_key] = points
                st.rerun()


    c1, c2, c3, c4, c5 = st.columns(5)

    if c1.button('Clear all', key='wf_mask_clear'):
        st.session_state[pts_key] = []
        st.session_state.pop(last_key, None)
        st.rerun()

    if c2.button('Undo last', key='wf_mask_undo', disabled=len(points) == 0):
        st.session_state[pts_key] = points[:-1]
        st.session_state.pop(last_key, None)
        st.rerun()

    if circlepar is not None:
        xc, yc, r = circlepar
        resolution = windowsize / (2 * r) if r > 0 else None
        c3.metric('Center (col, row)', f'({xc:.0f}, {yc:.0f})')
        c4.metric('Radius (px)', f'{r:.1f}')
        if resolution:
            c5.metric('Resolution (mm/px)', f'{resolution:.4f}')

        if st.button('Save to ImagingWindow', type='primary', key='wf_mask_save'):
            ImagingWindow.insert1(dict(
                **widefield_key,
                window_size=float(windowsize),
                resolution=float(resolution) if resolution else None,
                points=np.array(points, dtype=float),
                circle_parameters=np.array([xc, yc, r], dtype=float),
            ), replace=True)
            get_imaging_window.clear()
            st.success(f'Saved ImagingWindow — center=({xc:.0f}, {yc:.0f}), '
                       f'radius={r:.1f} px, resolution={resolution:.4f} mm/px.')
            st.rerun()
    else:
        st.caption(f'Click at least 3 points on the window boundary ({len(points)} so far).')

    if existing:
        with st.expander('Delete ImagingWindow entry'):
            st.warning('Permanently removes this ImagingWindow from the database.')
            if st.button('Confirm delete', type='primary', key='wf_mask_delete'):
                (ImagingWindow & widefield_key).delete(safemode=False)
                get_imaging_window.clear()
                st.session_state[pts_key] = []
                st.session_state.pop(last_key, None)
                st.rerun()
